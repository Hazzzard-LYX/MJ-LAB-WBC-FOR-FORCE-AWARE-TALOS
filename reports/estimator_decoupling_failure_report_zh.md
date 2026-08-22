# TALOS 自由托盘任务中 Payload Estimator 解耦训练失败报告

日期：2026-08-22

项目：MJ-LAB WBC for Force-Aware TALOS

相关 PR：[PR #8 — Estimate payload state during contact-only TALOS grasping](https://github.com/Hazzzard-LYX/MJ-LAB-WBC-FOR-FORCE-AWARE-TALOS/pull/8)

相关提交：`7bcd1c8f220d`（联合 estimator）与 `a4ee496d73e2`（独立 estimator）

## 1. 摘要

我们希望 TALOS 在双手仅依靠刚性接触和摩擦抓住自由托盘的情况下，根据实机可获得的传感器历史估计：

1. 托盘上重物的质量；
2. 重物在托盘坐标系中的三维位置。

最初采用 PPO 与 estimator 联合训练。联合模型完成 30,000 iterations 后，重量 MAE 约为 `6.78 kg`，与 `2.5–30 kg` 均匀分布下预测常数中位数的理论 MAE `6.875 kg` 基本相同，说明 estimator 几乎没有提取出有效重量信息。

随后我们将 estimator 从 PPO 优化中分离，冻结已有行为策略，用在线 rollout 产生监督数据，并使用独立 AdamW 优化器训练 estimator。独立训练完成 50,000 steps、约 51.2 million 个时序样本后，最终验证重量 MAE 为 `6.675 kg`，固定验证环境上的常数 baseline 为 `7.446 kg`；最终位置 MAE 为 `0.281 m`。虽然重量指标略优于常数预测，但改进较小且验证曲线不稳定，尚不能认为解耦训练成功。

核心判断是：**优化器和梯度路径已经解耦，但数据分布并未解耦。独立 estimator 的 rollout 仍由旧的“联合 estimator + actor”策略产生。**旧 estimator 的低质量预测会间接决定机器人动作和系统激励，新的 estimator 只能被动学习这一受偏置的行为分布。

## 2. 任务与物理设置

- 机器人：PAL Robotics TALOS。
- 托盘：独立六自由度刚体，与机器人之间没有 weld/fixed joint。
- 抓握：双手指 articulation、被动连杆和 fingertip mimic coupling；依靠刚性接触和高摩擦抓握。
- 重物：托盘上的自由刚体。
- 训练重量范围：`2.5–30 kg`。
- 目标状态：

  \[
  s=[m,x_T,y_T,z_T]
  \]

  其中 \(m\) 为重量，\((x_T,y_T,z_T)\) 为重物在托盘坐标系中的位置。

- 实机约束：actor 和 critic 不应直接依赖不可靠的关节 torque，也不能获得 simulator 中的真实重量与位置；torque 仅允许作为 estimator 的输入。

## 3. 联合 estimator 版本

### 3.1 信息流

```text
带噪硬件传感器历史（8帧）
              │
              ▼
     联合 Payload Estimator
              │
       预测重量与托盘内位置
              │
              ▼
        Actor 与 Critic
              │
              ▼
             动作
```

信息隔离规则：

- actor 不直接接收 joint torque；
- critic 不直接接收 joint torque；
- actor/critic 不接收真实 payload mass/position；
- estimator 接收带噪 joint torque、proprioception 和双手腕 F/T 历史；
- simulator 真值只进入 estimator 的监督损失；
- estimator、actor、critic 在同一个 PPO 更新过程中被优化。

### 3.2 联合训练结果

- Slurm 作业：`132061`；
- 并行环境：1536；
- PPO iterations：30,000；
- 最终 checkpoint：`model_29999.pt`；
- 重量 MAE：约 `6.78 kg`；
- 理论常数中位数 baseline：`6.875 kg`。

联合 estimator 的梯度同时受到策略优化、价值学习和辅助监督目标影响。结果没有明显优于不观察传感器、始终预测中位重量的常数模型，因此我们尝试将 estimator 独立训练。

## 4. 独立 estimator 版本

### 4.1 实际数据流

独立训练使用上述联合模型的最终 checkpoint 作为冻结行为策略：

```text
传感器历史（8帧）
       │
       ▼
旧联合 estimator（冻结）
       │
       ▼
旧 actor（冻结）──────────────┐
       │                      │
       ▼                      │ 决定采样状态分布
     动作                     │
       │                      │
       ▼                      │
机器人/托盘在线 rollout ◄─────┘
       │
       ├── 32帧带噪传感器历史
       └── simulator 重量/位置标签
                    │
                    ▼
          新独立 estimator
          （唯一被优化的网络）
```

因此：

- 新 estimator 不读取旧 estimator 的输出；
- 新 estimator 的梯度不会进入旧 actor；
- 新 estimator 的预测不影响 rollout 动作；
- 但是 rollout 的动作和状态覆盖由旧联合 estimator 策略决定。

这属于**梯度解耦、行为分布未解耦**。

### 4.2 Rollout 来源

冻结策略 checkpoint：

```text
/home/stud_yuxuan1/IAS_Workspace/logs/mjlab-training/7bcd1c8f220d/132061/
talos_contact_grasp_tray_random_mass_state_estimator_h8/
2026-08-18_03-10-20_7bcd1c8-free-contact-state-estimator-1536env/
model_29999.pt
```

训练器只加载 actor：

```python
runner.load(..., load_cfg={"actor": True})
behavior_policy = runner.get_inference_policy(...)
```

critic 不参与数据采集。

### 4.3 Estimator 输入

每帧共 144 维：

- base linear velocity；
- base angular velocity；
- projected gravity；
- joint position；
- joint velocity；
- previous action；
- IMU linear acceleration；
- joint torque sensors；
- left/right wrist force；
- left/right wrist torque。

独立 estimator 使用 32 帧历史，展平后输入维数为：

\[
144\times32=4608
\]

输入 observation corruption 保持开启，因此训练数据包含仿真噪声和延迟。

### 4.4 网络与输出约束

```text
4608 → 512 → 256 → 128 → 4
```

- 激活函数：ELU；
- 可训练参数约 2.52 million；
- 输入：Empirical Normalization；
- 重量输出：sigmoid 后映射到 `[2.5, 30.0] kg`；
- 位置输出：tanh 后映射到：
  - x：`[-0.38, 0.38] m`
  - y：`[-0.49, 0.49] m`
  - z：`[-0.05, 0.40] m`

直接后果：模型从结构上不可能输出超过 `30 kg` 的重量。因此之前 32–40 kg OOD 测试中的饱和并不是单纯的训练失败，而是输出参数化带来的硬限制。

### 4.5 损失与优化

重量与位置分别归一化，并分别计算 Smooth-L1：

\[
L=L_{mass}+L_{position}
\]

- `mass_loss_weight = 1.0`；
- `position_loss_weight = 1.0`；
- Smooth-L1 beta：`0.05`；
- 优化器：AdamW；
- learning rate：`3e-4`；
- weight decay：`1e-5`；
- gradient clipping：`5.0`。

### 4.6 在线采样配置

- GPU：RTX 3080；
- 环境数量：1536；
- 固定训练环境：1229；
- 固定验证环境：307；
- batch size：1024；
- optimizer steps：50,000；
- 日志间隔：100 steps；
- checkpoint 间隔：2500 steps；
- seed：42；
- 25% 环境执行 standing command；
- `lin_vel_x`：`[-0.8, 1.2] m/s`；
- `lin_vel_y`：`[-0.4, 0.4] m/s`；
- `ang_vel_z`：`[-0.5, 0.5] rad/s`。

每一步先从当前训练环境快照中选择最多 1024 个环境训练一次 estimator，然后冻结 policy 推进一步仿真。日志中的 51.2 million samples 是 `50,000 × 1024`，但这些样本具有很强的时间相关性，不能视为 51.2 million 个独立监督样本。

## 5. 定量结果

正式作业：`132896`

运行时间：约 1 小时 10 分钟

退出状态：`COMPLETED (0:0)`

| Step | Train mass MAE | Validation mass MAE | Validation position MAE | Validation loss |
|---:|---:|---:|---:|---:|
| 1 | 6.585 kg | 7.626 kg | 0.096 m | 0.357462 |
| 100 | 5.692 kg | **6.134 kg** | 0.167 m | **0.286216** |
| 1,000 | 5.933 kg | 7.294 kg | 0.101 m | 0.299664 |
| 5,000 | 4.969 kg | 7.300 kg | 0.263 m | 0.356651 |
| 10,000 | 5.228 kg | 6.770 kg | 0.251 m | 0.335869 |
| 20,000 | 5.404 kg | 7.124 kg | 0.269 m | 0.341714 |
| 30,000 | 5.716 kg | 7.221 kg | 0.271 m | 0.348657 |
| 40,000 | 5.398 kg | 6.638 kg | 0.250 m | 0.324303 |
| 50,000 | 5.560 kg | 6.675 kg | 0.281 m | 0.330039 |

其他指标：

- 固定验证环境的重量常数 baseline：`7.446 kg`；
- 最终重量 MAE 相对该 baseline 改善约 `10.4%`；
- 训练重量 MAE 的最低瞬时值：`4.802 kg`（step 13,500）；
- 同一步验证重量 MAE：`6.949 kg`；
- `estimator_best.pt` 保存于 step 100；
- `estimator_final.pt` 保存于 step 50,000。

## 6. 为什么目前不能认为解耦成功

### 6.1 已确认的问题

#### A. 数据策略仍与旧 estimator 耦合

旧联合 estimator 的重量预测接近常数 baseline，但它仍然影响冻结 actor 的动作。若旧 actor 在不同重量下产生几乎相同的运动，新的 estimator 就得不到足够的负载相关激励。

#### B. 重量只在环境启动时随机化

每个训练环境在整个作业中长期对应一个重量。大量时序样本重复使用同一组约 1229 个训练重量，可能造成：

- 有效重量样本数远小于日志中的样本计数；
- 对环境/轨迹特征过拟合；
- 无法验证同一环境在重新采样重量后的泛化。

#### C. 验证集不是固定离线样本集

验证环境 ID 固定，但每次日志记录时使用的是这些环境当前时刻的一帧状态。因此不同 step 的验证输入、重物位置、接触状态和运动阶段都不同。

这意味着 step 100 和 step 50,000 并不是在相同验证样本上比较。`estimator_best.pt` 在 step 100 被选中，可能部分源于当时状态更容易，而不一定是模型真正更好。

#### D. 输出存在硬边界

重量使用 sigmoid 映射到 `[2.5, 30] kg`，所以不可能预测 30 kg 以上的 OOD 重量，也无法表达“不确定”或“超出训练范围”。

#### E. 位置标签与验证难度随运动变化

位置在自由托盘和自由重物接触过程中快速变化。当前只看单帧验证快照，位置任务的难度会随滑动、掉落、reset 和运动阶段变化，并影响用于保存 best checkpoint 的组合损失。

### 6.2 尚未证实但高度可能的原因

1. **可辨识性不足**：重量、抓握力、双手内力、接触位置、托盘加速度和机器人姿态共同决定 torque/F-T 信号；在缺少受控激励时，质量可能无法从短时历史中唯一辨识。
2. **行为不够丰富**：冻结策略的轻载和重载动作分化很弱，导致不同重量的观测分布高度重叠。
3. **时间相关性过强**：展平 32 帧后使用 MLP，但连续在线样本高度相关，实际有效数据量较低。
4. **reset/drop 样本污染**：托盘或重物失稳、episode reset 附近的状态可能混入训练，而没有按接触质量和运动阶段分类。
5. **多任务干扰**：重量是近似静态参数，位置是快速动态状态。共享 backbone 和固定 1:1 loss 权重可能让两种时间尺度不同的目标互相干扰。
6. **仿真 torque 的物理信息不足或被控制器抵消**：闭环控制可能使不同重量下的姿态和动作相近，把负载差异吸收到接触力、控制误差或未观测内部状态中。
7. **缺少闭环迭代**：新 estimator 只在旧策略分布上训练；装回 policy 后会产生新的动作分布，但当前没有 DAgger 式重复采集与再训练。

## 7. 希望外部方案重点回答的问题

### 数据采集

1. 应使用旧接触策略、oracle policy、脚本化激励，还是多策略混合来产生系统辨识数据？
2. 是否应周期性重新采样每个环境的重量，而不是每个环境固定一个重量？
3. 如何设计不破坏抓握稳定性的受控激励，使质量和重心位置可辨识？
4. 是否应将站立、匀速、加减速、转向和托盘扰动分层采样并保持重量分桶均衡？

### 模型与物理先验

5. 展平历史 MLP 是否应替换为 GRU、TCN、Transformer 或显式状态空间模型？
6. 是否应先利用手腕 F/T、托盘加速度和逆动力学残差构造解析重量估计，再由神经网络学习残差？
7. 重量和动态位置是否应使用独立 encoder/head，或使用不同历史时间尺度？
8. 如何估计不确定度并识别 OOD，而不是通过 sigmoid 强制输出训练范围内的值？

### 训练与验证

9. 如何建立固定、可重复的离线验证集，避免每次验证状态不同？
10. 应采用按重量区间留出、按轨迹留出、按环境参数留出，还是跨随机种子验证？
11. 应如何衡量 estimator 对 policy 的实际价值：质量 MAE、位置 MAE、闭环奖励、抓握成功率，还是负载分桶下的速度跟踪？
12. 是否需要 estimator-policy 交替训练或 DAgger 式闭环数据聚合？

### Sim-to-real

13. TALOS joint torque 与 wrist F/T 的噪声、偏置、延迟和温漂应如何随机化？
14. 若实机 torque 不可靠，是否只凭 wrist F/T、IMU、joint position/velocity 和 action 就能辨识 2.5–30 kg 负载？
15. 双手抓握中的内部夹持力是否需要显式建模或估计，才能把“夹得更紧”和“负载更重”区分开？

## 8. 建议的下一版最低实验设计

为了判断问题来自数据、模型还是物理不可辨识性，建议先做一个不与 policy 闭环耦合的诊断实验：

1. 使用固定重量分桶，例如 `2.5/5/10/15/20/25/30 kg`；
2. 每个重量使用相同命令序列和相同初始抓握状态；
3. 保存完整 episode 为离线数据集；
4. 按 episode 和随机种子划分 train/validation/test；
5. 固定测试集，不随训练变化；
6. 首先只预测重量，暂时移除位置任务；
7. 对比：
   - 常数中位数；
   - 仅 wrist F/T 的线性回归；
   - wrist F/T + IMU 的小型 MLP；
   - 当前完整 32-frame MLP；
   - GRU/TCN；
8. 绘制真实重量—预测重量标定曲线、每个重量分桶 MAE 和混淆/饱和情况；
9. 如果离线固定轨迹下仍无法显著优于 baseline，再讨论可辨识性和传感器设计；
10. 只有离线估计可靠后，再将 estimator 接回 policy，并进行闭环数据聚合。

这一设计能够先回答最基础的问题：**在受控轨迹和可靠标签下，现有传感器中究竟是否包含足够的重量信息。**

## 9. 复现信息

代码：

- `src/pal_mjlab/tasks/velocity/talos/mass_estimation.py`
- `src/pal_mjlab/tasks/velocity/talos/env_cfgs.py`
- `slurm/train_standalone_payload_estimator.py`
- `slurm/mjlab-standalone-estimator.sbatch`

独立训练输出：

```text
/home/stud_yuxuan1/IAS_Workspace/logs/standalone-estimator/
a4ee496d73e2/132896/
```

主要产物：

- `config.json`
- `metrics.jsonl`
- `estimator_best.pt`
- `estimator_final.pt`
- 每 2500 steps 保存的阶段 checkpoint

## 10. 对外求助的简短问题描述

> We are training a payload mass and tray-frame position estimator for a TALOS humanoid carrying a free tray through two-hand rigid contact. The estimator receives 32 frames of noisy proprioception, joint torque, IMU acceleration, previous actions, and bilateral wrist force/torque. The policy and critic do not directly receive torque or ground-truth payload state. Joint PPO/estimator training produced a mass MAE of 6.78 kg over a uniform 2.5–30 kg range, essentially equal to a constant-median baseline. We then froze that joint-estimator policy, collected online rollouts, and trained a separate supervised 4608-512-256-128-4 MLP. Final validation mass MAE was 6.67 kg versus a 7.45 kg held-out constant baseline, with unstable validation and overfitting. However, the rollout policy still contains the old weak estimator, mass is sampled only once per environment, and validation uses changing online states rather than a fixed dataset. We need advice on excitation/data collection, identifiability, fixed validation design, temporal architectures, physics priors, and iterative estimator-policy training.
