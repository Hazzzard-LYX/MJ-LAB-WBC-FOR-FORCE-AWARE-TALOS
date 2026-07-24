"""Finite environment and CUDA smoke test for the MJLab container."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata

import torch
import warp as wp

EXPECTED_VERSIONS = {
  "mjlab": "1.5.0",
  "mujoco": "3.10.0",
  "mujoco-warp": "3.10.0.1",
  "numpy": "2.4.2",
  "pal-mjlab": "0.1.0",
  "rsl-rl-lib": "5.4.0",
  "torch": "2.10.0",
  "warp-lang": "1.15.0",
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--cpu-only",
    action="store_true",
    help="Validate packages without requiring a visible NVIDIA GPU.",
  )
  args = parser.parse_args()

  actual_versions = {
    package: metadata.version(package) for package in EXPECTED_VERSIONS
  }
  mismatches = {
    package: {"expected": expected, "actual": actual_versions[package]}
    for package, expected in EXPECTED_VERSIONS.items()
    if actual_versions[package] != expected
  }
  if mismatches:
    raise RuntimeError(f"Package version mismatch: {mismatches}")

  wp.init()
  cuda_devices = [str(device) for device in wp.get_cuda_devices()]
  report = {
    "python": sys.version.split()[0],
    "packages": actual_versions,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "warp_cuda_devices": cuda_devices,
  }

  if not args.cpu_only:
    if not torch.cuda.is_available():
      raise RuntimeError("PyTorch cannot see a CUDA device inside the container.")
    if not cuda_devices:
      raise RuntimeError("Warp cannot see a CUDA device inside the container.")

    tensor = torch.arange(1024, device="cuda", dtype=torch.float32)
    expected_sum = 1024 * 1023 / 2
    actual_sum = tensor.sum().item()
    if actual_sum != expected_sum:
      raise RuntimeError(
        f"CUDA tensor validation failed: expected {expected_sum}, got {actual_sum}"
      )
    report["cuda_device_name"] = torch.cuda.get_device_name(0)
    report["cuda_tensor_sum"] = actual_sum

  print(json.dumps(report, indent=2, sort_keys=True), flush=True)
  print("MJLAB_CONTAINER_SMOKE_OK", flush=True)


if __name__ == "__main__":
  main()
