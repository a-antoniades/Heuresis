#!/usr/bin/env python3
"""Pre-download flash-attn3 kernels to the host HF cache so bwrap sandboxes
can use them with HF_HUB_OFFLINE=1.

Run once on each host (needs GPU access):
    venvs/nanogpt/bin/python scripts/preflight_nanogpt.py
"""

import torch
from kernels import get_kernel

cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
print(f"GPU capability: {cap}, downloading kernel from {repo}...")
mod = get_kernel(repo)
print(f"Kernel cached: {mod}")
