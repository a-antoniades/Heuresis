# NanoGPT Pretraining

NanoGPT is a small decoder-only transformer trained from scratch on the
FineWeb-Edu dataset. The objective is to minimize validation bits-per-byte
(`val_bpb`) on a held-out slice within a fixed wall-clock budget on a single
A100 GPU.

The seed configuration uses 8 layers, hidden size 512, 4 attention heads, and
the Muon + AdamW optimizer mix from Karpathy's reference. Baseline `val_bpb`
on this configuration is approximately 0.992 after a 30-minute training run.

The agent edits `train.py` to propose architectural, optimization, or training
modifications. Common axes of variation: optimizer (Muon, AdamW, SGD,
Lion, Sophia), depth/width tradeoffs, attention variants, normalization,
learning-rate schedules, and curriculum tweaks. Modifications must keep the
training step compatible with the surrounding harness and finish within the
budget.
