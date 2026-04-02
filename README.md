# FlashAttentionFromScratch

A from-scratch Triton kernel implementation of the FlashAttention-2 forward pass and backward pass,
achieving up to **23× speedup** and **14.1 TFLOPS** over naive PyTorch attention on an RTX 3050 Laptop GPU.

> Memory movement — not FLOPs — is the bottleneck in attention. This kernel eliminates most of it.

---

## Benchmarks

<img width="2230" height="693" alt="image" src="https://github.com/user-attachments/assets/23835a05-61e5-4327-8566-a07cde6c7a4c" /> 

*Three views of the same result: runtime diverges exponentially, memory stays linear, speedup compounds with sequence length.*

### TFLOPS vs Sequence Length

| Sequence Length | FlashAttention (Triton) | Naive PyTorch | Speedup |
|---|---|---|---|
| 256 | 3.64 | 0.54 | 6.7× |
| 512 | 7.28 | 0.56 | 13.0× |
| 1024 | 10.38 | 0.59 | 17.6× |
| 2048 | 12.91 | 0.52 | **24.8×** |
| 4096 | **14.15** | 0.61 | **23.2×** |

*Config: RTX 3050 Laptop GPU (4GB VRAM), B=2, H=4, D=64, fp16, causal masking.*

---

## Why Does the Speedup Compound With Sequence Length?

The ~23× result isn't a coincidence but a predictable consequence of where the bottleneck lies.

**Standard attention is memory-bandwidth bound, not compute-bound.** It materializes the full N×N attention matrix, which means memory reads and writes scale quadratically with sequence length. As the table above shows, PyTorch's TFLOPS stays nearly flat (~0.5–0.6) across all sequence lengths meaning the GPU is idle waiting on memory, not doing math.

FlashAttention avoids this by tiling computation into on-chip SRAM and tracking running softmax statistics, so intermediate attention weights never leave the chip. The memory plot tells the story directly: PyTorch's memory usage curves exponentially while FlashAttention's stays near-linear.

**Three hardware-specific factors amplify the gap on this setup:**

1. **Bandwidth-constrained GPU.** The RTX 3050 Laptop GPU has limited HBM bandwidth relative to data-center GPUs. Memory inefficiencies are proportionally more expensive here, so IO-aware algorithms see larger gains.

2. **Small batch / head count.** At B=2, H=4, naive PyTorch kernels underutilize the GPU as there isn't enough parallelism to saturate compute. FlashAttention's tiling achieves better occupancy even at low batch sizes.

3. **Long sequence lengths.** The quadratic memory cost of naive attention only becomes punishing at scale. Below seq_len 512, the gap is modest (6–13×). Above 1024, it compounds rapidly.

**Why causal attention sees higher speedup than full attention** is also visible in the rightmost plot: FlashAttention skips tiles below the causal diagonal entirely rather than processing and then masking them. This cuts compute proportionally to sequence length while naive PyTorch still incurs the overhead.

**Honest caveat:** The baseline here is `torch.matmul` + manual softmax — the naive reference implementation. Comparing against `F.scaled_dot_product_attention` or xFormers (which already apply fused kernels) would reduce the gap to approximately **2–5×** on comparable hardware. The architectural advantage is real; the magnitude depends on how memory-bound your baseline is.

> **Note:** All reported results measure **forward pass performance only**.  
> The backward pass (implemented using FlashAttention v1-style recomputation) is not included in these benchmarks.

---

## What's Implemented

- **Forward pass** — tiled SRAM computation with online softmax normalization (Algorithm 1, FA2 paper)
- **Backward pass** — recomputation-based gradient calculation following the FlashAttention v1 paper (Algorithm 2), avoiding storage of the full attention matrix
- **Causal masking** — applied at tile level; tiles entirely below the diagonal are skipped
- **dtype support** — `fp16` / `bf16`
- **Triton primer** — `triton_vector_add.ipynb`: worked walkthrough of how Triton kernel programming works, using vector addition as the entry point

---

## Repository Structure

```
flashattention-triton/
├── flashattention.py          # Forward pass Triton kernel
├── benchmarks.py              # Reproduces all three benchmark plots
└── triton_vector_add.ipynb    # Triton introduction — vector addition walkthrough
```

---

## Quick Start

```bash
pip install torch triton matplotlib

# Reproduce the benchmark plots
python benchmarks.py

# Run the kernel directly
python flashattention.py
```

---

## Design Notes

- **Block size 64×64** — tuned to fit the RTX 3050's shared memory constraints without thrashing
- **Online softmax** — only running max and running sum are tracked per tile; the full N×N softmax is never materialized
- **Tile-level causal masking** — tiles that are fully masked are skipped entirely; only boundary tiles apply the mask, minimizing wasted compute
- **Backward pass implemented** — uses recomputation (FlashAttention v1) to avoid materializing attention matrices during gradient computation *(not benchmarked yet)*

---

## References

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Tri Dao, 2023
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Tillet et al., 2019
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — Milakov & Gimelshein, 2018
