import torch
import triton
import triton.testing
from flashattention import TritonAttention   

# triton.testing.perf_report takes a list of configs that define what
# to sweep over on the x-axis of the resulting plot. Here we sweep SEQ_LEN.
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],           # what varies along the x-axis
        x_vals=[256, 512, 1024, 2048, 4096],  # realistic values for a 3050
        line_arg="provider",           # what distinguishes the lines on the plot
        line_vals=["triton", "torch"],
        line_names=["Triton FlashAttention", "PyTorch Attention"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="fwd-attention-benchmark",
        args={                         # fixed parameters across all runs
            "BATCH_SIZE": 2,
            "NUM_HEADS": 4,
            "HEAD_DIM": 64,
            "causal": True,
            "dtype": torch.float16,
        },
    )
)
def benchmark(SEQ_LEN, provider, BATCH_SIZE, NUM_HEADS, HEAD_DIM, causal, dtype):
    # Set up input tensors on GPU
    Q = torch.randn(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        device="cuda", dtype=dtype, requires_grad=True
    )
    K = torch.randn_like(Q).requires_grad_(True)
    V = torch.randn_like(Q).requires_grad_(True)
    softmax_scale = 1.0 / (HEAD_DIM ** 0.5)

    # quantiles tells the benchmarker to return median, 20th, and 80th
    # percentile timings — this gives you a sense of variance too
    quantiles = [0.5, 0.2, 0.8]

    if provider == "triton":
        fn = lambda: TritonAttention.apply(Q, K, V, causal, softmax_scale)
    else:
        # Standard PyTorch attention — materializes the full N x N matrix
        MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device="cuda"))
        def fn():
            P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
            if causal:
                P = P.masked_fill(MASK[None, None, :, :] == 0, float("-inf"))
            P = torch.softmax(P.float(), dim=-1).half()
            return torch.matmul(P, V)

    # do_bench returns median ms by default; with quantiles it returns
    # (median_ms, low_ms, high_ms) which perf_report uses for error bars
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    # Convert milliseconds → TFLOPS
    # The FLOPs formula for attention is 4 * B * H * N^2 * D
    # (factor of 4 comes from: QK^T matmul = 2BHN^2D, PV matmul = 2BHN^2D)
    # For causal attention it's roughly half since we skip the upper triangle
    flops = 4 * BATCH_SIZE * NUM_HEADS * (SEQ_LEN ** 2) * HEAD_DIM
    if causal:
        flops /= 2

    # Convert: flops / (ms * 1e3) gives FLOPS/s, divide by 1e12 for TFLOPS
    tflops     = lambda ms: flops / ms * 1e3 / 1e12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True)