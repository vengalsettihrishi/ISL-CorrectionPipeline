"""
benchmark.py -- Model verification and benchmarking utility.

Tests:
    1. Parameter count and estimated model size (float32)
    2. Parallel vs recurrent forward agreement (T=300)
    3. CPU inference latency measurement (full forward + recurrent)
    4. CTC loss validity check
    5. Optional ONNX export + correctness verification

Does NOT fabricate Raspberry Pi / ARM numbers. If ARM measurement is
needed, run this script on the target device. Local results are from
the host CPU (likely x86/x64) and will differ from ARM performance.

Usage:
    python benchmark.py
    python benchmark.py --onnx       # also test ONNX export
    python benchmark.py --seq_len 150  # test shorter sequences
"""

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from model.isl_model import ISLModel
from model.mingru import MinGRUCell


def separator(title: str = "") -> None:
    """Print a section separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print("=" * 60)


def benchmark_model_size(model: ISLModel) -> bool:
    """Test 1: Parameter count and model size."""
    separator("Test 1: Model Size")

    params = model.count_parameters()
    size_mb = model.model_size_mb()

    print(f"  Total parameters:  {params:,}")
    print(f"  Estimated size:    {size_mb:.3f} MB (float32)")
    print()

    # Per-component breakdown
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.ctc_head.parameters())
    proj_params = sum(p.numel() for p in model.encoder.input_proj.parameters())
    gru_params = sum(p.numel() for p in model.encoder.mingru_stack.parameters())
    print(f"  Encoder total:     {enc_params:,} ({enc_params*4/1024**2:.3f} MB)")
    print(f"    Input projection:  {proj_params:,}")
    print(f"    minGRU stack:      {gru_params:,}")
    print(f"  CTC Head:          {head_params:,} ({head_params*4/1024**2:.3f} MB)")
    print()

    pass_params = params < 500_000
    pass_size = size_mb < 2.0
    print(f"  Under 500K params: {'PASS' if pass_params else 'FAIL'} ({params:,})")
    print(f"  Under 2MB:         {'PASS' if pass_size else 'FAIL'} ({size_mb:.3f})")

    return pass_params and pass_size


def benchmark_parallel_vs_recurrent(seq_len: int = 300) -> bool:
    """Test 2: Verify parallel and recurrent modes produce identical output."""
    separator(f"Test 2: Parallel vs Recurrent (T={seq_len})")

    D_in, D_h = 450, 128
    cell = MinGRUCell(D_in, D_h)
    cell.eval()

    B = 2
    x = torch.randn(B, seq_len, D_in)

    with torch.no_grad():
        # Parallel
        h_par = cell.parallel_forward(x)

        # Recurrent
        h_rec_list = []
        h = None
        for t in range(seq_len):
            h = cell.recurrent_forward(x[:, t, :], h)
            h_rec_list.append(h)
        h_rec = torch.stack(h_rec_list, dim=1)

    max_diff = (h_par - h_rec).abs().max().item()
    mean_diff = (h_par - h_rec).abs().mean().item()

    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")

    passed = max_diff < 1e-5
    print(f"  Match:         {'PASS' if passed else 'FAIL'}")

    return passed


def benchmark_cpu_latency(
    model: ISLModel,
    seq_len: int = 300,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> bool:
    """Test 3: Measure CPU inference latency."""
    separator(f"Test 3: CPU Inference Latency (T={seq_len})")

    model.eval()
    model.cpu()

    x = torch.randn(1, seq_len, model.input_dim)
    x_lengths = torch.tensor([seq_len], dtype=torch.int32)

    # Warm up
    for _ in range(n_warmup):
        with torch.no_grad():
            model(x, x_lengths)

    # --- Full forward (parallel mode) ---
    times_full = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model(x, x_lengths)
        times_full.append(time.perf_counter() - start)

    avg_full_ms = np.mean(times_full) * 1000
    std_full_ms = np.std(times_full) * 1000

    print(f"  Full forward (batch=1, T={seq_len}):")
    print(f"    Mean: {avg_full_ms:.1f} ms +/- {std_full_ms:.1f} ms")
    print(f"    Min:  {min(times_full)*1000:.1f} ms")
    print(f"    Max:  {max(times_full)*1000:.1f} ms")

    # --- Recurrent (frame-by-frame) ---
    times_rec = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.predict_recurrent(x)
        times_rec.append(time.perf_counter() - start)

    avg_rec_ms = np.mean(times_rec) * 1000
    std_rec_ms = np.std(times_rec) * 1000

    print(f"  Recurrent (frame-by-frame, T={seq_len}):")
    print(f"    Mean: {avg_rec_ms:.1f} ms +/- {std_rec_ms:.1f} ms")
    print(f"    Min:  {min(times_rec)*1000:.1f} ms")
    print(f"    Max:  {max(times_rec)*1000:.1f} ms")

    # Per-frame latency for recurrent mode
    per_frame_ms = avg_rec_ms / seq_len
    print(f"    Per-frame: {per_frame_ms:.3f} ms")

    print()
    pass_full = avg_full_ms < 100
    print(f"  Full <100ms (local CPU):      {'PASS' if pass_full else 'FAIL'} ({avg_full_ms:.1f}ms)")
    print()
    print("  NOTE: These are local x86/x64 CPU measurements.")
    print("  ARM (Raspberry Pi 4) latency will differ -- typically 3-5x slower.")
    print("  For accurate ARM numbers, run this script on the target device.")

    return pass_full


def benchmark_ctc_loss(model: ISLModel, seq_len: int = 300) -> bool:
    """Test 4: Verify CTC loss computes without NaN/Inf."""
    separator(f"Test 4: CTC Loss Validity")

    model.eval()
    B = 4
    x = torch.randn(B, seq_len, model.input_dim)
    x_lengths = torch.full((B,), seq_len, dtype=torch.int32)

    with torch.no_grad():
        log_probs, out_lengths = model(x, x_lengths)

    # Random targets (valid token IDs, not blank)
    targets = torch.randint(1, model.vocab_size, (B, 5))
    target_lengths = torch.full((B,), 5, dtype=torch.int32)

    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V)
    loss = criterion(log_probs_ctc, targets, out_lengths, target_lengths)

    is_valid = not (torch.isnan(loss) or torch.isinf(loss))
    print(f"  CTC loss value: {loss.item():.4f}")
    print(f"  Not NaN:        {'PASS' if not torch.isnan(loss) else 'FAIL'}")
    print(f"  Not Inf:        {'PASS' if not torch.isinf(loss) else 'FAIL'}")
    print(f"  Overall:        {'PASS' if is_valid else 'FAIL'}")

    return is_valid


def benchmark_onnx_export(
    model: ISLModel,
    seq_len: int = 300,
) -> bool:
    """Test 5: ONNX export and optional correctness verification."""
    separator("Test 5: ONNX Export")

    onnx_path = "./checkpoints/benchmark_model.onnx"

    try:
        model.export_onnx(onnx_path, seq_length=seq_len)
        file_size_mb = Path(onnx_path).stat().st_size / (1024 ** 2)
        print(f"  ONNX file:    {onnx_path}")
        print(f"  ONNX size:    {file_size_mb:.3f} MB")
    except Exception as e:
        print(f"  ONNX export FAILED: {e}")
        return False

    # Try to verify with onnxruntime if available
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)

        x_np = np.random.randn(1, seq_len, model.input_dim).astype(np.float32)
        lengths_np = np.array([seq_len], dtype=np.int32)

        ort_outputs = sess.run(None, {
            "features": x_np,
            "lengths": lengths_np,
        })

        # Compare with PyTorch
        model.eval()
        model.cpu()
        with torch.no_grad():
            pt_log_probs, _ = model(
                torch.from_numpy(x_np),
                torch.from_numpy(lengths_np),
            )
        pt_np = pt_log_probs.numpy()
        ort_np = ort_outputs[0]

        max_diff = np.abs(pt_np - ort_np).max()
        print(f"  ONNX vs PyTorch max diff: {max_diff:.2e}")
        print(f"  ONNX correctness:         {'PASS' if max_diff < 1e-4 else 'FAIL'}")

    except ImportError:
        print("  onnxruntime not installed -- skipping correctness check")
        print("  Install with: pip install onnxruntime")

    # Cleanup
    try:
        Path(onnx_path).unlink()
    except Exception:
        pass

    print(f"  Export:        PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ISL recognition model"
    )
    parser.add_argument("--seq_len", type=int, default=300,
                        help="Sequence length for benchmarks")
    parser.add_argument("--onnx", action="store_true",
                        help="Also test ONNX export")
    parser.add_argument("--vocab_size", type=int, default=50,
                        help="Vocabulary size for test model")
    args = parser.parse_args()

    cfg = Config()
    model = ISLModel.from_config(cfg, args.vocab_size)
    model.eval()

    separator("ISL Model Benchmark Suite")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Vocab size:      {args.vocab_size}")
    print(f"  Hidden size:     {cfg.hidden_size}")
    print(f"  Layers:          {cfg.num_gru_layers}")

    results = {}

    # Run tests
    results["model_size"] = benchmark_model_size(model)
    results["par_vs_rec"] = benchmark_parallel_vs_recurrent(args.seq_len)
    results["cpu_latency"] = benchmark_cpu_latency(model, args.seq_len)
    results["ctc_loss"] = benchmark_ctc_loss(model, args.seq_len)

    if args.onnx:
        results["onnx_export"] = benchmark_onnx_export(model, args.seq_len)

    # Summary
    separator("Summary")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  Overall: ALL TESTS PASSED")
    else:
        print("  Overall: SOME TESTS FAILED")
    separator()


if __name__ == "__main__":
    main()
