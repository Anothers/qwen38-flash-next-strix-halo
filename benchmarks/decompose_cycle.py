#!/usr/bin/env python3
"""
Decompose speculative decode cycles into draft cost vs target-model cost.

This is what turns "n=8 is slower" into "the verify batch crossed
mul_mat_vec_max_cols = 8 and fell off the fast Vulkan matmul path".

Feed it the server logs produced by benchmarks/mtp_sweep.sh (run with -lv 4 so the
`spec common_specu: statistics` line is emitted) plus the decode rate each run
reported.

Usage:
    decompose_cycle.py --predict 400 \
        --run 0:13.28 \
        --run 3:21.01:/tmp/mtp_sweep_n3.log \
        --run 5:27.18:/tmp/mtp_sweep_n5.log ...
"""
import argparse, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", default=[], metavar="N:TPS[:LOG]",
                help="n_max, measured decode tok/s, and the server log for that run")
ap.add_argument("--predict", type=int, default=400, help="n_predict used in the runs")
ap.add_argument("--threshold", type=int, default=8,
                help="mul_mat_vec_max_cols in your llama.cpp build")
a = ap.parse_args()
if not a.run:
    ap.error("give at least one --run")

STAT = re.compile(r"#gen drafts = *(\d+),.*?#mean acc len = *([0-9.]+)")
DUR  = re.compile(r"dur\(b,g,a\) = *([0-9.]+), *([0-9.]+), *([0-9.]+)")

print(f"{'n':>3} {'batch':>6} {'cycle ms':>9} {'draft ms':>9} {'target ms':>10} "
      f"{'tgt/token':>10}  path")
print("-" * 62)
for spec in a.run:
    parts = spec.split(":")
    n, tps = int(parts[0]), float(parts[1])
    log = parts[2] if len(parts) > 2 else None

    if n == 0 or not log:
        mean, draft_ms = 1.0, 0.0
    else:
        try:
            txt = open(log, errors="replace").read()
        except OSError as e:
            print(f"{n:>3}  cannot read log: {e}", file=sys.stderr); continue
        m, d = STAT.search(txt), DUR.search(txt)
        if not m:
            print(f"{n:>3}  no speculation statistics in {log} (run with -lv 4)", file=sys.stderr)
            continue
        cycles, mean = int(m.group(1)), float(m.group(2))
        draft_ms = (float(d.group(2)) / cycles) if d and cycles else 0.0

    cycle_ms = mean / tps * 1000.0
    target_ms = cycle_ms - draft_ms
    batch = n + 1
    path = "fast MUL_MAT_VEC" if batch <= a.threshold else "tiled MUL_MAT (slow)"
    print(f"{n:>3} {batch:>6} {cycle_ms:>9.1f} {draft_ms:>9.1f} {target_ms:>10.1f} "
          f"{target_ms/batch:>10.1f}  {path}")

print()
print(f"note: verify batch is n_max+1; llama.cpp Vulkan leaves the fast vector path")
print(f"      above mul_mat_vec_max_cols = {a.threshold} (ggml-vulkan.cpp).")
