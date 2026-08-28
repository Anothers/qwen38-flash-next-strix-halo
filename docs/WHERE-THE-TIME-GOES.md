# Where decode time actually goes — measured, not modelled

This supersedes the reasoning in [`WHY-IT-IS-FAST.md`](WHY-IT-IS-FAST.md) §2–3. Those
sections argued from architecture and source constants. Every one of those arguments
turned out to be wrong when measured. This document is the per-operation profile from
`GGML_VK_PERF_LOGGER`, and it disagrees with almost all of them.

Ryzen AI Max+ 395 (gfx1151), Vulkan/RADV, 115 W, `-c 16384`, no speculation,
200-token generation. Sample covers 25 decode steps. Raw logs in
[`../results/perf-ops-20260828/`](../results/perf-ops-20260828/).

---

## The headline number

```
GPU busy time      42.60 ms / token
measured eval time 43.66 ms / token
                => GPU is busy 97.6% of the wall clock
```

**There is no idle time to reclaim.** An earlier version of this repository claimed
72% of a decode step was launch and synchronisation overhead. That was inferred from a
bandwidth model, not measured, and it was wrong. The bandwidth model assumed only
"6B active params × 4 bits = 3.96 GB/token" gets read; the actual traffic is far
higher (the output projection alone, m=248320 × k=2560, is 675 MB per token).

`graphs reused = 199 / 200` — graph rebuilding is not a cost either.

---

## Top decode operations

| operation | calls | ms | share | GFLOPS/s |
|---|---:|---:|---:|---:|
| `MUL_MAT_ID_VEC q4_K m=640 n=10 k=2560` (MoE experts) | 2350 | 140.8 | 13.2% | 547 |
| `MUL_MAT_VEC q8_0 m=10240 k=2560` (HC up) | 925 | 120.8 | 11.3% | 401 |
| `MUL_MAT_VEC q8_0 m=2560 k=6144` | 1200 | 97.3 | 9.1% | 388 |
| `MUL_MAT_ID_MUL … q5_1 m=2560` (shared expert) | 1075 | 76.4 | 7.2% | 461 |
| `MUL_MAT_VEC q8_0 m=248320 k=2560` (output head) | 25 | 73.7 | 6.9% | 431 |
| `MUL_MAT_VEC q8_0 m=6144 k=2560` | 900 | 67.9 | 6.4% | 417 |
| `MUL_MAT_VEC q8_0 m=320 k=10240` (HC down) | 2425 | 48.5 | 4.6% | 328 |
| `MUL_MAT_VEC q8_0 m=12288 k=2560` | 300 | 46.0 | 4.3% | 410 |
| `MUL_MAT_VEC q8_0 m=10240 k=320` (HC up) | 2425 | 42.5 | 4.0% | 374 |
| **`MUL_MAT_VEC f32 m=512 k=2560` (MoE router)** | 1200 | 35.2 | 3.3% | **89** |
| `MUL` | 10800 | 26.6 | 2.5% | — |
| `MUL_MAT_VEC q8_0 m=640 k=2560` | 2400 | 23.5 | 2.2% | 334 |
| `SCALE` | 9975 | 23.3 | 2.2% | — |
| **`MUL_MAT_VEC f32 m=4 k=10240` (HC inject)** | 2400 | 21.3 | 2.0% | **9** |

The first nine entries run at **328–547 GFLOPS/s**. For `n=1` matrix-vector work that is
memory-bound arithmetic sitting at this machine's bandwidth ceiling (~190–215 GB/s).
They are not slow; that is what the work costs.

---

## The only two inefficient operations

| operation | share | GFLOPS/s | why |
|---|---:|---:|---|
| `f32 m=512 k=2560` | 3.3% | 89 | MoE router, kept in f32 — 4× off the quantized path |
| `f32 m=4 k=10240` | 2.0% | 9 | HyperConnections inject: only **4** outputs, so the GPU is essentially idle inside the kernel |

`m=4` is the four residual branches of HyperConnections (`hc_dim = 10240 = n_embd 2560 × 4`,
`hc_*_inject` is `[10240, 4]`). Computing 4 output values from a 10240-wide reduction
gives a modern GPU nothing to parallelise over.

**Combined headroom: 5.3% of GPU time.** Even if both were brought to 400 GFLOPS/s, the
end-to-end gain would be roughly **+4.7%** (42.60 → 40.7 ms/token). Worth reporting
upstream; not worth restructuring anything locally.

---

## What is *not* the problem

Each of these was proposed here and then killed by measurement.

| hypothesis | prediction | measured |
|---|---|---|
| MoE tile occupancy at `ub=1024` is only 62.5% | full-tile `ub=1638` should be fastest | **1638 was the slowest** (196.96 vs 204.39 t/s prefill); the fastest was `ub=2048`, which has the *same* 62.5% occupancy as the baseline |
| `mul_mat_vec_max_cols = 8` caps the speculation peak | raising it to 16 moves the peak past n=5 | peak stayed at n=5. Rebuilt with 16 and re-profiled: decode ops are **bit-for-bit the same cost** (total 1.06495e6 vs 1.06771e6 µs). In decode `n=1`, so the threshold never binds |
| submissions are too fragmented (≈80 per step) | batching them should help | opposite and non-monotonic: 100→26.20, 512→15.95, 2048→12.28, 8192→24.71 t/s. Default 100 is already right |
| GDN scan dominates | recurrent layers are the prefill/decode cost | `GATED_DELTA_NET` is **1.23%** of decode |
| 72% of a step is launch/sync overhead | — | GPU is busy **97.6%** of wall time |

The `mul_mat_vec_max_cols` change does help *speculative verify batches larger than 8*
(n=7 +22.5%, n=8 +13.9%, n=10 +27.6%), but since the optimum is `n_max=5` — verify
batch 6 — it never applies in a tuned configuration.

---

## Practical conclusion

On this hardware and backend, **decode for this model is done**. 97.6% GPU occupancy,
nine of the top ten operations at the memory-bandwidth ceiling, 5.3% of time in two
kernels whose inefficiency is upstream rather than configurable.

Nothing in `-ub`, `mul_mat_vec_max_cols`, or `GGML_VK_MAX_NODES_PER_SUBMIT` improved
anything. The shipped configuration in the main README is the fastest one found.

---

## Method

```bash
GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=50 llama-server ... -lv 4
```

The logger prints per-op call counts, mean and total time, and GFLOPS every N graph
executions, then clears. **Compare like with like**: early samples mix prefill and
decode (ops appear with `n=9`), later samples are pure decode (`n=1`). An earlier draft
of this analysis compared a mixed sample against a decode-only sample and drew the
wrong conclusion from it.
