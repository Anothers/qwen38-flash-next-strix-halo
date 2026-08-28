# Why it runs this fast — and what actually limits it

A 176B-parameter model decoding at 27.2 tok/s on an integrated GPU is not obvious.
This is the arithmetic behind it, measured rather than assumed.

All numbers: Ryzen AI Max+ 395 (gfx1151), Vulkan/RADV, Mesa 26.1.7, 115 W cap,
`-c 131072`, 52K cold prefill, `n_predict 400`, seed 42.

---

## 1. Sparsity is what makes it possible

| | |
|---|---|
| total parameters | 176B (125B backbone + 51B n-gram/PLE table) |
| active per token | **6B** (10 of 512 experts, + 1 shared) |
| UD-Q4_K_XL on disk | 103.7 GiB |
| PLE table (CPU-resident) | 26.8 GiB |
| **GPU-resident weights** | **76.2 GiB** |

The PLE table is a lookup, not a matmul — it stays on CPU and is read a couple of KB
per token. What the GPU streams per token is roughly the active 6B at ~5.3 bits/param:

```
6e9 params × 5.3 bits / 8 = 3.96 GB per token
```

This machine sustains **190 GB/s** effective (88% of a measured 215 GB/s peak).
So a purely bandwidth-bound decode would predict:

```
190 GB/s ÷ 3.96 GB = ~48 tok/s
```

Measured without speculation: **13.28 tok/s** — about **28%** of that ceiling.

**Decode here is not bandwidth-bound.** That single fact drives everything below.

---

## 2. Where the time actually goes

Splitting each speculative cycle into draft generation and target verification
(from `llama.cpp`'s own `dur(b,g,a)` counters and cycle counts):

| n_max | verify batch | cycle (ms) | draft (ms) | target (ms) | target per token (ms) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 75.3 | 0.0 | 75.3 | 75.3 |
| 3 | 4 | 162.3 | 15.1 | 147.2 | 36.8 |
| **5** | **6** | **164.8** | **26.7** | **138.1** | **23.0** |
| 7 | 8 | 252.0 | 34.9 | 217.1 | 27.1 |
| 8 | 9 | 368.5 | 40.3 | 328.2 | 36.5 |
| 10 | 11 | 448.0 | 51.1 | 396.9 | 36.1 |

Two things stand out.

**A single decode step is mostly fixed cost.** Verifying 6 tokens costs 138 ms;
verifying 1 costs 75 ms. Six times the work for 1.8× the time. Per-token cost falls
from 75.3 ms to 23.0 ms. This is why speculative decoding pays so well here — it is
amortizing a large per-step overhead, not just saving weight reads.

**Cycle time is nearly flat from batch 4 to batch 6** (162.3 → 164.8 ms) even though
the batch grows 50%. The extra tokens are almost free. All of the 1.60× → 2.07×
improvement comes from producing more tokens per cycle, not from cycles getting cheaper.

---

## 3. Why the peak is exactly at n_max = 5

`llama.cpp`'s Vulkan backend has a hard threshold:

```c
// ggml/src/ggml-vulkan/ggml-vulkan.cpp:389
static constexpr uint32_t mul_mat_vec_max_cols = 8;

// :10126 — above this, the fast vector path is not selected
} else if ((dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && ...)) &&
```

At or below 8 columns, matmuls take the `MUL_MAT_VEC` path. Above it, they fall through
to tiled `MUL_MAT`. Earlier measurements on this same machine put those two paths at
**98.7 GB/s** and **12.0 GB/s** respectively.

The verify batch is `n_max + 1`. Lining that up against the table:

| n_max | verify batch | vs threshold | target cost |
|---:|---:|---|---:|
| 3 | 4 | under | 147.2 ms |
| **5** | **6** | **under** | **138.1 ms** |
| 7 | 8 | at the limit | 217.1 ms |
| 8 | **9** | **over** | **328.2 ms** |
| 10 | 11 | over | 396.9 ms |

**Crossing from batch 8 to batch 9 — exactly the threshold — the target pass jumps
from 217 ms to 328 ms.** Past that point every extra draft token is paid for on the
slow path, and the larger mean accept length can no longer make up for it.

So `n_max = 5` is not a tuning coincidence. It is the largest draft that keeps the
verify batch comfortably inside the fast kernel path.

> **Open prediction.** Raising `mul_mat_vec_max_cols` (it is a compile-time constant)
> should move the peak to larger `n_max` and yield more than 2.07×. Acceptance is still
> healthy out at n=10 — the first draft position accepts 0.864 — so there is unused
> headroom. This has **not** been tested here; it is the obvious next experiment, and it
> would also affect every other speculative-decoding setup on the Vulkan backend.

---

## 4. Why acceptance is not the limiting factor

Per-position acceptance rates:

```
n=5   (0.854, 0.764, 0.674, 0.618, 0.573)
n=7   (0.812, 0.725, 0.625, 0.537, 0.487, 0.425, 0.375)
n=8   (0.944, 0.789, 0.592, 0.521, 0.479, 0.465, 0.451, 0.380)
n=10  (0.864, 0.831, 0.763, 0.729, 0.644, 0.508, 0.492, 0.373, 0.288, 0.271)
```

Mean accepted length keeps **rising** with `n_max` — 3.41 at n=3, 4.48 at n=5, 6.76 at
n=10. The draft head keeps doing useful work. Throughput still collapses. The loss is
entirely on the verification side, which is what §3 explains.

This is worth stating plainly because the intuitive reading of "n=8 is slower" is
"the draft stopped being accurate." It didn't.

---

## 5. What 256K context actually costs

| `-c` | ready GTT | Vulkan0 compute buffer | prefill | decode @52K | decode, short prompt |
|---:|---:|---:|---:|---:|---:|
| 131072 | 87.1 GiB | 2322 MiB | 208.1 | **27.18** | 27.55 |
| 262144 | 93.0 GiB | 4386 MiB | 206.8 | **20.23** | 26.87 |

Same seed, same prompt. Acceptance is **bit-identical** across the two runs — 0.69663,
310/445 accepted, mean length 4.48, same per-position vector — so the loss is not on
the speculation side. Prefill is unaffected (206.8 vs 208.1), and short prompts are
essentially unaffected (26.87 vs 27.55). The cost appears only once there is
substantial context, and the compute buffer doubling from 2322 to 4386 MiB is the
visible symptom.

Practical consequence: allocating 256K costs roughly a quarter of long-context decode
and nothing at short context. Whether that trade is worth it depends on how often you
actually fill the window.

> **This section was wrong in the first version of this document.** It reported 13.74
> t/s for `-c 262144` and concluded decode was halved. A later run of the identical
> configuration measured 20.23 — a 47% spread on the same settings, same prompt, same
> seed. The 13.74 run followed several back-to-back server restarts; the 20.23 run was
> a clean start. Decode on this model is **not reproducible from a single run**. Every
> other figure in this repository is also a single run, so treat any difference under
> ~10% here as unresolved; the effects this document builds its argument on are
> 1.5–2.5× and survive that uncertainty, but this one did not.

---

## 6. How this compares to other hardware

Published decode figures for the same model:

| machine | memory bandwidth | decode | notes |
|---|---:|---:|---|
| M5 Max 64 GB | ~546 GB/s | 36 t/s | Metal, no speculation |
| DGX Spark 128 GB | ~273 GB/s | 15.7 t/s | CUDA, no speculation |
| **this machine** | **~215 GB/s peak** | **13.3 / 27.2** | Vulkan, without / with MTP |

Without speculation the three land roughly in bandwidth order, and all three sit far
below their own naive bandwidth ceilings — consistent with §1's finding that fixed
per-step cost dominates for this architecture.

With the MTP head this machine passes the DGX Spark result by 1.7× despite having
~20% less bandwidth. The draft head is worth more than the hardware gap.

---

## 7. What this means if you are tuning

1. **Get the MTP draft working before tuning anything else.** It is worth 2.07×;
   nothing else here is worth more than a few percent.
2. **Set `n_max` so that `n_max + 1 ≤ 8`.** On this backend that means 5 is the
   practical maximum; 7 already costs you.
3. **Allocate only the context you need.** 256K costs half your decode speed for a
   window you probably will not fill.
4. **`--load-mode none` over `mmap`** — worth ~1.9%, and the `--override-tensor` flag
   from the widely-copied recipe does nothing (see the main README).
5. Do not expect prefill gains. It measured 208 t/s here versus 228 for a comparable
   MoE — prefill is compute-bound and this architecture does not help there.

---

## Method notes

- Cycle counts, mean accept length, per-position acceptance and draft duration come
  from `llama.cpp`'s own `spec common_specu: statistics` line at `-lv 4`.
- Target-model cost is derived: `cycle_time − draft_time`, where
  `cycle_time = mean_accept_len / decode_rate`. Cycle counts check out against the
  400-token budget to within one token in all six runs.
- The 190 GB/s effective / 215 GB/s peak figures are from earlier bandwidth work on
  this same machine, not from this model's runs.
- Single machine, single run per configuration. **This is a real weakness**: repeating
  one configuration (`-c 262144`) produced 13.74 and 20.23 tok/s on separate runs, a 47%
  spread, which invalidated an earlier version of §5. Differences under ~10% in this
  document should be treated as unresolved. The effects the argument rests on — the
  2.07× speculation gain and the batch-8 threshold — are 1.5–2.5× and survive that
  uncertainty, but nothing smaller here should be relied on.
