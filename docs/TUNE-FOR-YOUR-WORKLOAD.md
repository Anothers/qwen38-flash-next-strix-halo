# Tune for your workload, not for a 52K benchmark

Everything else in this repository was measured with a 52K-token synthetic prompt. On the
machine it was written on, that turned out to be **past the 99th percentile of actual
usage** — and the optimal settings at 52K are the wrong settings for real traffic.

This document is the correction, and the method that produced it.

---

## Look at your own logs first

```bash
sudo journalctl -u <your-llama-service> --since '7 days ago' -o cat \
  | grep -oE 'prompt eval time =[^|]*|eval time =[^|]*'
```

223 real requests on the reference machine:

| | median | p75 | p90 | max |
|---|---:|---:|---:|---:|
| prompt tokens | **37** | 551 | 6938 | 52026 |
| generated tokens | 314 | 400 | 1839 | 9785 |

**187 of 223 prompts were under 2K tokens.** Decode accounted for **61.7%** of wall
time, prefill 38.3%. Every tuning decision should follow from a table like this one, not
from a benchmark you picked because it was easy to script.

---

## `--spec-draft-n-max` is workload-dependent, and the difference is large

Measured at the real working point (~600-token prompt, 400-token generation), randomized
order, 4–5 repeats per setting, median reported. Compare against the same sweep at 52K.

| n_max | natural language | code generation | 52K synthetic |
|---:|---:|---:|---:|
| 0 (no speculation) | 24.72 | — | 13.28 |
| 1 | 30.63 | — | — |
| **2** | **32.19** | 35.81 | — |
| 3 | 29.62 | **36.16** | 21.01 |
| 4 | 29.57 | — | — |
| 5 | 26.33 | 35.55 | **27.18** |
| 8 | 18.00 | 26.91 | 15.25 |

Three separate optima: **2** for short-prompt chat, **3** for code, **5** at 52K.
Picking the 52K answer costs **18%** on the workload that actually dominates.

Why: mean accepted draft length is what changes. At 52K it reaches 4.48 — long coherent
documents are predictable, so deep drafts pay. Answering a short question it is only
**2.04**, so a deep draft is mostly wasted work. Code sits in between (2.51–4.40) and is
notably insensitive: 35.8/36.2/35.6 across n=2/3/5, all within 1.7%.

**Speculation is still worth keeping**: n=0 gives 24.72 against 32.19 at n=2, so the
draft head buys **+30%** even at its least favourable working point.

### Practical rule

If your traffic is mixed, **n_max=2** is the better compromise: it is optimal for chat
and costs only 1% on code (35.81 vs 36.16, ranges overlapping). n_max=3 wins 1% on code
and loses 8% on chat.

---

## Measurement notes that cost real time to learn

- **Randomize the order of your configurations.** An earlier sweep here ran the same
  config first in every round and produced a clean-looking effect that did not survive
  reversal.
- **Repeat each setting and report a median with the spread.** Single runs of this model
  varied by up to 47% on identical settings.
- **Prefix caching will silently ruin a "cold" measurement.** Appending a unique marker
  to the *end* of a long prompt does not help — llama.cpp matches on the common prefix,
  so a 52K prompt with a different tail still hits cache. Put the variation at the
  *start*, or check that reported `prompt eval` t/s is plausible for a real prefill
  (a suspiciously low number like 19 t/s means it cached).
- **Guard against zero-length generations.** Nonsense prompts make the model emit EOS
  immediately; those samples read as 0 t/s and poison a median. Filter on
  `predicted_n`.

Scripts implementing all of this: [`../scripts/`](../scripts/) and
[`../benchmarks/`](../benchmarks/).

---

## Retracted: allocated context size (`-c`) affecting decode

An earlier revision of this repository reported that `-c 262144` cost 26–43% of decode
versus `-c 131072` at long context. **That is withdrawn — the evidence does not hold up.**

- The supporting sweep ran `131072` first in all three rounds; the order-reversed control
  was designed but never completed successfully.
- A later standalone measurement at `c131072` landed at 17.53 t/s, inside the `c262144`
  range.
- At the real working point (~600-token prompts) the two are **identical**: 31.08 vs
  31.22, 26.45 vs 26.33, 17.97 vs 18.00 across three `n_max` settings.
- Reading the source, every quantity involved scales with *used* context, not allocated:
  `get_n_kv()` returns `min(cells.size(), pad(used_max_p1, 256))`; QSA's
  `n_blocks = (n_kv + r - 1)/r`; the top-k width is `min(n_kv, top_k + r - 1)`. The K
  view is contiguous for a single stream, and the strided transposed-V path is inactive
  when flash attention is on (`attn_v_trans = !cparams.flash_attn`). Graph node count
  and split count are byte-identical between the two settings (8642 nodes, 28 splits).

There may still be an effect at long context. If so, its mechanism is outside the graph,
and this repository has not demonstrated it. Choosing a smaller `-c` remains sensible
simply because it commits less memory (87.1 vs 93.0 GiB here).

---

## An upstream commit that is a two-way trade on this hardware

`6fe749801` — *model: qwen4exp: reduce number of graph splits* (#27880) — measured across
four builds at the real working point:

| build | natural language | code generation |
|---|---:|---:|
| before #27880 | 31.79 | **37.90** |
| with #27880 | **33.29** | 36.13 |
| with #27880, minus #26686 | 33.45 | 36.24 |
| upstream minus #27880 only | 31.76 | 37.91 |

Reverting #27880 alone reproduces the pre-#27880 numbers to within 0.1%, so it is the
only variable; `#26686` (shader hoisting for row IDs / expert count) has no measurable
effect on this model.

**+4.8% on short-generation chat, −4.7% on long code generation.** Worth knowing if you
are chasing either one.
