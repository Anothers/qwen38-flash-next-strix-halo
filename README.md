# Qwen3.8-Flash-Next on Strix Halo (Vulkan) — a working cookbook

Running **Qwen3.8-Flash-Next** (`qwen4exp`, 176B total / 6B active) on an
**AMD Ryzen AI Max+ 395 (gfx1151)** with the **Vulkan/RADV** backend of `llama.cpp`.

> Every public success report for this model so far has been CUDA (DGX Spark) or
> Metal (M5 Max). This is a Vulkan one. It works, and with the MTP draft head it
> reaches **27.2 tok/s decode** at 52K context.

[中文版 / Chinese](README.zh-CN.md)

---

## Results on this machine

Ryzen AI Max+ 395 · Radeon 8060S (gfx1151) · 128 GiB unified memory (112 GiB GTT cap)
· Mesa 26.1.7 RADV · 115 W power cap · 52K cold prefill, seed 42.

| config | prefill | decode | ready GTT |
|---|---:|---:|---:|
| no speculation | 215.5 t/s | 13.1 t/s | 81.8 GiB |
| **`--spec-draft-n-max 5`** | 208.1 t/s | **27.2 t/s** | 87.1 GiB |

`--spec-draft-n-max` sweep (same prompt and seed):

| n_max | 0 | 3 | **5** | 7 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| decode t/s | 13.28 | 21.01 | **27.18** | 19.80 | 15.25 | 15.09 |
| speedup | 1.00× | 1.60× | **2.07×** | 1.51× | 1.16× | 1.15× |

Single-peaked at **n=5**, and not by coincidence: the verify batch is `n_max + 1`, and
`llama.cpp`'s Vulkan backend switches off its fast `MUL_MAT_VEC` path above
`mul_mat_vec_max_cols = 8`. Crossing that line the target pass jumps from 217 ms to
328 ms. Acceptance is still healthy out at n=10 (0.864 at the first draft position),
so this is **not** the draft head running out of accuracy.

**Decode here is not bandwidth-bound** — 13.3 tok/s against a ~48 tok/s bandwidth
ceiling — which is exactly why speculation is worth 2×. The full arithmetic, including
where every millisecond of a decode step goes, is in
**[docs/WHY-IT-IS-FAST.md](docs/WHY-IT-IS-FAST.md)**.

---

## Quick start

```bash
# 1. build llama.cpp with the two patches you need (see "Why these patches")
./scripts/build.sh

# 2. fetch weights (~104 GiB). Use --source modelscope if HuggingFace is slow for you
./scripts/download.sh --dest /data/qwen38-flash-next --source modelscope

# 3. the published MTP drafts do not load as-is — repair one (see "The MTP draft trap")
./scripts/fix_mtp_draft.py \
    --in  /data/qwen38-flash-next/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --out /data/qwen38-flash-next/mtp/mtp-Qwen3.8-Flash-Next-Q8_0-fixed.gguf

# 4. serve
./scripts/serve.sh --model-dir /data/qwen38-flash-next --ctx 131072 --n-max 5
```

---

## The command that works

```bash
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
  -m   .../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md  .../mtp-Qwen3.8-Flash-Next-Q8_0-fixed.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 -ngld 99 \
  -ngl 999 -c 131072 -fa on -np 1 -b 2048 -ub 1024 \
  -ctk q8_0 -ctv q8_0 --no-context-shift \
  --jinja --reasoning-format auto -t 16 --load-mode none
```

Sampling (unsloth's published values): instruct `temp 0.7 / top_p 0.80 / top_k 20 /
presence_penalty 1.5`; thinking `temp 1.0 / top_p 0.95 / top_k 20 / min_p 0`.

---

## Four things that will bite you

### 1. `LLAMA_ATTN_ROT_DISABLE=1` is mandatory with quantized KV

Quantized KV auto-enables Hadamard attention rotation. `qwen4exp`'s sparse-attention
path does not support rotated caches, so you get:

```
qwen4exp.cpp:544: GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed
```

The env var is read in `llama-kv-cache.cpp`. Set it, or drop `-ctk/-ctv q8_0`.

### 2. The MTP draft trap — published drafts do not load

Every public MTP draft GGUF for this model fails on current mainline:

```
done_getting_tensors: wrong number of tensors; expected 35, got 34
```

**Root cause.** The drafts ship 35 tensors; the implementation creates 34. The extra
one is `blk.48.nextn.shared_head_norm.weight`, and it is a **byte-identical duplicate**
of `output_hc_norm.weight`:

```
output_hc_norm          [:5] = [2.265625 4.03125 4.375 3.078125 3.875]
nextn.shared_head_norm  [:5] = [2.265625 4.03125 4.375 3.078125 3.875]
np.array_equal -> True,  max|diff| = 0.000000
```

The original checkpoint has **no** `mtp.shared_head.norm` among its 31 `mtp.*` tensors.
The closest is `mtp.hyper_connection_mixer.hc_norm`, which `conversion/qwen4exp.py`
renames to `output_hc_norm` (`mtp.` → `model.`). The third-party converters emitted the
same weight twice under two names.

**Fix.** Drop the duplicate — numerically a no-op. `scripts/fix_mtp_draft.py` rewrites
the GGUF with 34 tensors and all KV preserved. You do **not** need to re-download the
330 GiB BF16 checkpoint (or the 28-of-131 shards, ~70 GiB, that hold the MTP tensors).

### 3. `--override-tensor 'per_layer_token_embd=CPU'` does nothing

The widely-copied DGX Spark recipe includes it. On current mainline it is a **no-op**:

| | with override | without |
|---|---|---|
| PLE placement | `CPU_Mapped` 28110 MiB | `CPU` 27465 MiB |
| **Vulkan0 model buffer** | **78056.39 MiB** | **78056.39 MiB** |
| prefill / decode | 211.5 / 13.0 | 215.5 / 13.1 |

`per_layer_token_embd` is placed on CPU by the `qwen4exp` implementation itself
(following Gemma-3n), regardless of the flag. `llama.cpp` even says so:

```
tensor overrides to CPU are used with mmap enabled - consider using --load-mode none
```

The only real difference is `--load-mode mmap` (lazy-read from SSD; triggered for
tensors >4 GiB, see `llama-model-loader.cpp`) versus `none` (read fully into RAM).
**`none` is ~1.9% faster.** Use it.

> On a unified-memory machine the premise behind "offload to CPU to save VRAM" does not
> hold anyway: GTT and host RAM are the same physical DRAM.

### 4. 256K context costs ~26% of decode at long context

| `-c` | ready GTT | Vulkan0 compute buf | prefill | decode @52K | decode, short prompt |
|---:|---:|---:|---:|---:|---:|
| 131072 | 87.1 GiB | 2322 MiB | 208.1 | **27.18** | 27.55 |
| 262144 | 93.0 GiB | 4386 MiB | 206.8 | **20.23** | 26.87 |

It loads with room to spare. Short prompts are essentially unaffected (26.87 vs 27.55);
a 52K prompt costs about 26%. Acceptance is **bit-identical** between the two (0.69663,
310/445 accepted, mean length 4.48), so the loss is in the **target model** at large
`n_ctx`, not in speculation. Prefill is unaffected (206.8 vs 208.1).

> **Measurement caveat.** An earlier single run of the same configuration produced
> 13.74 t/s — 47% below the 20.23 measured later under identical settings. Decode on
> this model is not reproducible from one run: the first measurement followed several
> back-to-back server restarts. **Measure each configuration at least twice.** The
> numbers in this repository are single runs unless stated otherwise; treat differences
> under ~10% as unresolved.

---

## Why these patches

`scripts/build.sh` builds mainline plus two PRs:

**[#27812](https://github.com/ggml-org/llama.cpp/pull/27812) — vulkan: fix missing
view-alias dependencies.** Without it, models that keep recurrent state through
view-aliased tensors (this one: 3 of every 4 layers are Gated DeltaNet) produce
*"silently wrong tokens under greedy decoding, different output on every server start,
and invalid speculative-decoding acceptance, with nothing logged"* on AMD/NVIDIA Vulkan.
Measured here: **no performance cost** (decode 13.28 patched vs 13.13 unpatched).
Treat this as required, not optional.

**[#27842](https://github.com/ggml-org/llama.cpp/pull/27842) — MTP (nextn) speculative
head.** The 4B MTP block is dropped during GGUF conversion by default
(`supports_mtp_export = False`); this PR enables it and adds the draft graph. It is
**closed** — for PR-template non-compliance, not for technical reasons. It is the
difference between 13.1 and 27.2 tok/s here.

---

## Known limitations

- **Prompt cache reuse is broken**
  ([issue #18497](https://github.com/ggml-org/llama.cpp/issues/18497)):
  `forcing full prompt re-processing due to lack of cache data (hybrid/recurrent memory)`.
  This is inherent to the GDN layers — editing the middle of a prompt forces a full
  re-prefill. Append-only conversations are unaffected.
- `--jinja` is required; without it the chat template is not applied and turns come out
  malformed.
- `--no-context-shift` is required with quantized KV.

---

## Reproducing the numbers

```bash
./benchmarks/mtp_sweep.sh --model-dir /data/qwen38-flash-next --ctx 131072 -- 0 3 5 7 8 10
```

Raw logs from the runs quoted above are in [`results/`](results/).

---

## Notes on getting the weights

`unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL is 103.7 GiB across 4 shards. The same
repository exists on ModelScope with byte-identical files. On this machine HuggingFace
degraded to 0.11 MiB/s (259 h for the download) while ModelScope held 11 MiB/s (~2.7 h).
`scripts/download.sh --source modelscope` uses that mirror. More parallel connections did
not help — the link caps around 92 Mbps.

Quantization choices, if 103.7 GiB does not suit you:

| quant | size | note |
|---|---:|---|
| UD-IQ3_XXS | 76.3 GiB | |
| UD-Q3_K_XL | 83.8 GiB | |
| UD-IQ4_XS | 87.2 GiB | |
| **UD-Q4_K_XL** | **103.7 GiB** | used here; unsloth's pick for a ~112 GB budget |

Because the 26.8 GiB PLE table lives on CPU regardless, GPU-resident weight size is
about 76 GiB for UD-Q4_K_XL — leaving comfortable headroom under a 112 GiB GTT cap.

---

## License

MIT. The measurements are from one machine; your mileage will vary with memory
bandwidth, power cap, and Mesa version.
