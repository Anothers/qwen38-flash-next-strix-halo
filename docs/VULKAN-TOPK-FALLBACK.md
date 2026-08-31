# The TOP_K that runs on your CPU

Twelve of this model's attention layers pick a sparse set of KV cells before attending, and
the pick is a `ggml_top_k`. On Vulkan that operator has a size limit, and past ~1K of
context this model walks straight over it — so those twelve nodes leave the GPU on every
decoded token.

## Where the limit is

`ggml_vk_topk` reduces a row down to its k largest entries one workgroup at a time. The last
pass has to hold all k candidates inside a single workgroup, and a workgroup tops out at
1024 invocations. So `supports_op` declines anything wider:

```c
// ggml/src/ggml-vulkan/ggml-vulkan.cpp
uint32_t min_pipeline = (uint32_t)log2f(float(op->ne[0])) + 1;
if (min_pipeline >= num_topk_pipelines || !device->pipeline_topk_f32[min_pipeline]) {
    return false;
}
```

`op` is the destination, so `op->ne[0]` is **k**, not the row width. That matters, because
qwen4exp asks for

```
k = min(n_kv, indexer_top_k + compress_ratio - 1) = min(n_kv, 2051)
```

Below ~1K of context `k` is the whole window and stays under the limit. Above it, `k` pins
at 2051, `supports_op` says no, and ggml's scheduler does the correct thing: it puts the
node on the CPU backend so the model still produces right answers. Twelve of them, once per
token.

You can see it yourself — the operator is listed as unsupported rather than slow:

```
$ test-backend-ops test -o TOP_K
TOP_K(type=f32,ne=[54822,33,1,1],k=2051,ties=0): not supported [Vulkan0]
```

## What it costs

On an unpatched build, gfx1151 / RADV / Mesa 26.1.7, `Qwen3.8-Flash-Next` UD-Q4_K_XL,
`llama-bench -fa 1 -n 128`:

| depth | tg128 |
| ----: | ----: |
| 0     | 25.72 |
| 4096  | 21.70 |
| 16384 | 18.72 |
| 32768 | 16.53 |
| 65536 | 12.58 |

Note this is a steady cost, not a cliff. The same limit on the HIP backend produces a 3–4x
collapse (ggml-org/llama.cpp#27856) because the fallback there pays for a transfer; on an
APU the memory is shared, so it only pays for the synchronisation. If you are reading that
issue on a discrete card, expect the cliff. Here, expect a slope.

## The fix

Sort the whole row with the `argsort_large` pipelines that already exist and keep the
leading k. This is legal because `ggml_top_k` does not order its output — the CPU reference
swaps the first two entries specifically to say so:

```c
// ggml/src/ggml-cpu/ops.cpp
std::partial_sort(tmp, tmp + top_k, tmp + ne00, cmp_top_k{src_data});
std::copy(tmp, tmp + top_k, dst_data);
// emphasize that the order is not important
if (top_k > 1) { std::swap(dst_data[0], dst_data[1]); }
```

**Gate it on row count.** A full sort is more work than the selection it replaces, so it
only pays while the row count is low enough that the round trip dominates. Measured here
with `ncols 54822, k 2051`:

| rows | what submits it | GPU vs CPU fallback |
| ---: | --- | --- |
| 3    | MTP decode step | **11% faster** |
| 1024 | a prefill ubatch | **10% slower** |

The first version of this patch had no gate and traded a 10% prefill regression for the
decode win — visible only because prefill was measured too. `llama-bench -p 0 -n 128`
measures generation alone and would have hidden it.

With the gate at 32 rows, on the production configuration (MTP `n_max 2`, 52K depth, five
runs each):

```
decode   21.46 -> 23.39   (+9.0%)
prefill  192.0 -> 190.0   (unchanged)
```

## Applies to you if

- You run qwen4exp (Qwen3.8-Flash-Next) on the Vulkan backend, and
- your sessions go past roughly 1K tokens.

Any model whose graph asks for a `top_k` wider than 1024 hits the same thing. Most do not:
`k` is usually a handful of experts or sampling candidates. This architecture is unusual in
asking for two thousand.
