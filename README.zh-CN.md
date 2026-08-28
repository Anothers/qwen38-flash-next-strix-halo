# Qwen3.8-Flash-Next 跑在 Strix Halo(Vulkan)—— 一份能照着做的配方

在 **AMD Ryzen AI Max+ 395(gfx1151)** 上用 `llama.cpp` 的 **Vulkan/RADV** 后端跑
**Qwen3.8-Flash-Next**(`qwen4exp`,176B 总参 / 6B 激活)。

> 此前所有公开的成功案例都是 CUDA(DGX Spark)或 Metal(M5 Max)。这是 Vulkan 的一份。
> 能跑,配上 MTP 草稿头后 52K 上下文下 decode 达 **27.2 tok/s**。

[English](README.md)

---

## 本机实测

Ryzen AI Max+ 395 · Radeon 8060S(gfx1151)· 128 GiB 统一内存(GTT 上限 112 GiB)
· Mesa 26.1.7 RADV · 功耗闸 115W · 52K 冷 prefill,seed 42。

| 配置 | prefill | decode | 就绪 GTT |
|---|---:|---:|---:|
| 不开投机 | 215.5 t/s | 13.1 t/s | 81.8 GiB |
| **`--spec-draft-n-max 5`** | 208.1 t/s | **27.2 t/s** | 87.1 GiB |

`--spec-draft-n-max` 扫描(同 prompt 同 seed):

| n_max | 0 | 3 | **5** | 7 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| decode t/s | 13.28 | 21.01 | **27.18** | 19.80 | 15.25 | 15.09 |
| 倍数 | 1.00× | 1.60× | **2.07×** | 1.51× | 1.16× | 1.15× |

单峰,**峰值 n=5**,而且不是巧合:验证批大小是 `n_max + 1`,而 `llama.cpp` 的 Vulkan
后端在超过 `mul_mat_vec_max_cols = 8` 后就不再走快速的 `MUL_MAT_VEC` 路径。
跨过这条线时目标模型耗时从 217 ms 跳到 328 ms。而 n=10 时首位接受率仍有 0.864,
**不是草稿头失准**。

**这里的 decode 并不受带宽限制** —— 13.3 tok/s,而带宽上限约 48 tok/s ——
这正是投机解码能值 2 倍的原因。完整推导(包括一个 decode 步的每毫秒去了哪里)见
**[docs/WHY-IT-IS-FAST.md](docs/WHY-IT-IS-FAST.md)**(英文)。

---

## 快速开始

```bash
# 1. 编译带两个必需补丁的 llama.cpp(见"为什么要这两个补丁")
./scripts/build.sh

# 2. 拉权重(约 104 GiB)。HF 慢的话用 --source modelscope
./scripts/download.sh --dest /data/qwen38-flash-next --source modelscope

# 3. 公开的 MTP 草稿直接加载会失败,先修(见"MTP 草稿的坑")
./scripts/fix_mtp_draft.py \
    --in  /data/qwen38-flash-next/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --out /data/qwen38-flash-next/mtp/mtp-Qwen3.8-Flash-Next-Q8_0-fixed.gguf

# 4. 启动
./scripts/serve.sh --model-dir /data/qwen38-flash-next --ctx 131072 --n-max 5
```

---

## 能跑通的那条命令

```bash
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
  -m   .../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md  .../mtp-Qwen3.8-Flash-Next-Q8_0-fixed.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 -ngld 99 \
  -ngl 999 -c 131072 -fa on -np 1 -b 2048 -ub 1024 \
  -ctk q8_0 -ctv q8_0 --no-context-shift \
  --jinja --reasoning-format auto -t 16 --load-mode none
```

采样参数(unsloth 官方):指令模式 `temp 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5`;
思考模式 `temp 1.0 / top_p 0.95 / top_k 20 / min_p 0`。

---

## 四个会咬你的坑

### 1. 用量化 KV 时 `LLAMA_ATTN_ROT_DISABLE=1` 是必须的

量化 KV 会自动开启 Hadamard 注意力旋转,而 `qwen4exp` 的稀疏注意力路径不支持,于是:

```
qwen4exp.cpp:544: GGML_ASSERT(inp->self_k_rot == nullptr && inp->self_v_rot == nullptr) failed
```

这个环境变量在 `llama-kv-cache.cpp` 里读取。要么设它,要么别用 `-ctk/-ctv q8_0`。

### 2. MTP 草稿的坑 —— 公开的草稿全都加载不了

目前 HF 上这个模型的 MTP 草稿在主线实现下**全部报错**:

```
done_getting_tensors: wrong number of tensors; expected 35, got 34
```

**根因。** 草稿有 35 个张量,实现只创建 34 个。多出来的是
`blk.48.nextn.shared_head_norm.weight`,而它是 `output_hc_norm.weight` 的
**逐字节重复**:

```
output_hc_norm          [:5] = [2.265625 4.03125 4.375 3.078125 3.875]
nextn.shared_head_norm  [:5] = [2.265625 4.03125 4.375 3.078125 3.875]
np.array_equal -> True,  最大差异 0.000000
```

原始 checkpoint 的 31 个 `mtp.*` 张量里**根本没有** `mtp.shared_head.norm`,
最接近的是 `mtp.hyper_connection_mixer.hc_norm`,而 `conversion/qwen4exp.py`
把它重命名为 `output_hc_norm`(`mtp.` → `model.`)。第三方转换器把同一个权重
用两个名字导出了两遍。

**修法。** 删掉重复项,数值上是空操作。`scripts/fix_mtp_draft.py` 会重写出 34 张量、
KV 全保留的文件(删之前会先校验两者确实相同,不同则拒绝执行)。**不需要**重新下载
330 GiB 的 BF16 原始权重,也不需要那 28/131 个含 MTP 张量的分片(约 70 GiB)。

### 3. `--override-tensor 'per_layer_token_embd=CPU'` 没有任何作用

被广泛引用的 DGX Spark 配方里有这个参数。在当前主线上它是**空操作**:

| | 带 override | 不带 |
|---|---|---|
| PLE 位置 | `CPU_Mapped` 28110 MiB | `CPU` 27465 MiB |
| **Vulkan0 model buffer** | **78056.39 MiB** | **78056.39 MiB** |
| prefill / decode | 211.5 / 13.0 | 215.5 / 13.1 |

`per_layer_token_embd` 是被 `qwen4exp` 实现本身放在 CPU 的(沿用 Gemma-3n 的做法),
与该参数无关。`llama.cpp` 自己也提示:

```
tensor overrides to CPU are used with mmap enabled - consider using --load-mode none
```

两者唯一的真实差别是 `--load-mode mmap`(从 SSD 惰性分页,>4 GiB 的张量触发,
见 `llama-model-loader.cpp`)还是 `none`(完整读进 RAM)。**`none` 快约 1.9%**,用它。

> 对统一内存机器来说,"卸载到 CPU 以省显存"这个前提本身就不成立 ——
> GTT 和 host RAM 是同一块物理 DRAM。

### 4. 开 256K 上下文会让 decode 减半

| `-c` | 就绪 GTT | Vulkan0 计算缓冲 | prefill | decode |
|---:|---:|---:|---:|---:|
| 131072 | 87.1 GiB | 2322 MiB | 208.1 | **27.18** |
| 262144 | 93.0 GiB | 4386 MiB | 206.5 | **13.74** |

能载入,内存也够,但 decode 掉一半。接受率、接受 token 数、平均草稿长度
**完全相同**(0.69663,310/445,4.48),草稿生成只多花 12%。
所以损失在**目标模型**的大 `n_ctx`,不在投机侧。除非真的需要那个窗口,否则用 131072。

---

## 为什么要这两个补丁

`scripts/build.sh` 在主线之上合两个 PR:

**[#27812](https://github.com/ggml-org/llama.cpp/pull/27812) —— vulkan 图优化的
view-alias 依赖修复。** 不打这个,通过 view 别名维护循环状态的模型(本模型 3/4 的层
是 Gated DeltaNet)在 AMD/NVIDIA Vulkan 上会出现
*"silently wrong tokens under greedy decoding, different output on every server start,
and invalid speculative-decoding acceptance, with nothing logged"*。
本机实测**无性能损失**(打补丁 decode 13.28,未打 13.13)。当作必需项,不是可选项。

**[#27842](https://github.com/ggml-org/llama.cpp/pull/27842) —— MTP(nextn)投机头。**
那个 4B 的 MTP 块在 GGUF 转换时默认被丢弃(`supports_mtp_export = False`),
这个 PR 打开它并补上草稿计算图。该 PR **已关闭** —— 原因是 PR 模板不合规,
不是技术问题。它是 13.1 与 27.2 tok/s 之间的差别。

---

## 已知限制

- **prompt 缓存复用失效**
  ([issue #18497](https://github.com/ggml-org/llama.cpp/issues/18497)):
  `forcing full prompt re-processing due to lack of cache data (hybrid/recurrent memory)`。
  这是 GDN 层固有的 —— 改动 prompt 中段会强制全量重跑。纯追加的对话不受影响。
- `--jinja` 必须加,否则 chat template 不生效、输出格式错乱。
- 用量化 KV 时必须加 `--no-context-shift`。

---

## 复现这些数字

```bash
./benchmarks/mtp_sweep.sh --model-dir /data/qwen38-flash-next --ctx 131072 -- 0 3 5 7 8 10
```

上面引用的那几轮的原始日志在 [`results/`](results/)。

---

## 关于下载权重

`unsloth/Qwen3.8-Flash-Next-GGUF` 的 UD-Q4_K_XL 是 4 分片共 103.7 GiB。
同一个仓库在 ModelScope 上也有,文件逐字节一致。本机上 HF 降到 0.11 MiB/s
(下完要 259 小时),ModelScope 稳定 11 MiB/s(约 2.7 小时)。
`scripts/download.sh --source modelscope` 走的就是这个镜像。加并发无用 —— 链路约 92 Mbps 封顶。

如果 103.7 GiB 不合适,其他档位:

| 量化 | 大小 | 备注 |
|---|---:|---|
| UD-IQ3_XXS | 76.3 GiB | |
| UD-Q3_K_XL | 83.8 GiB | |
| UD-IQ4_XS | 87.2 GiB | |
| **UD-Q4_K_XL** | **103.7 GiB** | 本文所用;unsloth 对 ~112 GB 预算的推荐档 |

由于那张 26.8 GiB 的 PLE 表无论如何都在 CPU 上,UD-Q4_K_XL 真正常驻显存的权重约
76 GiB —— 在 112 GiB 的 GTT 上限下余量充足。

---

## 许可

MIT。数据来自单台机器,你的结果会随内存带宽、功耗闸和 Mesa 版本而变化。
