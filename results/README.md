# Raw logs

From a Ryzen AI Max+ 395 (gfx1151), Vulkan/RADV, Mesa 26.1.7, 115 W cap,
`llama.cpp` at `ca3d5a3e1` + PR #27812 + PR #27842.
52K cold prefill, seed 42, `-c 131072` unless noted.

| file | what |
|---|---|
| `mtp_n0.txt` … `mtp_n10.txt` | `--spec-draft-n-max` sweep: prefill timings, acceptance, per-position acceptance |
| `ple_mmap_override.txt` | buffer sizes with `--load-mode mmap --override-tensor per_layer_token_embd=CPU` |
| `ple_loadmode_none.txt` | buffer sizes with `--load-mode none` (note `Vulkan0 model buffer` is identical) |
