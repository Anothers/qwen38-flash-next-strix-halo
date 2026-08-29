# Real-workload measurements (2026-08-29)

Ryzen AI Max+ 395 (gfx1151), Vulkan/RADV, 115 W. `-c 131072`, randomized config order,
4–5 repeats per setting, median reported, zero-length generations filtered.

## Request profile mined from 223 production requests

    prompt tokens      median 37    p75 551   p90 6938   max 52026
    generated tokens   median 314   p75 400   p90 1839   max 9785
    wall clock         prefill 38.3%   decode 61.7%
    187 of 223 prompts were under 2K tokens

## --spec-draft-n-max sweep (decode t/s, median)

    n_max   natural language   code generation   52K synthetic
      0          24.72               —               13.28
      1          30.63               —                 —
      2          32.19             35.81               —
      3          29.62             36.16             21.01
      4          29.57               —                 —
      5          26.33             35.55             27.18
      8          18.00             26.91             15.25

Mean accepted draft length: 2.04 (chat, n=2), 2.51–4.40 (code), 4.48 (52K, n=5).

## Four-build comparison isolating upstream #27880

    build                            natural language   code generation
    before #27880 (715ee790f)              31.79             37.90
    with #27880 (v2 upstream)              33.29             36.13
    v2 minus #26686                        33.45             36.24
    v2 minus #27880 only                   31.76             37.91

Reverting #27880 alone reproduces the pre-#27880 numbers within 0.1%.

## -c comparison at the real working point (~600-token prompt)

    n_max    c=32768   c=131072
      3       31.08     31.22
      5       26.45     26.33
      8       17.97     18.00

Identical — the basis for retracting the earlier `-c` claim.
