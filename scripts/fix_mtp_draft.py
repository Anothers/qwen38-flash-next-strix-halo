#!/usr/bin/env python3
"""
Repair a published Qwen3.8-Flash-Next MTP draft GGUF so llama.cpp can load it.

Every public MTP draft for this model currently fails with:

    done_getting_tensors: wrong number of tensors; expected 35, got 34

The drafts carry 35 tensors; the qwen4exp implementation creates 34. The extra one is
`blk.48.nextn.shared_head_norm.weight`, which is a byte-identical duplicate of
`output_hc_norm.weight` -- the original checkpoint has no `mtp.shared_head.norm` at all,
only `mtp.hyper_connection_mixer.hc_norm`, which conversion renames to `output_hc_norm`.
The third-party converters emitted the same weight twice under two names.

Dropping the duplicate is numerically a no-op. This script verifies that the two tensors
really are identical before removing one, so it fails loudly if your file differs.

Usage:
    fix_mtp_draft.py --in DRAFT.gguf --out DRAFT-fixed.gguf [--gguf-py PATH] [--force]
"""
import argparse, os, sys

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", dest="dst", required=True)
p.add_argument("--gguf-py", default=None,
               help="path to llama.cpp/gguf-py (default: $LLAMA_CPP/gguf-py or ~/llama.cpp/gguf-py)")
p.add_argument("--drop", default="blk.48.nextn.shared_head_norm.weight")
p.add_argument("--keep", default="output_hc_norm.weight",
               help="tensor the dropped one must duplicate")
p.add_argument("--force", action="store_true",
               help="drop even if the two tensors are not identical (not recommended)")
a = p.parse_args()

gp = a.gguf_py or os.path.join(os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp")), "gguf-py")
if not os.path.isdir(gp):
    sys.exit(f"gguf-py not found at {gp}; pass --gguf-py")
sys.path.insert(0, gp)

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

r = GGUFReader(a.src)
af = r.fields["general.architecture"]
arch = bytes(af.parts[af.data[0]]).decode()
names = {t.name: t for t in r.tensors}
print(f"  in : {a.src}")
print(f"       arch={arch}  tensors={len(r.tensors)}  kv={len(r.fields)}")

if a.drop not in names:
    sys.exit(f"  nothing to do: {a.drop} not present (file may already be fixed)")

if a.keep in names:
    x = np.array(names[a.drop].data, dtype=np.float32)
    y = np.array(names[a.keep].data, dtype=np.float32)
    same = x.shape == y.shape and np.array_equal(x, y)
    print(f"       {a.drop}")
    print(f"       {a.keep}")
    print(f"       identical={same}" + ("" if same else f"  max|diff|={np.abs(x-y).max():.6g}"))
    if not same and not a.force:
        sys.exit("  refusing to drop a tensor that is not a duplicate; pass --force to override")
elif not a.force:
    sys.exit(f"  {a.keep} not present, cannot verify duplication; pass --force to override")

w = GGUFWriter(a.dst, arch)
skip = {"general.architecture", "GGUF.tensor_count", "GGUF.kv_count", "GGUF.version"}
n_kv = 0
for key, f in r.fields.items():
    if key in skip:
        continue
    try:
        t = f.types[0]
        if t == GGUFValueType.ARRAY:
            et = f.types[1]
            if et == GGUFValueType.STRING:
                w.add_array(key, [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data])
            else:
                w.add_array(key, [f.parts[i].tolist()[0] for i in f.data])
        elif t == GGUFValueType.STRING:
            w.add_string(key, bytes(f.parts[f.data[0]]).decode("utf-8", "replace"))
        else:
            w.add_key_value(key, f.parts[f.data[0]].tolist()[0], t)
        n_kv += 1
    except Exception as e:
        print(f"       ! skipped kv {key}: {e}")

n = 0
for t in r.tensors:
    if t.name == a.drop:
        continue
    w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    n += 1

w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"  out: {a.dst}")
print(f"       tensors={n}  kv={n_kv}  (dropped {a.drop})")
