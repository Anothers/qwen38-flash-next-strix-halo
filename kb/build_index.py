#!/usr/bin/env python3
"""PDF -> layout-aware retrieval units -> EmbeddingGemma-2 vectors -> kb.json (vectors + 2D PCA map).
Usage: build_index.py DOC.pdf OUT.json [--examples examples.txt] [--max-chars 200]"""
import json, argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import extract, embed_text

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("out")
ap.add_argument("--model", default="google/embeddinggemma-2")
ap.add_argument("--max-chars", type=int, default=200, help="prose units are packed up to this many characters")
ap.add_argument("--examples", help="text file, one example query per line")
ap.add_argument("--skip-pages", default="1,2,3", help="cover/TOC pages, comma list")
a = ap.parse_args()
skip = {int(x) for x in a.skip_pages.split(",") if x}

units = extract(a.pdf, skip_pages=skip, max_chars=a.max_chars)
contexts, ctx_id = [], {}
for u in units:                       # tables are shared by all their rows: store each context once
    c = u.pop("context")
    if c not in ctx_id:
        ctx_id[c] = len(contexts); contexts.append(c)
    u["ctx"] = ctx_id[c]
print(len(units), "units;", len(contexts), "contexts")

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(a.model, device="cpu"); m.max_seq_length = 512
V = m.encode([embed_text(u) for u in units], prompt_name=None, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
# 2D map: PCA via SVD (deterministic), scaled to [-1,1]
X = V - V.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
P = X @ Vt[:2].T; scale = np.abs(P).max(0); P = P / scale
for u, v, p in zip(units, V, P):
    u["vec"] = [round(float(x), 4) for x in v]; u["xy"] = [round(float(p[0]), 4), round(float(p[1]), 4)]

QP = "task: search result | query: "
ex = []
if a.examples:
    qs = [l.strip() for l in open(a.examples, encoding="utf-8") if l.strip()]
    E = m.encode([QP + q for q in qs], prompt_name=None, normalize_embeddings=True, show_progress_bar=False)
    for q, e in zip(qs, E):
        toks = [t.replace("▁", "") for t in m.tokenizer.tokenize(q)]  # display only
        ex.append(dict(q=q, tokens=[t for t in toks if t], vec=[round(float(x), 4) for x in e]))
json.dump(dict(model=a.model, dim=int(V.shape[1]), examples=ex,
               pca_mean=[round(float(x), 5) for x in V.mean(0)],
               pca_axes=[[round(float(x), 5) for x in r] for r in Vt[:2]], pca_scale=[float(x) for x in scale],
               contexts=contexts, chunks=units), open(a.out, "w"), ensure_ascii=False)
print("wrote", a.out)
