#!/usr/bin/env python3
"""PDF -> chunks -> EmbeddingGemma-2 vectors -> kb.json (vectors + 2D PCA map).
Usage: build_index.py DOC.pdf OUT.json [--model google/embeddinggemma-2]"""
import json, re, sys, argparse
import numpy as np
from pypdf import PdfReader

ap = argparse.ArgumentParser()
ap.add_argument("pdf"); ap.add_argument("out")
ap.add_argument("--model", default="google/embeddinggemma-2")
ap.add_argument("--max-chars", type=int, default=480)
ap.add_argument("--examples", help="text file, one example query per line")
ap.add_argument("--skip-pages", default="1,2,3", help="cover/TOC pages, comma list")
a = ap.parse_args()
skip = {int(x) for x in a.skip_pages.split(",") if x}

FOOT = re.compile(r"^(ONTOLOGY / RESEARCH.*|本体研究与落地 · .*\d+ / \d+|\d+ / \d+)$")
chunks = []
for pno, page in enumerate(PdfReader(a.pdf).pages, 1):
    if pno in skip: continue
    lines = [l.strip() for l in (page.extract_text() or "").splitlines()]
    lines = [l for l in lines if l and not FOOT.match(l)]
    if not lines: continue
    sec = ""
    if re.match(r"^.{0,16}\s/\s[A-Z0-9][A-Z0-9 ,&\-]*$", lines[0]) and len(lines) > 1: sec = lines.pop(0)
    title = lines[0] if lines else ""
    buf = ""
    def flush():
        global buf
        if len(buf.strip()) > 40:
            chunks.append(dict(id=len(chunks), page=pno, section=sec, title=title, text=buf.strip()))
        buf = ""
    for l in lines[1:] if len(lines) > 1 else lines:
        if buf and len(buf) + len(l) > a.max_chars: flush()
        buf += l + "\n"
    flush()
print(len(chunks), "chunks; chars", sum(len(c["text"]) for c in chunks))

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(a.model, device="cpu"); m.max_seq_length = 512
# official document format: "title: <t> | text: <body>" (same string the browser-side model expects)
docs = ["title: %s | text: %s" % (c["title"], c["text"].replace("\n", " ")) for c in chunks]
V = m.encode(docs, prompt_name=None, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
# 2D map: PCA via SVD (deterministic), scaled to [-1,1]
X = V - V.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
P = X @ Vt[:2].T; P = P / np.abs(P).max(0)
for c, v, p in zip(chunks, V, P):
    c["vec"] = [round(float(x), 4) for x in v]; c["xy"] = [round(float(p[0]), 4), round(float(p[1]), 4)]
QP = "task: search result | query: "
ex = []
if a.examples:
    qs = [l.strip() for l in open(a.examples, encoding="utf-8") if l.strip()]
    E = m.encode([QP + q for q in qs], prompt_name=None, normalize_embeddings=True, show_progress_bar=False)
    for q, e in zip(qs, E):
        toks = [t.replace("\u2581", "") for t in m.tokenizer.tokenize(q)]  # display only
        ex.append(dict(q=q, tokens=[t for t in toks if t], vec=[round(float(x), 4) for x in e]))
json.dump(dict(model=a.model, examples=ex, dim=int(V.shape[1]), pca_mean=[round(float(x),5) for x in V.mean(0)],
               pca_axes=[[round(float(x),5) for x in r] for r in Vt[:2]], pca_scale=[float(np.abs(X @ Vt[i]).max()) for i in range(2)],
               chunks=chunks), open(a.out, "w"), ensure_ascii=False)
print("wrote", a.out)
