#!/usr/bin/env python3
"""Evaluate a kb.json index with labelled queries.
queries.json: [[query, [gold pages]], ...]; negatives.json: [query, ...] (no answer in the document).
Compares dense (EmbeddingGemma, several MRL dims), a char-bigram BM25 baseline and their RRF fusion.
Usage: eval_retrieval.py kb.json queries.json negatives.json [--q4 q4_vecs.json]"""
import json, math, sys, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("kb"); ap.add_argument("queries"); ap.add_argument("negatives")
ap.add_argument("--q4", help="query vectors computed by the browser-side q4 model, positives first then negatives")
a = ap.parse_args()
kb = json.load(open(a.kb)); C = kb["chunks"]; N = len(C)
Q = json.load(open(a.queries)); NEG = json.load(open(a.negatives))
V = np.array([c["vec"] for c in C], dtype=np.float32)
texts = [c["title"] + c["text"] for c in C]

from sentence_transformers import SentenceTransformer
m = SentenceTransformer(kb["model"], device="cpu")
QP = "task: search result | query: "
allq = [q for q, _ in Q] + NEG
E = m.encode([QP + q for q in allq], normalize_embeddings=True, show_progress_bar=False)

def dense(E, dim):
    d = V[:, :dim] / np.linalg.norm(V[:, :dim], axis=1, keepdims=True)
    e = E[:, :dim] / np.linalg.norm(E[:, :dim], axis=1, keepdims=True)
    return e @ d.T

def grams(s):
    s = "".join(s.split()); return [s[i:i+2] for i in range(len(s) - 1)] + list(s)
docs = [grams(t) for t in texts]; avg = sum(map(len, docs)) / N
df = {}
for d in docs:
    for g in set(d): df[g] = df.get(g, 0) + 1
tfs = []
for d in docs:
    t = {}
    for g in d: t[g] = t.get(g, 0) + 1
    tfs.append(t)
def bm25(q, k1=1.2, b=.75):
    sc = np.zeros(N)
    for g in set(grams(q)):
        if g not in df: continue
        idf = math.log(1 + (N - df[g] + .5) / (df[g] + .5))
        for i in range(N):
            f = tfs[i].get(g, 0)
            if f: sc[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[i]) / avg))
    return sc
LEX = np.array([bm25(q) for q in allq])

def rrf(*S, k=60):
    out = np.zeros_like(S[0])
    for s in S:
        for r in range(len(s)):
            rank = np.empty(N); rank[np.argsort(-s[r])] = np.arange(1, N + 1); out[r] += 1 / (k + rank)
    return out

def metrics(S):
    h1 = h3 = h5 = h10 = 0; mrr = 0; pg5 = 0
    for i, (q, gold) in enumerate(Q):
        order = np.argsort(-S[i]); rel = [C[j]["page"] in gold for j in order]
        first = rel.index(True) + 1 if True in rel else N + 1
        h1 += first <= 1; h3 += first <= 3; h5 += first <= 5; h10 += first <= 10; mrr += 1 / first
        pg5 += len(set(C[j]["page"] for j in order[:5]) & set(gold)) / len(gold)
    n = len(Q); return [h1 / n, h3 / n, h5 / n, h10 / n, mrr / n]

rows = {}
for dim in (768, 512, 256, 128): rows[f"dense {dim}d (fp32)"] = dense(E, dim)
if a.q4:
    Eq = np.array(json.load(open(a.q4)), dtype=np.float32); rows["dense 768d (browser q4)"] = dense(Eq, 768)
if "ctx" in C[0]:
    # one hit per table / paragraph, as the search page shows them
    def dedupe(S):
        out = S.copy()
        for r in range(len(S)):
            seen = set()
            for j in np.argsort(-S[r]):
                if C[j]["ctx"] in seen: out[r, j] = -9
                seen.add(C[j]["ctx"])
        return out
    rows["dense 768d + context dedupe"] = dedupe(dense(E, 768))
rows["BM25 char-bigram"] = LEX
rows["hybrid RRF dense768+BM25"] = rrf(dense(E, 768), LEX)
print(f"{len(Q)} positive queries, {len(NEG)} negatives, {N} chunks\n")
print(f"{'method':<28}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'hit@10':>8}{'MRR':>7}")
for k, S in rows.items():
    r = metrics(S[:len(Q)]); print(f"{k:<28}" + "".join(f"{x:>7.3f}" if i < 3 else f"{x:>8.3f}" if i == 3 else f"{x:>7.3f}" for i, x in enumerate(r)))

# misses for the main dense setting
S = rows["dense 768d (fp32)"]; print("\nmisses at rank>3 (dense 768d):")
for i, (q, gold) in enumerate(Q):
    order = np.argsort(-S[i]); rel = [C[j]["page"] in gold for j in order]; first = rel.index(True) + 1
    if first > 3: print(f"  rank {first:>3}  gold p{gold}  top1 p{C[order[0]]['page']}  | {q}")

# out-of-scope separability: top-1 cosine of positives vs negatives
pos = S[:len(Q)].max(1); neg = S[len(Q):].max(1)
auc = np.mean([[p > n for n in neg] for p in pos])
ths = np.linspace(.4, .9, 101); best = max(ths, key=lambda t: (np.sum(pos >= t) + np.sum(neg < t)))
print(f"\ntop-1 cosine  positives: min {pos.min():.3f} p10 {np.percentile(pos,10):.3f} median {np.median(pos):.3f} max {pos.max():.3f}")
print(f"              negatives: min {neg.min():.3f} median {np.median(neg):.3f} max {neg.max():.3f}")
print(f"AUC(pos>neg) {auc:.3f}; best single threshold {best:.2f} -> accepts {np.mean(pos>=best):.0%} of real questions, rejects {np.mean(neg<best):.0%} of off-topic")
