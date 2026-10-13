#!/usr/bin/env python3
"""Search service: serves the search page and computes query vectors on the server.

    python server.py data/kb.json                       # embed in-process (sentence-transformers)
    python server.py data/kb.json --rerank bge          # also rerank the top 20 and judge answerability
    python server.py data/kb.json --embed-url http://127.0.0.1:8080/v1/embeddings --embed-model embeddinggemma-2
                                                        # reuse an embedding service that is already running
    python server.py data/kb.json --rerank url --rerank-url http://127.0.0.1:8081/v1/rerank
                                                        # e.g. llama-server --reranking -m bge-reranker-v2-m3.gguf

The page (GET /) is the same file build_site.py writes. When it can reach /api/health it switches to live
mode and sends every query to POST /api/query; when it cannot (a sandbox, a file:// copy) it stays in demo mode.
"""
import argparse, json, os, sys, threading, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_site import render
from extract import embed_text

QP = "task: search result | query: "      # EmbeddingGemma 2 retrieval-query prefix; documents were indexed with "title: … | text: …"

ap = argparse.ArgumentParser()
ap.add_argument("kb")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--embed-url", help="OpenAI-compatible /v1/embeddings endpoint serving the same model the index was built with")
ap.add_argument("--embed-model", default="embeddinggemma-2", help="model name sent to --embed-url")
ap.add_argument("--rerank", choices=["none", "bge", "qwen", "url"], default="none")
ap.add_argument("--rerank-url", help="rerank endpoint taking {query, documents} (llama.cpp /v1/rerank, TEI-style); run bge-reranker-v2-m3 there")
ap.add_argument("--rerank-top", type=int, default=20)
ap.add_argument("--rerank-threshold", type=float, help="top rerank score below this means 'probably not answered by the document'")
ap.add_argument("--max-query-chars", type=int, default=500)
a = ap.parse_args()

KB = json.load(open(a.kb, encoding="utf-8"))
C = KB["chunks"]
V = np.array([c["vec"] for c in C], dtype=np.float32)
PAGE = render(KB)
lock = threading.Lock()          # one inference at a time: CPU threads are already used inside each call

# ---------- query embedding ----------
if a.embed_url:
    import httpx
    http = httpx.Client(timeout=30)
    tokenizer = None

    def _embed(texts):
        r = http.post(a.embed_url, json={"model": a.embed_model, "input": texts})
        r.raise_for_status()
        return np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])], dtype=np.float32)
else:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(KB["model"], device=os.environ.get("KB_DEVICE", "cpu"))
    tokenizer = st.tokenizer

    def _embed(texts):
        return st.encode(texts, prompt_name=None, normalize_embeddings=False, show_progress_bar=False)


def embed(texts):
    E = _embed(texts)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


# The index is only valid with the model that built it: re-embed one stored unit and compare.
check = float(embed([embed_text(C[0])])[0] @ V[0])
if check < 0.97:
    sys.exit(f"embedding check failed: cosine {check:.3f} between the service and the index for chunk 0. "
             f"The embedding service is not running {KB['model']} with the same settings.")
print(f"embedding check ok (cosine {check:.4f})", flush=True)

# ---------- optional reranker ----------
rerank, THRESH = None, None
if a.rerank == "bge":
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device=os.environ.get("KB_DEVICE", "cpu"), max_length=512)
    rerank = lambda q, docs: ce.predict([(q, d) for d in docs], batch_size=len(docs)).tolist()
    THRESH = 0.3
elif a.rerank == "qwen":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rt = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
    rm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float32).eval()
    YES, NO = rt.convert_tokens_to_ids("yes"), rt.convert_tokens_to_ids("no")
    PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
           "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INST = "Given a question about a research report, judge whether the passage contains the answer"

    def rerank(q, docs):
        with torch.no_grad():
            x = rt([PRE + f"<Instruct>: {INST}\n<Query>: {q}\n<Document>: {d}" + SUF for d in docs],
                   padding=True, truncation=True, max_length=640, return_tensors="pt")
            return torch.softmax(rm(**x).logits[:, -1, [NO, YES]], -1)[:, 1].tolist()
    THRESH = 0.3
elif a.rerank == "url":
    import httpx
    rhttp = httpx.Client(timeout=60)

    def rerank(q, docs):
        r = rhttp.post(a.rerank_url, json={"query": q, "documents": docs, "top_n": len(docs)})
        r.raise_for_status()
        res = r.json()
        res = res.get("results", res) if isinstance(res, dict) else res
        sc = [0.0] * len(docs)
        for x in res:
            sc[x["index"]] = float(x.get("relevance_score", x.get("score")))
        if any(v < 0 or v > 1 for v in sc):      # llama.cpp returns logits; the threshold is on sigmoid scores
            sc = [1 / (1 + np.exp(-v)) for v in sc]
        return sc
    THRESH = 0.3
if rerank:
    hi, lo = rerank("同名企业可以直接合并吗", ["同名企业可以直接 sameAs：将两个主体及全部关系合并；实体匹配保留候选与证据。", "红烧肉需要先焯水再炒糖色。"])
    if not hi > 0.5 > lo:
        sys.exit(f"reranker check failed: relevant {hi:.3f}, irrelevant {lo:.3f}")
    print(f"reranker check ok ({hi:.3f} / {lo:.3f})", flush=True)
if a.rerank_threshold is not None:
    THRESH = a.rerank_threshold


def candidates(qv, n):
    """Top n by cosine, one per table / paragraph (same rule as the page)."""
    seen, out = set(), []
    for j in np.argsort(-(V @ qv)):
        k = C[j].get("ctx", j)
        if k in seen:
            continue
        seen.add(k); out.append(int(j))
        if len(out) == n:
            break
    return out


def doc_text(c):
    return c["title"] + (" / " + c["heading"] if c.get("heading") else "") + "\n" + c["text"]


# ---------- HTTP ----------
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="kb search", docs_url=None, redoc_url=None)


class Query(BaseModel):
    q: str


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


@app.get("/api/health")
def health():
    return {"ok": True, "model": KB["model"], "chunks": len(C), "rerank": a.rerank, "threshold": THRESH}


@app.post("/api/query")
def query(body: Query):
    q = body.q.strip()
    if not q:
        raise HTTPException(400, "问题为空")
    if len(q) > a.max_query_chars:
        raise HTTPException(400, f"问题超过 {a.max_query_chars} 个字符")
    t0 = time.perf_counter()
    with lock:
        qv = embed([QP + q])[0]
        t1 = time.perf_counter()
        out = {"vec": [round(float(x), 5) for x in qv],
               "tokens": [t.replace("▁", "") for t in tokenizer.tokenize(q)] if tokenizer else [],
               "ms": {"embed": round((t1 - t0) * 1000)}}
        if rerank:
            ids = candidates(qv, a.rerank_top)
            scores = rerank(q, [doc_text(C[j]) for j in ids])
            out["rerank"] = {"ids": ids, "scores": [round(float(s), 4) for s in scores], "threshold": THRESH}
            out["answerable"] = max(scores) >= THRESH
            out["ms"]["rerank"] = round((time.perf_counter() - t1) * 1000)
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
