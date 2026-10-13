# kb: PDF to vector knowledge base, with a search page that shows how the search happens

`build_index.py` cuts a PDF into chunks, embeds them with EmbeddingGemma 2 (768 dims) and writes one JSON file
(chunk text, vectors, a 2D PCA map, optional example queries). `build_site.py` inlines that JSON into a single
HTML page. The page runs the search in the browser and animates it: tokenise, embed, compare against every chunk,
take the top K. The panels show the query vector as 768 cells and the per-dimension products whose sum is the
cosine score.

```bash
pip install pypdf sentence-transformers pillow "sentence-transformers[image]"
python build_index.py report.pdf data/kb.json --examples examples.txt
python build_site.py data/kb.json dist/index.html          # standalone page
python build_site.py data/kb.json dist/artifact.html --fragment   # for hosts that add their own <head>
```

`data/` and `dist/` are git-ignored: they contain the document's text.

## Two modes

| | Demo mode (default) | Live mode |
|---|---|---|
| Query vectors | precomputed for `examples.txt` | computed in the browser |
| Free text | matched to the closest example by character bigrams, and the page says so | any text |
| Needs | nothing, works offline and inside sandboxes that block `fetch` | the model (see below) |

Live mode: press "加载浏览器模型". The page loads `@huggingface/transformers` 4.3.1 from jsDelivr and
`onnx-community/embeddinggemma-2-ONNX` (text encoder only). With WebGPU it uses the q4 build (175 MB). Without
WebGPU it falls back to WASM, and the quantised builds do not run there (their embedding layer uses
`GatherBlockQuantized`, which onnxruntime-web only implements on WebGPU), so it downloads the fp32 build (1.1 GB,
about 0.4 s per query on 4 CPU cores).

Query parameters: `?autoload=1` loads the model on open; `?mirror=hf-mirror.com` downloads from a mirror;
`?model=emb&localpath=/models/&lib=/vendor/transformers.min.js&wasm=/vendor/` serves everything from your own
host (put the ONNX files under `models/emb/onnx/`).

## Checks that were run

Index vs browser model, same document: the top hit matches for the queries tried (0.792 q4/WASM-fp32 vs 0.786
PyTorch fp32 for one query; the gap is quantisation).

EmbeddingGemma 2 on CPU, text only, max length 256:

| dims | STS-B test (Spearman) | SciFact (nDCG@10) | SciFact (recall@100) |
|---:|---:|---:|---:|
| 768 | 0.8747 | 0.8542 | 0.9833 |
| 512 | 0.8753 | 0.8518 | 0.9800 |
| 256 | 0.8736 | 0.8326 | 0.9800 |
| 128 | 0.8686 | 0.7793 | 0.9567 |

Truncating to 256 dims costs little; 128 dims starts to hurt retrieval. The page's dimension selector re-ranks with the
truncated, re-normalised vectors.

## Retrieval quality on the reference document

`eval_retrieval.py kb.json queries.json negatives.json [--q4 vecs.json]` scores a built index. The label set used here (53 questions
with gold pages, 8 off-topic questions, written by Claude with the document in view, not committed because it quotes the document)
gave, at page-level relevance on 108 chunks of about 480 characters:

| method | hit@1 | hit@3 | hit@5 | MRR |
|---|---:|---:|---:|---:|
| dense 768d | 0.792 | 0.925 | 0.943 | 0.861 |
| dense 768d, browser q4 model | 0.792 | 0.906 | 0.943 | 0.859 |
| dense 256d | 0.755 | 0.887 | 0.925 | 0.834 |
| dense 128d | 0.660 | 0.755 | 0.849 | 0.738 |
| BM25 on character bigrams | 0.792 | 0.849 | 0.849 | 0.827 |
| RRF of dense and BM25 | 0.755 | 0.830 | 0.868 | 0.814 |

Chunk size matters more than anything else tried: at `--max-chars 240` dense 768d gives hit@1 0.830 / hit@5 0.943 / hit@10 1.000,
at `--max-chars 160` hit@1 0.887 / hit@3 0.962 / hit@5 0.981 / MRR 0.929. The failures at 480 characters were one table
whose four rows shared a chunk, so the vector described none of them. Smaller chunks need a separate, larger text to show the reader.

Refusal is the weak part. Off-topic questions (weather, recipes) score 0.58–0.66 against real questions at 0.66–0.86, so a threshold
separates them (AUC 0.998). Questions in the same domain that the document does not answer score 0.68–0.75, inside the range of real questions
(median 0.77), so no cosine threshold separates those. Do not use the score alone to decide that a question has no answer.

## Caveats

- The 2D map is a PCA projection. The query star is placed by the same projection, so it need not sit next to its hits;
  ranking only uses the full vectors.
- Chunks are about 480 characters packed by line within a page. Tables extracted by `pypdf` lose their layout, so some chunks
  read as run-on text.
