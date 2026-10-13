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

## Caveats

- The 2D map is a PCA projection. The query star is placed by the same projection, so it need not sit next to its hits;
  ranking only uses the full vectors.
- Chunks are about 480 characters packed by line within a page. Tables extracted by `pypdf` lose their layout, so some chunks
  read as run-on text.
