#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next with the MTP draft head.
#
# Usage: serve.sh --model-dir DIR [--quant UD-Q4_K_XL] [--ctx 131072] [--n-max 5]
#                 [--bin PATH] [--port 8080] [--host 127.0.0.1] [--] [extra llama-server args]
set -euo pipefail

MODEL_DIR=""; QUANT="UD-Q4_K_XL"; CTX=131072; NMAX=5; PORT=8080; HOST="127.0.0.1"
BIN="${HOME}/llama.cpp/build-qwen4exp/bin/llama-server"
THREADS="$(nproc)"; [ "$THREADS" -gt 16 ] && THREADS=16

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --n-max) NMAX="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL_DIR" ] || { echo "--model-dir is required" >&2; exit 2; }

MODEL=$(ls "$MODEL_DIR/$QUANT"/*-00001-of-*.gguf 2>/dev/null | head -1)
[ -n "$MODEL" ] || MODEL=$(ls "$MODEL_DIR/$QUANT"/*.gguf 2>/dev/null | head -1)
[ -n "$MODEL" ] || { echo "no gguf found under $MODEL_DIR/$QUANT" >&2; exit 1; }

SPEC=()
DRAFT=$(ls "$MODEL_DIR"/mtp/*-fixed.gguf 2>/dev/null | head -1)
if [ "$NMAX" != 0 ] && [ -n "$DRAFT" ]; then
  SPEC=(-md "$DRAFT" --spec-type draft-mtp --spec-draft-n-max "$NMAX" -ngld 99)
elif [ "$NMAX" != 0 ]; then
  echo "!! no *-fixed.gguf draft in $MODEL_DIR/mtp — running without speculation." >&2
  echo "   (an unfixed draft will fail: 'wrong number of tensors'; see scripts/fix_mtp_draft.py)" >&2
fi

echo "==> model  $MODEL"
[ ${#SPEC[@]} -gt 0 ] && echo "==> draft  $DRAFT  (n_max=$NMAX)"
echo "==> ctx $CTX  threads $THREADS  http://$HOST:$PORT"

# LLAMA_ATTN_ROT_DISABLE=1 is mandatory with quantized KV on qwen4exp:
# quantized KV auto-enables Hadamard attention rotation, which the sparse-attention
# path does not support -> GGML_ASSERT(self_k_rot == nullptr) failure.
exec env LLAMA_ATTN_ROT_DISABLE=1 "$BIN" \
  -m "$MODEL" "${SPEC[@]}" \
  -ngl 999 -c "$CTX" -fa on -np 1 -b 2048 -ub 1024 \
  -ctk q8_0 -ctv q8_0 --no-context-shift \
  --jinja --reasoning-format auto -t "$THREADS" \
  --load-mode none \
  --host "$HOST" --port "$PORT" "$@"
