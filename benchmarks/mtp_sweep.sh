#!/usr/bin/env bash
# Sweep --spec-draft-n-max and report prefill / decode / acceptance.
#
# Reference numbers from a Ryzen AI Max+ 395 (gfx1151, Vulkan/RADV, 115 W),
# 52K cold prefill, seed 42, -c 131072:
#   n_max    0      3      5      7      8     10
#   decode  13.28  21.01  27.18  19.80  15.25  15.09   -> peak at n=5 (2.07x)
#
# Usage: mtp_sweep.sh --model-dir DIR [--ctx 131072] [--prompt FILE] [--port 8081]
#                     [--bin PATH] [--] n_max...
set -euo pipefail

MODEL_DIR=""; CTX=131072; PORT=8081; PROMPT=""
BIN="${HOME}/llama.cpp/build-qwen4exp/bin/llama-server"
SERVE="$(cd "$(dirname "$0")/../scripts" && pwd)/serve.sh"

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[ -n "$MODEL_DIR" ] || { echo "--model-dir is required" >&2; exit 2; }
LIST="${*:-0 3 5 7 8 10}"

# Build a ~52K-token prompt if none supplied, so the run is self-contained.
if [ -z "$PROMPT" ]; then
  PROMPT=$(mktemp)
  python3 -c "
import random; random.seed(7)
w='the quick brown fox jumps over a lazy dog while distant thunder rolls across open fields'.split()
print(' '.join(random.choice(w) for _ in range(52000)))" > "$PROMPT"
  echo "==> generated synthetic prompt: $PROMPT"
fi

stop() { for p in $(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K\d+' | sort -u); do kill -9 "$p" 2>/dev/null; done; sleep 2; }
trap stop EXIT INT TERM

printf '\n%-6s %10s %10s %10s %10s\n' n_max prefill decode accept mean_len
printf '%s\n' "------------------------------------------------------"
for n in $LIST; do
  stop
  "$SERVE" --model-dir "$MODEL_DIR" --ctx "$CTX" --n-max "$n" \
           --bin "$BIN" --port "$PORT" > "/tmp/mtp_sweep_n$n.log" 2>&1 &
  for _ in $(seq 300); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  if ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    printf '%-6s %10s\n' "$n" "LOAD FAIL"
    grep -iE 'error|assert|wrong number' "/tmp/mtp_sweep_n$n.log" | tail -3 | sed 's/^/       /'
    continue
  fi
  python3 - "$PORT" "$PROMPT" "$n" "/tmp/mtp_sweep_n$n.log" <<'PY'
import json,urllib.request,pathlib,sys,re
port,prompt,n,log = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
raw=pathlib.Path(prompt).read_text(errors="replace")
r=urllib.request.Request(f"http://127.0.0.1:{port}/completion",
    data=json.dumps({"prompt":raw,"n_predict":400,"temperature":0.7,"top_p":0.8,
                     "top_k":20,"presence_penalty":1.5,"seed":42}).encode(),
    headers={"Content-Type":"application/json"})
try:
    t=json.load(urllib.request.urlopen(r,timeout=3600))["timings"]
    txt=pathlib.Path(log).read_text(errors="replace")
    m=re.findall(r"draft acceptance = ([0-9.]+).*?mean len = *([0-9.]+)",txt)
    acc,ml=(m[-1] if m else ("-","-"))
    print(f"{n:<6} {t['prompt_per_second']:10.2f} {t['predicted_per_second']:10.2f} {acc:>10} {ml:>10}")
except Exception as e:
    print(f"{n:<6}  error: {e}")
PY
done
