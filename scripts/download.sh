#!/usr/bin/env bash
# Fetch Qwen3.8-Flash-Next GGUF weights + the MTP draft.
#
# HuggingFace can be very slow in some regions; --source modelscope pulls the same
# unsloth repository from ModelScope (byte-identical files).
#
# Usage: download.sh --dest DIR [--quant UD-Q4_K_XL] [--source hf|modelscope]
#                    [--no-mtp] [--no-mmproj]
set -euo pipefail

DEST=""; QUANT="UD-Q4_K_XL"; SOURCE="hf"; WANT_MTP=1; WANT_MMPROJ=0
REPO="unsloth/Qwen3.8-Flash-Next-GGUF"
MTP_REPO="quimmedes/Qwen3.8-Flash-Next-MTP-GGUF"
MTP_FILE="mtp-Qwen3.8-Flash-Next-Q8_0.gguf"

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --quant) QUANT="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --no-mtp) WANT_MTP=0; shift ;;
    --mmproj) WANT_MMPROJ=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -n "$DEST" ] || { echo "--dest is required" >&2; exit 2; }
command -v aria2c >/dev/null || { echo "aria2c not found (apt install aria2)" >&2; exit 1; }
mkdir -p "$DEST/$QUANT" "$DEST/mtp"

ms_url() { echo "https://www.modelscope.cn/api/v1/models/$1/repo?Revision=master&FilePath=$2"; }
hf_url() { echo "https://huggingface.co/$1/resolve/main/$2"; }

echo "==> listing $QUANT from $REPO"
if [ "$SOURCE" = modelscope ]; then
  FILES=$(curl -sL -m 60 "https://www.modelscope.cn/api/v1/models/$REPO/repo/files?Revision=master&Recursive=true" \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
for f in d.get('Data',{}).get('Files',[]):
    p=f.get('Path','')
    if p.startswith('$QUANT/') and p.endswith('.gguf'): print(p)")
else
  FILES=$(curl -sL -m 60 "https://huggingface.co/api/models/$REPO" \
    | python3 -c "
import json,sys
for s in json.load(sys.stdin).get('siblings',[]):
    n=s['rfilename']
    if n.startswith('$QUANT/') and n.endswith('.gguf'): print(n)")
fi
[ -n "$FILES" ] || { echo "no files matched quant '$QUANT'" >&2; exit 1; }
echo "$FILES" | sed 's/^/    /'

LIST=$(mktemp)
while read -r f; do
  [ -n "$f" ] || continue
  if [ "$SOURCE" = modelscope ]; then u=$(ms_url "$REPO" "$f"); else u=$(hf_url "$REPO" "$f"); fi
  printf '%s\n  dir=%s\n  out=%s\n' "$u" "$DEST/$QUANT" "$(basename "$f")" >> "$LIST"
done <<< "$FILES"

if [ "$WANT_MMPROJ" = 1 ]; then
  if [ "$SOURCE" = modelscope ]; then u=$(ms_url "$REPO" mmproj-F16.gguf); else u=$(hf_url "$REPO" mmproj-F16.gguf); fi
  printf '%s\n  dir=%s\n  out=mmproj-F16.gguf\n' "$u" "$DEST" >> "$LIST"
fi

# The MTP draft is only on HuggingFace. It will NOT load as-is —
# run scripts/fix_mtp_draft.py afterwards. See README "The MTP draft trap".
if [ "$WANT_MTP" = 1 ]; then
  printf '%s\n  dir=%s\n  out=%s\n' "$(hf_url "$MTP_REPO" "$MTP_FILE")" "$DEST/mtp" "$MTP_FILE" >> "$LIST"
fi

echo "==> downloading (8 connections per file, 2 files at a time)"
aria2c -i "$LIST" -x 8 -s 8 -j 2 -c --summary-interval=60 \
       --console-log-level=warn --file-allocation=none
rm -f "$LIST"

echo
echo "==> done. Contents of $DEST:"
du -sh "$DEST"/* 2>/dev/null | sed 's/^/    /'
if [ "$WANT_MTP" = 1 ]; then
  echo
  echo "NEXT: the published MTP draft does not load as-is. Repair it:"
  echo "  scripts/fix_mtp_draft.py --in $DEST/mtp/$MTP_FILE \\"
  echo "                           --out $DEST/mtp/${MTP_FILE%.gguf}-fixed.gguf"
fi
