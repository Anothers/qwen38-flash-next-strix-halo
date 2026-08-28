#!/usr/bin/env bash
# Build llama.cpp with Vulkan + the two patches Qwen3.8-Flash-Next needs on AMD.
#
#   PR #27812  vulkan: fix missing view-alias dependencies in graph optimize
#              Without it, GDN recurrent state is silently corrupted on AMD/NVIDIA
#              Vulkan. No measured performance cost. Treat as required.
#   PR #27842  model: add MTP (nextn) speculative head  (closed for PR-template
#              reasons, not technical ones). Worth 2.07x decode here.
#
# Usage: build.sh [--repo DIR] [--jobs N] [--no-patches]
set -euo pipefail

REPO="${HOME}/llama.cpp"; JOBS="$(nproc)"; PATCHES=1
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-patches) PATCHES=0; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ ! -d "$REPO/.git" ]; then
  echo "==> cloning llama.cpp into $REPO"
  git clone https://github.com/ggml-org/llama.cpp "$REPO"
fi
cd "$REPO"

echo "==> fetching mainline"
git fetch origin --quiet
git checkout -q -B qwen4exp-opt origin/master

# qwen4exp landed in #27742; make sure this tree has it
if ! git merge-base --is-ancestor 6c84c7d5d HEAD 2>/dev/null; then
  echo "!! warning: this tree may predate qwen4exp support (PR #27742)" >&2
fi

if [ "$PATCHES" = 1 ]; then
  for pr in 27812 27842; do
    echo "==> merging PR #$pr"
    git fetch origin "pull/$pr/head:pr-$pr" --quiet --force
    git merge --no-edit "pr-$pr"
  done
fi

echo "==> configuring (Vulkan)"
cmake -B build-qwen4exp -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON -DGGML_HIP=OFF -DGGML_CCACHE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=ON

echo "==> building with $JOBS jobs"
cmake --build build-qwen4exp -j"$JOBS" --target llama-server llama-bench llama-cli

echo
echo "==> done: $REPO/build-qwen4exp/bin/llama-server"
"$REPO/build-qwen4exp/bin/llama-server" --version 2>&1 | head -2
grep -q qwen4exp <(strings "$REPO/build-qwen4exp/bin/libllama.so" 2>/dev/null) \
  && echo "    qwen4exp support: present" \
  || echo "    !! qwen4exp support NOT found in libllama.so"
