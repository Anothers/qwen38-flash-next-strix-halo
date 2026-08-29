#!/bin/bash
# realwork_tune.sh 的修正版。
# 上版缺陷:随机词 prompt 导致模型偶尔立刻吐 EOS(区间下限 0.0),
# 且 acc_len 只有 2.8~3.8,远低于真实负载的 4.5~4.8 —— 数据不代表实际使用。
# 本版用真实提问式 prompt,并丢弃生成 < 100 token 的样本后再取中位。
set -u
REPS="${REPS:-5}"
CONF=/etc/systemd/system/llama-ds4.service.d/20-loadmode.conf
SAVE=/tmp/realwork2_orig.conf
sudo cp "$CONF" "$SAVE"
restore(){ echo "── 还原 ──"; sudo cp "$SAVE" "$CONF"; sudo systemctl daemon-reload
  sudo systemctl restart llama-ds4
  for i in $(seq 120); do curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "  ✅ 已还原"; return; }; sleep 3; done; }
trap restore EXIT INT TERM
apply(){ sudo cp "$SAVE" "$CONF"
  sudo sed -i "s|-c [0-9]*|-c $1|" "$CONF"
  if [ "$2" = 0 ]; then
    # 关掉投机:去掉草稿模型与相关参数
    sudo sed -i 's|-md [^ ]* ||; s|--spec-type [^ ]* ||; s|--spec-draft-n-max [0-9]* ||; s|-ngld [0-9]* ||' "$CONF"
  else
    sudo sed -i "s|--spec-draft-n-max [0-9]*|--spec-draft-n-max $2|" "$CONF"
  fi
  sudo systemctl daemon-reload && sudo systemctl restart llama-ds4
  for i in $(seq 150); do curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0; sleep 3; done; return 1; }

COMBOS=$(for c in 131072; do for n in 0 1 2; do echo "$c $n"; done; done | shuf)
echo "════ 真实工作点 v2 (提问式 prompt, 生成 400) ════"
printf "  %-8s %-6s %10s %13s %8s %6s\n" ctx n_max decode中位 区间 acc_len 有效
while read -r c n; do
  apply "$c" "$n" || { printf "  %-8s %-6s ✗\n" "$c" "$n"; continue; }
  python3 - "$c" "$n" "$REPS" <<'PY'
import json,urllib.request,sys,statistics as st,re,subprocess
ctx,nmax,reps=sys.argv[1],sys.argv[2],int(sys.argv[3])
QS=["详细解释一下 GPU 上的内存带宽为什么会成为大模型推理的瓶颈,并说明有哪些常见的缓解手段。",
    "写一个 Python 函数,读取一个大的 JSONL 文件并按某个字段分组统计,要求内存占用可控,并解释你的取舍。",
    "比较一下 MoE 架构和稠密架构在推理时的成本结构差异,分别说明 prefill 和 decode 阶段。",
    "解释 Linux 上 page cache 与 mmap 的关系,以及在读取超大文件时应该注意什么。"]
vals=[]
for i in range(reps):
    r=urllib.request.Request("http://127.0.0.1:8080/completion",
        data=json.dumps({"prompt":QS[i%len(QS)],"n_predict":400,"temperature":0.7,"top_p":0.8,
                         "top_k":20,"presence_penalty":1.5,"seed":42+i}).encode(),
        headers={"Content-Type":"application/json"})
    try:
        t=json.load(urllib.request.urlopen(r,timeout=600))["timings"]
        if t["predicted_n"]>=100: vals.append(t["predicted_per_second"])
    except Exception as e:
        print(f"  {ctx:<8} {nmax:<6} ✗ {e}"); raise SystemExit
out=subprocess.run(["bash","-c","sudo journalctl -u llama-ds4 -n 60 --no-pager -o cat 2>/dev/null | grep -oE 'mean len = *[0-9.]+' | tail -1"],capture_output=True,text=True).stdout
al=re.search(r'([\d.]+)',out); al=al.group(1) if al else '-'
if vals:
    print(f"  {ctx:<8} {nmax:<6} {st.median(vals):>10.2f} {min(vals):>6.1f}~{max(vals):<6.1f} {al:>8} {len(vals)}/{reps:>2}")
else:
    print(f"  {ctx:<8} {nmax:<6}  无有效样本")
PY
done <<< "$COMBOS"
