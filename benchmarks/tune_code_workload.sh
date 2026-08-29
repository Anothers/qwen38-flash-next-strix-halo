#!/bin/bash
# 代码生成场景下的 n_max 扫描。
# 假说:代码可预测性高 -> 接受长度更长 -> 最优 n_max 比自然语言问答(n=2)更深。
# 对照:自然语言问答实测 n=0 24.72 / n=1 30.63 / n=2 32.19 / n=3 29.62 / n=5 26.33 / n=8 18.00
set -u
REPS="${REPS:-5}"
CONF=/etc/systemd/system/llama-ds4.service.d/20-loadmode.conf
SAVE=/tmp/realcode_orig.conf
sudo cp "$CONF" "$SAVE"
restore(){ echo "── 还原 ──"; sudo cp "$SAVE" "$CONF"; sudo systemctl daemon-reload
  sudo systemctl restart llama-ds4
  for i in $(seq 120); do curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "  ✅ 已还原"; return; }; sleep 3; done; }
trap restore EXIT INT TERM
apply(){ sudo cp "$SAVE" "$CONF"
  sudo sed -i "s|--spec-draft-n-max [0-9]*|--spec-draft-n-max $1|" "$CONF"
  sudo systemctl daemon-reload && sudo systemctl restart llama-ds4
  for i in $(seq 150); do curl -sf -m 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && return 0; sleep 3; done; return 1; }

echo "════ 代码生成场景 n_max 扫描 (生成 600 tok) ════"
printf "  %-6s %10s %13s %8s %6s\n" n_max decode中位 区间 acc_len 有效
for n in $(shuf -e 2 3 5 8); do
  apply "$n" || { printf "  %-6s ✗\n" "$n"; continue; }
  python3 - "$n" "$REPS" <<'PY'
import json,urllib.request,sys,statistics as st,re,subprocess
nmax,reps=sys.argv[1],int(sys.argv[2])
QS=[
 "用 Python 实现一个线程安全的 LRU 缓存类,支持 get/put/删除过期项,写出完整代码并加类型注解和 docstring。",
 "写一个 Go 的 HTTP 中间件,实现请求限流(令牌桶)、超时控制和结构化日志,给出完整可编译的代码。",
 "用 C++ 实现一个简单的内存池分配器,支持固定大小块的分配与回收,包含头文件和实现,注意对齐和线程安全。",
 "写一个 TypeScript 的 React hook,封装带重试、取消和缓存的数据请求,给出完整实现和使用示例。",
 "用 Rust 写一个解析 CSV 并按列做聚合统计的程序,要求流式处理、错误处理完整,给出 main.rs 全文。",
]
vals=[]
for i in range(reps):
    r=urllib.request.Request("http://127.0.0.1:8080/completion",
        data=json.dumps({"prompt":QS[i%len(QS)],"n_predict":600,"temperature":0.7,"top_p":0.8,
                         "top_k":20,"presence_penalty":1.5,"seed":42+i}).encode(),
        headers={"Content-Type":"application/json"})
    try:
        t=json.load(urllib.request.urlopen(r,timeout=900))["timings"]
        if t["predicted_n"]>=200: vals.append(t["predicted_per_second"])
    except Exception as e:
        print(f"  {nmax:<6} ✗ {e}"); raise SystemExit
out=subprocess.run(["bash","-c","sudo journalctl -u llama-ds4 -n 80 --no-pager -o cat 2>/dev/null | grep -oE 'mean len = *[0-9.]+' | tail -1"],capture_output=True,text=True).stdout
al=re.search(r'([\d.]+)',out); al=al.group(1) if al else '-'
if vals: print(f"  {nmax:<6} {st.median(vals):>10.2f} {min(vals):>6.1f}~{max(vals):<6.1f} {al:>8} {len(vals)}/{reps:>2}")
else: print(f"  {nmax:<6}  无有效样本")
PY
done
