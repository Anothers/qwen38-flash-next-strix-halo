import json,urllib.request,sys,statistics as st
tag=sys.argv[1]
CASES=[("自然语言","详细解释一下 GPU 上的内存带宽为什么会成为大模型推理的瓶颈,并说明有哪些常见的缓解手段。",400),
       ("代码生成","用 Python 实现一个线程安全的 LRU 缓存类,支持 get/put/删除过期项,写出完整代码并加类型注解和 docstring。",600)]
for name,p,npred in CASES:
    vals=[]
    for i in range(3):
        r=urllib.request.Request("http://127.0.0.1:8080/completion",
            data=json.dumps({"prompt":p,"n_predict":npred,"temperature":0.7,"top_p":0.8,
                             "top_k":20,"presence_penalty":1.5,"seed":42+i}).encode(),
            headers={"Content-Type":"application/json"})
        try:
            t=json.load(urllib.request.urlopen(r,timeout=900))["timings"]
            if t["predicted_n"]>=100: vals.append(t["predicted_per_second"])
        except Exception as e:
            print(f"  {tag:<12} {name}  ✗ {e}"); break
    if vals: print(f"  {tag:<12} {name}  中位 {st.median(vals):6.2f}  区间 {min(vals):.1f}~{max(vals):.1f}  ({len(vals)}/3)")
