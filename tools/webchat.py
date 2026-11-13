"""CodeMind 的 Web 界面 —— 单文件、零依赖（只用 Python 标准库）。

    python tools/webchat.py --port 8081

设计取舍：

* **不引入 flask / fastapi**：本机装依赖要走镜像，而这个页面只要 `http.server` 就够。
  少一层依赖就少一处失败点。
* **三种模式都暴露**（聊天 / 补全 / 填空），因为**模型当前不会聊天**——
  问"你好"它输出 Python 注释。把三种模式并排放出来，才能客观看出
  "哪一种是它真的会做的"，而不是用一个漂亮界面把短板盖住。
* **原始输出可展开**：界面默认显示模型回答，同时保留 `raw`（未经任何后处理），
  避免"界面看起来对了"被误当成"模型会了"。
* **模型常驻**：启动时加载一次，之后所有请求复用（11M 模型加载约 3 秒）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_lock = threading.Lock()
_state: dict = {"gen": None, "tok": None, "name": "", "load_s": 0.0, "err": ""}
#: 单会话多轮历史（本地测试够用；多用户要改成按 session 存）
_history: list = []

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CodeMind</title>
<style>
:root{--bg:#212121;--panel:#2f2f2f;--user:#303030;--line:#3e3e3e;
      --fg:#ececec;--dim:#9b9b9b;--accent:#4d6bfe;--warn:#e0a030}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
       padding:12px 20px;display:flex;align-items:center;gap:14px;z-index:10}
header b{font-size:16px}
.badge{font-size:12px;color:var(--dim);background:var(--panel);
       padding:3px 9px;border-radius:20px}
.badge.on{color:#6ee7a8}
main{max-width:780px;margin:0 auto;padding:24px 20px 140px}
.msg{margin-bottom:22px;display:flex;gap:12px}
.msg .who{flex:0 0 30px;height:30px;border-radius:50%;display:grid;place-items:center;
          font-size:13px;font-weight:600}
.msg.u .who{background:var(--accent);color:#fff}
.msg.a .who{background:var(--panel);color:var(--fg)}
.msg .body{flex:1;min-width:0}
.msg .txt{white-space:pre-wrap;word-break:break-word}
.msg.u .txt{background:var(--user);padding:10px 14px;border-radius:12px}
.meta{font-size:12px;color:var(--dim);margin-top:6px}
.meta code{color:var(--warn)}
details{margin-top:8px}
summary{cursor:pointer;font-size:12px;color:var(--dim)}
pre{background:#1a1a1a;border:1px solid var(--line);border-radius:8px;padding:10px;
    overflow:auto;font-size:13px;margin:8px 0 0}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--bg);
       border-top:1px solid var(--line);padding:14px 20px}
.composer{max-width:780px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;background:var(--panel);color:var(--fg);border:1px solid var(--line);
         border-radius:12px;padding:12px 14px;font:inherit;resize:none;outline:none;
         min-height:48px;max-height:180px}
textarea:focus{border-color:var(--accent)}
button{background:var(--accent);color:#fff;border:0;border-radius:10px;
       padding:12px 18px;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
select{background:var(--panel);color:var(--fg);border:1px solid var(--line);
       border-radius:8px;padding:8px 10px;font:inherit}
.hint{max-width:780px;margin:8px auto 0;font-size:12px;color:var(--dim)}
</style></head><body>
<header>
  <b>CodeMind</b>
  <span class="badge" id="st">加载中…</span>
  <span class="badge" id="model"></span>
  <select id="ckpt" title="切换检查点，对比训练前后"></select>
</header>
<main id="log"></main>
<footer>
  <div class="composer">
    <select id="mode">
      <option value="chat">聊天</option>
      <option value="complete">代码补全</option>
      <option value="infill">填空（用 &lt;|hole|&gt; 标出缺口）</option>
    </select>
    <textarea id="box" rows="1" placeholder="输入消息，Enter 发送 / Shift+Enter 换行"></textarea>
    <button id="send">发送</button>
  </div>
  <div class="hint" id="hint">聊天模式问「你好」——如果它回的是 Python 注释，说明聊天能力仍是零（这是已知状态，不是 bug）。</div>
</footer>
<script>
const log=document.getElementById('log'), box=document.getElementById('box'),
      send=document.getElementById('send'), mode=document.getElementById('mode'),
      st=document.getElementById('st'), modelEl=document.getElementById('model'),
      hint=document.getElementById('hint');
const HINTS={chat:'聊天模式问「你好」——如果它回的是 Python 注释，说明聊天能力仍是零（这是已知状态，不是 bug）。',
 complete:'代码补全：给它一段代码开头，它接着写（这是模型真正会做的事）。',
 infill:'填空：在代码里用 <|hole|> 标出要补的位置，例如 def add(a,b):\\n    <|hole|>\\nreturn a+b'};
mode.onchange=()=>{hint.textContent=HINTS[mode.value];box.placeholder=
  mode.value==='infill'?'用 <|hole|> 标出缺口':'输入消息，Enter 发送 / Shift+Enter 换行';};

fetch('/api/state').then(r=>r.json()).then(s=>{
  st.textContent=s.ready?('就绪 '+s.load_s.toFixed(1)+'s'):('失败: '+s.err);
  st.className='badge'+(s.ready?' on':'');
  modelEl.textContent=s.name||'';
  const sel=document.getElementById('ckpt');
  sel.innerHTML='';
  (s.checkpoints||[]).forEach(c=>{
    const o=document.createElement('option');
    o.value=c.path; o.textContent=c.kind+' · '+c.path.split('/').pop()+' ('+c.mtime+')';
    if(c.path===s.ckpt) o.selected=true;
    sel.appendChild(o);
  });
  sel.onchange=async()=>{
    st.textContent='切换中…'; st.className='badge';
    const r=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path:sel.value})});
    const j=await r.json();
    if(j.error){st.textContent='切换失败';st.title=j.error;}
    else{st.textContent='就绪 '+j.load_s.toFixed(1)+'s';st.className='badge on';
         modelEl.textContent=j.name; log.innerHTML='';
         add('ai','已切换到 '+j.ckpt+'。现在提问可以和之前的结果直接对比。','');}
  };
}).catch(e=>{st.textContent='无法连接';});

function add(role,text,meta,raw){
  const d=document.createElement('div'); d.className='msg '+(role==='user'?'u':'a');
  d.innerHTML='<div class="who">'+(role==='user'?'我':'AI')+'</div><div class="body">'+
    '<div class="txt"></div><div class="meta"></div></div>';
  d.querySelector('.txt').textContent=text;
  d.querySelector('.meta').innerHTML=meta||'';
  if(raw!==undefined&&raw!==text){
    const det=document.createElement('details');
    det.innerHTML='<summary>原始输出（未经后处理）</summary><pre></pre>';
    det.querySelector('pre').textContent=raw; d.querySelector('.body').appendChild(det);
  }
  log.appendChild(d); window.scrollTo(0,document.body.scrollHeight); return d;
}
async function go(){
  const t=box.value.trim(); if(!t) return;
  add('user',t); box.value=''; box.style.height='auto';
  send.disabled=true; const thinking=add('ai','…','');
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:t,mode:mode.value})});
    const j=await r.json();
    thinking.remove();
    if(j.error){add('ai','错误：'+j.error,'');}
    else{add('ai',j.reply,'<code>'+j.mode+'</code> · '+j.gen_chars+' 字符 · '+
      j.seconds.toFixed(2)+'s · temp '+j.temperature,j.raw);}
  }catch(e){thinking.remove();add('ai','请求失败：'+e,'');}
  send.disabled=false; box.focus();
}
send.onclick=go;
box.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();go();}});
box.addEventListener('input',()=>{box.style.height='auto';box.style.height=box.scrollHeight+'px';});
box.focus();
</script></body></html>
"""


#: 只显示这些检查点。**白名单，不是黑名单。**
#:
#: ## 为什么改成白名单
#:
#: 原来是全盘扫描 `runs/*/latest.pt`，结果列出了 23 个 —— 其中一半是
#: **实验中途废弃的**（`mix11` 停在 2%、`chatmix` 停在 75%、`corpus` 是
#: 600 步的旧基线）。用户在界面上看到一堆名字，**不知道哪个能用**。
#:
#: 白名单还有个更实际的好处：**它同时是一份"当前有哪些可用模型"的文档**。
#: 加新模型时必须在这里登记 —— 这比"扫出来什么算什么"更不容易出错。
#:
#: ## 排序即优先级
#:
#: 列表按顺序排，**第一个是综合最强的**（78M 融合版：代码 + 对话）。
#:
#: ## ⚠️ 训练中的模型不要放进来
#:
#: `latest.pt` 每 500 步被**覆盖写一次**，加载时若撞上写入会读到半写状态的文件
#: —— 可能张量不匹配直接报错，也可能**静默加载出错乱的权重**。
#: 所以 `runs/stage2/latest.pt`（训练中）不登记；要测就用它的 `best.pt`，
#: 或者等训练结束。
KEEP: list[tuple[str, str, str]] = [
    ("runs/_ref78mix/latest.pt", "⭐ 78M 融合版", "代码+对话（最强）"),
    ("runs/_ref78/latest.pt", "78M 纯代码", "代码 edit_sim +7.23pp"),
    ("runs/stage1/latest.pt", "20M 阶段1", "纯代码，1 epoch"),
    ("runs/big20mix/latest.pt", "20M 混合(旧)", "混合训练中途版"),
    ("runs/_verify/ckpt-step13428.pt", "11M 代码", "代码 edit_sim +7.42pp"),
    ("runs/chat2/latest.pt", "11M 聊天", "会聊天，代码能力已废"),
]


def list_checkpoints() -> list:
    """扫描可切换的检查点（**白名单**，见 `KEEP`）。

    **为什么必须能切**：判断"训练有没有用"要能在**同一页面、同一问题**下
    对比训练前后两个检查点，而不是开两次服务、凭记忆比 —— 那正是本项目
    反复踩过的坑（"感觉差不多"）。
    """
    out, seen = [], set()
    for rel, label, note in KEEP:
        p = ROOT / rel
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        st = p.stat()
        out.append({"path": rel, "kind": label, "note": note,
                    "mb": round(st.st_size / 1e6, 1),
                    "mtime": time.strftime("%m-%d %H:%M",
                                           time.localtime(st.st_mtime))})
    return out


def load_model(ckpt: str, tokenizer: str, threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(threads)
        from codemind.generate import load_runner
        t0 = time.perf_counter()
        gen = load_runner(Path(ROOT / ckpt), device="cpu",
                          tokenizer_path=Path(ROOT / tokenizer))
        _state.update(gen=gen, tok=gen.tokenizer, ckpt=ckpt,
                      name=f"{Path(ckpt).name} · {sum(p.numel() for p in gen.model.parameters())/1e6:.1f}M",
                      load_s=time.perf_counter() - t0, err="")
    except Exception:
        _state["err"] = traceback.format_exc()[-400:]


def infer(message: str, mode: str, temperature: float, top_k: int,
          rep: float, max_new: int, history=None,
          use_history: bool = False) -> dict:
    from codemind.model import SamplingParams
    gen, tok = _state["gen"], _state["tok"]
    if gen is None:
        raise RuntimeError("模型未就绪：" + _state["err"])
    params = SamplingParams(temperature=temperature, top_k=top_k,
                            repetition_penalty=rep, seed=None)
    t0 = time.perf_counter()
    with _lock:
        if mode == "infill":
            head, _, tail = message.partition("<|hole|>")
            raw = gen.infill(head, tail, max_new_tokens=max_new, params=params)
            hist = None
        elif mode == "complete":
            raw = gen.complete(message, max_new_tokens=max_new, params=params)
            hist = None
        else:
            # ⚠️ `chat()` 返回的是 **(reply, history) 元组**，不是字符串。
            # 直接把它当字符串用会在 JSON 序列化时炸掉——实测踩过。
            #
            # 🔴 **默认不带历史**（单轮），这不是保守，是被实测逼出来的：
            # 模型是用**单轮**数据训练的（`[用户] 问题 [助手] 回答`），
            # 从没训过多轮。喂历史 = 分布外输入，输出会崩：
            #
            #   干净历史：「作为一个人工智能助手，我无法为您提供关于这个信息的信息。」
            #   带历史：  「你可以将其分类为可能的数据，可以将其分类为可能的数据…」
            #
            # 多轮要等有**多轮对话语料**并训练过之后才开。
            # 想手动试可以传 `use_history: true`。
            hist_in = history if use_history else None
            raw, hist = gen.chat(message, history=hist_in, max_new_tokens=max_new,
                                 params=params)
    return {"raw": raw, "seconds": time.perf_counter() - t0, "history": hist}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):        # 安静一点，别把控制台刷满
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:                                   # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            body = json.dumps({"ready": _state["gen"] is not None,
                               "name": _state["name"], "load_s": _state["load_s"],
                               "ckpt": _state.get("ckpt", ""),
                               "checkpoints": list_checkpoints(),
                               "err": _state["err"][-200:]}).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:                                  # noqa: N802
        # 切换检查点 —— 让"训练前后"能在同一页面直接对比
        if self.path == "/api/load":
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                path = str(req.get("path") or "")
                allowed = {c["path"] for c in list_checkpoints()}
                if path not in allowed:
                    out = {"error": f"不在可切换列表里：{path}"}
                else:
                    _history.clear()                      # 换了模型，历史清零
                    load_model(path, self.server.tokenizer,   # type: ignore[attr-defined]
                               self.server.threads)           # type: ignore[attr-defined]
                    out = {"ok": _state["gen"] is not None, "ckpt": path,
                           "name": _state["name"], "load_s": _state["load_s"],
                           "err": _state["err"][-200:]}
            except Exception as exc:                    # noqa: BLE001
                out = {"error": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
            return
        if self.path != "/api/chat":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            message = str(req.get("message") or "")
            mode = str(req.get("mode") or "chat")
            if mode == "chat" and req.get("reset"):
                _history.clear()
            r = infer(message, mode,
                      # ⚠️ 默认 **温度 0（greedy）**，不是 0.8。
                      # 小模型的输出分布很平，温度 >0 等于在平分布上随机抽样 ——
                      # 实测同一个模型同一个问题，0.7 给乱码，0.0 给
                      # 「作为一个人工智能助手，我无法为您提供关于这个信息的信息」。
                      # 详见 codemind/generate.py 的 CHAT_DEFAULT_PARAMS。
                      float(req.get("temperature", 0.0)),
                      int(req.get("top_k", 1)),
                      float(req.get("rep", 1.1)),
                      int(req.get("max_new_tokens", 200)),
                      history=_history or None,
                      use_history=bool(req.get("use_history", False)))
            if r.get("history") is not None:
                # ⚠️ 必须**原地更新**，不能写 `_history = r["history"]`。
                # 只要函数里存在对 `_history` 的赋值，Python 就把它判成**局部变量**，
                # 而同函数内更早的 `_history.clear()` 立刻 UnboundLocalError。
                # 实测踩过：修完 global 之后，整个 /api/chat 与 /api/load 全部 500。
                _history.clear()
                _history.extend(r["history"])
            out = {"reply": r["raw"], "raw": r["raw"], "mode": mode,
                   "gen_chars": len(r["raw"]), "seconds": r["seconds"],
                   "temperature": req.get("temperature", 0.0)}
        except Exception as exc:                                # noqa: BLE001
            out = {"error": f"{type(exc).__name__}: {exc}"}
        self._send(200, json.dumps(out, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--run", default="runs/_verify/ckpt-step13428.pt")
    ap.add_argument("--tokenizer", default="runs/corpus2/tokenizer.json")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    print(f"[web] 加载模型 {args.run} …", flush=True)
    load_model(args.run, args.tokenizer, args.threads)
    if _state["gen"] is None:
        print("[web] 加载失败:\n" + _state["err"], flush=True)
        return 1
    print(f"[web] 就绪：{_state['name']}（{_state['load_s']:.1f}s）", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.tokenizer = args.tokenizer       # type: ignore[attr-defined]
    srv.threads = args.threads           # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] 打开 {url}   （Ctrl+C 停止）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 停止", flush=True)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
