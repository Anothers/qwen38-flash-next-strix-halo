#!/usr/bin/env python3
"""Inline kb.json into app/template.html -> one self-contained index.html.
Usage: build_site.py KB.json OUT.html [--fragment]
Default output is a complete standalone document; --fragment omits <html>/<head> (for hosts that add their own skeleton)."""
import json, sys, pathlib
kb, out = sys.argv[1:3]
fragment = "--fragment" in sys.argv
data = json.dumps(json.load(open(kb, encoding="utf-8")), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (pathlib.Path(__file__).parent / "app" / "template.html").read_text(encoding="utf-8").replace("__KB_JSON__", data)
if not fragment:
    html = ('<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light}body{margin:0}</style></head><body>\n' + html + '\n</body></html>')
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out).write_text(html, encoding="utf-8")
print(out, round(len(html) / 1e6, 2), "MB")
