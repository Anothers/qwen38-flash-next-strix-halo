"""Layout-aware PDF extraction for retrieval.

Each page becomes retrieval units:
  - one unit per table row, written as "header: value; header: value" so a row is searchable on its own;
  - prose split at sentence ends (code blocks at line ends) and packed into units of at most `max_chars`.
Every unit keeps the page title and the nearest sub-heading (used as embedding context) and a larger
`context` passage for display: the table the row belongs to, or the paragraph the piece came from.

Layout rules come from the reference report (running header/footer above y=40 / below y=800, bold
headings of 10pt and up, table text smaller than body text). They are parameters, not hard limits.
"""
import re
import pymupdf

SENT_END = re.compile(r"(?<=[。；！？!?;])")
PARA_GAP = 5          # pt between lines that starts a new paragraph (wrapped lines sit ~2.5pt apart)
ROW_GAP = 6           # pt between table rows (wrapped lines inside a cell sit ~1.3pt apart)
TABLE_END_GAP = 24


def _lines(page, margin_top=40, margin_bottom=800):
    out = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            if not t or l["bbox"][1] < margin_top or l["bbox"][1] > margin_bottom:
                continue
            s = max(l["spans"], key=lambda s: len(s["text"]))
            out.append(dict(bbox=l["bbox"], text=t, size=round(s["size"], 1),
                            bold="Bold" in s["font"], mono="Mono" in s["font"] or "Courier" in s["font"]))
    out.sort(key=lambda x: (round(x["bbox"][1], 1), x["bbox"][0]))
    return out


def _join(parts):
    """Join wrapped lines: no space between CJK characters, one space between Latin words."""
    s = ""
    for p in parts:
        if s and re.match(r"[A-Za-z0-9\[(]", p[:1]) and re.search(r"[A-Za-z0-9,.)\]]$", s):
            s += " "
        s += p
    return re.sub(r"\s{2,}", " ", s)


def _tables(page, lines):
    """Rows of the page's tables, rebuilt from header cell columns and vertical gaps."""
    found, used = [], set()
    for t in page.find_tables().tables:
        hdr = [c for c in t.header.cells if c]
        names = [(n or "").replace("\n", "") for n in t.header.names]
        if len(hdr) < 2:
            continue
        xs = [c[0] for c in hdr] + [hdr[-1][2]]
        htop, hbot = min(c[1] for c in hdr), max(c[3] for c in hdr)

        def col(x0):
            for i in range(len(hdr)):
                if x0 < xs[i + 1] - 4:
                    return i
            return len(hdr) - 1

        rows, cur, bottom, size = [], None, None, None
        for k, ln in enumerate(lines):
            x0, y0, x1, y1 = ln["bbox"]
            if k in used or y0 < htop - 1:
                continue
            if y0 < hbot - 1:
                used.add(k)                              # header text itself
                continue
            if bottom is not None and y0 - bottom > TABLE_END_GAP:
                break
            if ln["bold"] or (size is not None and ln["size"] > size + 0.5):
                break                                    # a heading or body text after the table
            c = col(x0)
            if c == 0 and x1 - x0 > (xs[1] - xs[0]) + 20:
                break                                    # full-width paragraph
            if cur is None or y0 - bottom > ROW_GAP:
                cur = [[] for _ in hdr]
                rows.append(cur)
            cur[c].append(ln["text"])
            used.add(k)
            size = ln["size"] if size is None else max(size, ln["size"])
            bottom = y1 if bottom is None else max(bottom, y1)
        rows = [[_join(c) for c in r] for r in rows if any(r)]
        if rows:
            found.append(dict(names=names, rows=rows, y=htop))
    return found, used


def _pack(segments, max_chars):
    out, buf = [], ""
    for s in segments:
        if buf and len(buf) + len(s) > max_chars:
            out.append(buf)
            buf = ""
        buf += s
    if buf:
        out.append(buf)
    return out


def extract(pdf_path, skip_pages=(), max_chars=200):
    units = []
    doc = pymupdf.open(pdf_path)
    for pno, page in enumerate(doc, 1):
        if pno in skip_pages:
            continue
        lines = _lines(page)
        if not lines:
            continue
        section = title = ""
        if lines[0]["bold"] and lines[0]["size"] < 9:
            section = lines.pop(0)["text"]
        if lines and lines[0]["size"] >= 16:
            title = lines.pop(0)["text"]
        tables, used = _tables(page, lines)

        # prose: paragraphs from vertical gaps and indentation; bold lines of 10pt+ are sub-headings
        items = [("table", t["y"], t) for t in tables]
        para, last = None, None
        for k, ln in enumerate(lines):
            if k in used:
                continue
            x0, y0, _, y1 = ln["bbox"]
            if ln["bold"] and ln["size"] >= 10:
                items.append(("heading", y0, ln["text"]))
                para = None
                continue
            if para is None or y0 - last["bbox"][3] > PARA_GAP or abs(x0 - para[0]["bbox"][0]) > 10 or ln["mono"] != para[0]["mono"]:
                para = []
                items.append(("para", y0, para))
            para.append(ln)
            last = ln
        items.sort(key=lambda x: x[1])

        heading = ""
        for kind, _, it in items:
            if kind == "heading":
                heading = it
            elif kind == "table":
                ctx = " | ".join(it["names"]) + "\n" + "\n".join(" | ".join(r) for r in it["rows"])
                for r in it["rows"]:
                    text = "；".join(f"{n}：{v.rstrip('。；')}" for n, v in zip(it["names"], r) if v) + "。"
                    units.append(dict(page=pno, section=section, title=title, heading=heading, kind="row",
                                      text=text, context=ctx))
            else:
                mono = it[0]["mono"]
                text = "\n".join(l["text"] for l in it) if mono else _join([l["text"] for l in it])
                if len(text) < 8:
                    continue
                segs = [l + "\n" for l in text.split("\n")] if mono else SENT_END.split(text)
                for piece in _pack([s for s in segs if s], max_chars):
                    piece = piece.strip()
                    if len(piece) >= 8:
                        units.append(dict(page=pno, section=section, title=title, heading=heading,
                                          kind="code" if mono else "para", text=piece, context=text))
    for i, u in enumerate(units):
        u["id"] = i
    return units


def embed_text(u):
    """String sent to the document encoder: official 'title: … | text: …' form with page title and sub-heading."""
    t = u["title"] + (" / " + u["heading"] if u["heading"] else "")
    return f"title: {t} | text: {u['text']}"
