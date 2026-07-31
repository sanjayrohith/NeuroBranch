"""Storyboard and painter for the NeuroBranch hero animation.

The whole story happens inside the app window. The only chrome is the window's
own title bar; there is no outer header, logo lockup or repo URL in any frame.

Camera: the canvas fits a focus rectangle expressed in graph columns, which is
animated per beat. That mirrors how the app's topology-aware layout zooms to
fit, and keeps card labels legible when the code panel is open.
"""

import math

from PIL import Image, ImageDraw

from draw import (Offset, Shape, along, bezier_pts, blend, clamp, ease_in_out,
                  ease_out, ease_out_back, edge_path, fit, font, partial,
                  pulse, seg, text)

# ---------------------------------------------------------------- layout

CARD_W, CARD_H = 1000, 620
TITLEBAR_H = 32
TOPBAR_H = 46
STATUS_H = 28
MAIN_Y0 = TITLEBAR_H + TOPBAR_H          # 78
MAIN_Y1 = CARD_H - STATUS_H              # 592
LIB_W = 158
PANEL_W = 214
SPLIT_FRAC = 0.42                        # code panel share of the main area

NODE_W, NODE_H = 116, 38
COL_PITCH = 136
COL_X0 = 26
ROW_Y = {0: -58, 1: 0, 2: 58}
GRAPH_CY = 316
CAM_PAD = 18
CAPTION_RESERVE = 56

# ---------------------------------------------------------------- timeline

B1 = (0.00, 2.40)    # hook / empty starter
B2 = (2.40, 7.60)    # compose + typed-port rejection
B3 = (7.60, 12.90)   # Ask NeuroBranch plans the graph
B4 = (12.90, 17.00)  # two-way PyTorch sync (split view)
B5 = (17.00, 19.40)  # MLP -> routed MoE, code updates with the graph
B6 = (19.40, 22.60)  # atomic player run + settle
DURATION = B6[1]

CAPTIONS = [
    (0.35, 2.28, "A model is a graph of small, typed, executable cards"),
    (2.75, 7.45, "Drag typed cards — incompatible ports refuse to connect"),
    (7.90, 12.75, "Describe it in English; the planner proposes, you approve"),
    (13.25, 16.90, "The graph compiles to real PyTorch, and parses back"),
    (17.25, 19.30, "Change the topology: MLP becomes a token-routed MoE"),
    (19.85, 22.45, "Run it on a local Python runtime, tensor by tensor"),
]

# id, label, short, sublabel, col, row, role
NODES = [
    ("embed", "Token Embed", "Embed", "50257 x 512", 0, 1, "hidden"),
    ("ln1",   "LayerNorm",   "Norm",  "d_model 512", 1, 1, "hidden"),
    ("q",     "Q Proj",      "Q",     "Linear 512",  2, 0, "query"),
    ("k",     "K Proj",      "K",     "Linear 512",  2, 1, "key"),
    ("v",     "V Proj",      "V",     "Linear 512",  2, 2, "value"),
    ("attn",  "Attention",   "Attn",  "8 heads",     3, 1, "hidden"),
    ("mlp",   "MLP",         "MLP",   "GELU 4x",     4, 1, "hidden"),
    ("head",  "Output Head", "Head",  "vocab 50257", 5, 1, "hidden"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [
    ("embed", "ln1", "hidden"),
    ("ln1", "q", "query"), ("ln1", "k", "key"), ("ln1", "v", "value"),
    ("q", "attn", "query"), ("k", "attn", "key"), ("v", "attn", "value"),
    ("attn", "mlp", "hidden"),
    ("mlp", "head", "hidden"),
]

COMPOSE_ORDER = ["embed", "ln1", "q", "k", "v", "attn"]

# id, label, short, sublabel, col, row
MOE = [
    ("exA", "Expert A", "E1", "top-2", 4, 0),
    ("exB", "Expert B", "E2", "top-2", 4, 1),
    ("exS", "Shared",   "S",  "always", 4, 2),
]
MOE_BY_ID = {m[0]: m for m in MOE}

LIBRARY = [
    ("EMBEDDING", ["Token Embed", "Positional", "Patch Embed"]),
    ("NORMALISE", ["LayerNorm", "RMSNorm"]),
    ("ATTENTION", ["Q Proj", "K Proj", "V Proj", "MultiHead"]),
    ("FEEDFORWARD", ["MLP", "GELU", "Router", "Expert"]),
]

_HEAD = [
    [("class ", "kw"), ("GPTQABlock", "cls"), ("(nn.Module):", "p")],
    [("    def ", "kw"), ("__init__", "fn"), ("(self, d_model=", "p"), ("512", "num"), ("):", "p")],
    [("        super().__init__()", "p")],
    [("        self.embed  = nn.Embedding(", "p"), ("50257", "num"), (", d_model)", "p")],
    [("        self.ln_1   = nn.LayerNorm(d_model)", "p")],
    [("        self.q_proj = nn.Linear(d_model, d_model)", "p")],
    [("        self.k_proj = nn.Linear(d_model, d_model)", "p")],
    [("        self.v_proj = nn.Linear(d_model, d_model)", "p")],
    [("        self.attn   = nn.MultiheadAttention(d_model, ", "p"), ("8", "num"), (")", "p")],
]
_MLP = [
    [("        self.mlp    = nn.Sequential(", "p")],
    [("            nn.Linear(d_model, ", "p"), ("4", "num"), (" * d_model),", "p")],
    [("            nn.GELU(),", "p")],
    [("            nn.Linear(", "p"), ("4", "num"), (" * d_model, d_model),", "p")],
    [("        )", "p")],
]
_MOE = [
    [("        self.router  = nn.Linear(d_model, ", "p"), ("3", "num"), (")", "p")],
    [("        self.experts = nn.ModuleList([", "p")],
    [("            FeedForward(d_model) ", "p"), ("for", "kw"), (" _ ", "p"), ("in", "kw"), (" range(", "p"), ("2", "num"), (")", "p")],
    [("        ])", "p")],
    [("        self.shared  = FeedForward(d_model)", "p")],
]
_TAIL_MLP = [
    [("        self.head   = nn.Linear(d_model, ", "p"), ("50257", "num"), (")", "p")],
    [("", "p")],
    [("    def ", "kw"), ("forward", "fn"), ("(self, ids):", "p")],
    [("        x = self.ln_1(self.embed(ids))", "p")],
    [("        q, k, v = self.q_proj(x), self.k_proj(x), \\", "p")],
    [("                  self.v_proj(x)", "p")],
    [("        x = x + self.attn(q, k, v)[", "p"), ("0", "num"), ("]", "p")],
    [("        x = x + self.mlp(x)", "p")],
    [("        return ", "kw"), ("self.head(x)", "p")],
]
_TAIL_MOE = _TAIL_MLP[:-2] + [
    [("        x = x + self.moe(x, self.router(x))", "p")],
    [("        return ", "kw"), ("self.head(x)", "p")],
]

CODE = _HEAD + _MLP + _TAIL_MLP
CODE_MOE = _HEAD + _MOE + _TAIL_MOE
MOE_DIFF_LINES = (len(_HEAD), len(_HEAD) + len(_MOE))          # highlighted band
MOE_DIFF_TAIL = len(_HEAD) + len(_MOE) + len(_TAIL_MOE) - 2

PLAN = [
    ("search_catalog", "9 matches"),
    ("add_card x 6", "typed"),
    ("connect_ports x 9", "compatible"),
    ("layout_parallel", "3 lanes"),
    ("validate", "passed"),
]

PROMPT = "Build a compact GPT-like QA model"

RUN_ORDER = ["embed", "ln1", "k", "attn", "exB", "head"]
RUN_EDGES = [("embed", "ln1"), ("ln1", "k"), ("k", "attn"),
             ("attn", "exB"), ("exB", "head")]
SHAPES = {"embed": "[8,128,512]", "attn": "[8,128,512]", "head": "[8,128,50257]"}


def role_colour(th, role):
    return {
        "query": th.p_query, "key": th.p_key, "value": th.p_value,
        "hidden": th.p_hidden, "image": th.p_image,
    }[role]


# ---------------------------------------------------------------- state

def build_state(t):
    s = {
        "panel": 0.0, "split": 0.0, "nodes": {}, "edges": {},
        "code_lines": 0.0, "reject": None, "plan": 0.0, "prompt_chars": 0,
        "agent_thinking": 0.0, "moe": 0.0, "run": None, "empty": 0.0,
        "view_tab": 0, "lib_highlight": None, "t": t, "sync_pulse": 0.0,
        "applied": 0.0, "cam": (0.0, 5.0), "cam_top": 0.0, "fade": 1.0,
        "code_moe": 0.0,
    }

    if t < B2[0]:
        s["empty"] = 1.0 - seg(t, B1[1] - 0.42, B1[1])
        s["cam"] = (0.0, 3.0)
        return s

    if t < B3[0]:
        u = t - B2[0]
        s["cam"] = (0.0, 3.0)
        for i, nid in enumerate(COMPOSE_ORDER):
            a = 0.10 + i * 0.36
            s["nodes"][nid] = ease_out_back(seg(u, a, a + 0.42))
            if 0.0 <= u - a < 0.30:
                s["lib_highlight"] = nid
        for a, b, at in [("embed", "ln1", 2.30), ("ln1", "q", 2.55),
                         ("ln1", "k", 2.70), ("ln1", "v", 2.85),
                         ("k", "attn", 3.90), ("v", "attn", 4.05)]:
            s["edges"][(a, b)] = ease_out(seg(u, at, at + 0.40))
        # the refusal: Q Proj's query output dragged onto K Proj's hidden input
        rj = seg(u, 3.10, 4.05)
        if 0.0 < rj < 1.0:
            s["reject"] = rj
        s["edges"][("q", "attn")] = ease_out(seg(u, 4.25, 4.70))
        return s

    for nid in COMPOSE_ORDER:
        s["nodes"][nid] = 1.0
    for a, b, _ in EDGES:
        s["edges"][(a, b)] = 1.0

    if t < B4[0]:
        u = t - B3[0]
        s["panel"] = ease_in_out(seg(u, 0.05, 0.55))
        c = ease_in_out(seg(u, 3.95, 4.95))
        s["cam"] = (0.0, 3.0 + 2.0 * c)
        s["prompt_chars"] = round(len(PROMPT) * seg(u, 0.60, 2.05))
        s["agent_thinking"] = 1.0 if 2.10 <= u < 2.80 else 0.0
        s["plan"] = seg(u, 2.80, 4.05)
        s["applied"] = seg(u, 4.15, 4.50)
        s["nodes"]["mlp"] = ease_out_back(seg(u, 4.25, 4.70))
        s["nodes"]["head"] = ease_out_back(seg(u, 4.45, 4.90))
        s["edges"][("attn", "mlp")] = ease_out(seg(u, 4.60, 4.95))
        s["edges"][("mlp", "head")] = ease_out(seg(u, 4.75, 5.10))
        return s

    s["nodes"]["mlp"] = 1.0
    s["nodes"]["head"] = 1.0
    s["plan"] = 1.0
    s["applied"] = 1.0
    s["prompt_chars"] = len(PROMPT)

    if t < B5[0]:
        u = t - B4[0]
        s["panel"] = 1.0 - ease_in_out(seg(u, 0.0, 0.40))
        s["split"] = ease_in_out(seg(u, 0.30, 0.90))
        s["view_tab"] = 2
        # camera slides onto the attention -> head half, which is what the
        # visible code describes
        s["cam"] = (2.16 * ease_in_out(seg(u, 0.35, 1.10)), 5.0)
        s["code_lines"] = seg(u, 1.00, 3.30) * len(CODE)
        s["sync_pulse"] = seg(u, 3.25, 4.05)
        return s

    s["split"] = 1.0
    s["view_tab"] = 2
    s["code_lines"] = len(CODE)

    if t < B6[0]:
        u = t - B5[0]
        s["moe"] = ease_in_out(seg(u, 0.20, 1.05))
        # hold the camera on a column boundary so no card is sliced in half
        s["cam"] = (2.16, 5.0)
        s["cam_top"] = 30.0 * s["moe"]
        s["code_moe"] = seg(u, 0.55, 0.95)
        s["sync_pulse"] = seg(u, 0.95, 1.75)
        return s

    s["moe"] = 1.0
    s["code_moe"] = 1.0

    # ---- B6: pull back to the whole graph in Blocks view and execute
    u = t - B6[0]
    s["split"] = 1.0 - ease_in_out(seg(u, 0.00, 0.45))
    s["view_tab"] = 0 if u > 0.22 else 2
    s["cam"] = (2.16 * (1 - ease_in_out(seg(u, 0.05, 0.60))), 5.0)
    s["cam_top"] = 30.0
    s["run"] = seg(u, 0.55, 2.55)
    s["sync_pulse"] = 0.0
    # dissolve at the very end so the loop point is clean
    s["fade"] = 1.0 - seg(u, 2.95, 3.20)
    return s


# ---------------------------------------------------------------- camera

def canvas_box(s):
    x0 = LIB_W
    x1 = CARD_W - PANEL_W * s["panel"]
    if s["split"] > 0:
        x1 = x0 + (x1 - x0) * (1.0 - SPLIT_FRAC * s["split"])
    return x0, MAIN_Y0, x1, MAIN_Y1


def transform(s):
    """Fit the focus rectangle (in graph columns) into the canvas."""
    cx0, _, cx1, _ = canvas_box(s)
    lo, hi = s["cam"]
    wx0 = COL_X0 + lo * COL_PITCH - 14
    wx1 = COL_X0 + hi * COL_PITCH + NODE_W + 14
    wy0 = GRAPH_CY - 58 - NODE_H / 2 - 16 - s["cam_top"]
    wy1 = GRAPH_CY + 58 + NODE_H / 2 + 16

    avail_w = (cx1 - cx0) - CAM_PAD * 2
    avail_h = (MAIN_Y1 - MAIN_Y0 - CAPTION_RESERVE) - CAM_PAD
    sc = min(1.0, avail_w / (wx1 - wx0), avail_h / (wy1 - wy0))
    tx = cx0 + CAM_PAD + (avail_w - (wx1 - wx0) * sc) / 2 - wx0 * sc
    ty = MAIN_Y0 + CAM_PAD / 2 + (avail_h - (wy1 - wy0) * sc) / 2 - wy0 * sc
    return sc, tx, ty


def node_rect(col, row, sc, tx, ty):
    w, h = NODE_W * sc, NODE_H * sc
    x = tx + (COL_X0 + col * COL_PITCH) * sc
    cy = ty + (GRAPH_CY + ROW_Y[row]) * sc
    return x, cy - h / 2, w, h


def node_pos(nid, sc, tx, ty):
    if nid in NODE_BY_ID:
        n = NODE_BY_ID[nid]
        return node_rect(n[4], n[5], sc, tx, ty)
    m = MOE_BY_ID[nid]
    return node_rect(m[4], m[5], sc, tx, ty)


# ---------------------------------------------------------------- painter

def paint(t, th):
    s = build_state(t)
    sh = Shape(CARD_W, CARD_H, th.bg)
    sc, tx, ty = transform(s)
    cx0, _, cx1, _ = canvas_box(s)
    moe_on = s["moe"] > 0.5

    # The graph lives in its own layer clipped to the canvas rect, so cards
    # never bleed into the block library or the code panel when the camera
    # pans or the panels animate.
    gx0, gx1 = int(round(cx0)), int(round(cx1))
    gl = Shape(max(1, gx1 - gx0), MAIN_Y1 - MAIN_Y0, th.panel, origin=(gx0, MAIN_Y0))

    # ---------- window chrome
    sh.rect(0, 0, CARD_W, TITLEBAR_H, fill=th.titlebar)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        sh.dot(20 + i * 18, TITLEBAR_H / 2, 5.5, blend(th.titlebar, c, 0.82))
    sh.line([(0, TITLEBAR_H), (CARD_W, TITLEBAR_H)], th.line_soft, 1)

    sh.rect(0, TITLEBAR_H, CARD_W, TOPBAR_H, fill=th.bg)
    sh.line([(0, MAIN_Y0), (CARD_W, MAIN_Y0)], th.line, 1)
    sw_x, sw_y, sw_w, sw_h = 316, TITLEBAR_H + 10, 236, 26
    sh.rrect(sw_x, sw_y, sw_w, sw_h, 7, fill=th.bg_deep, outline=th.line, width=1)
    tab_w = sw_w / 3
    sh.rrect(sw_x + 2 + s["view_tab"] * tab_w, sw_y + 2, tab_w - 4, sw_h - 4, 5,
             fill=blend(th.panel_raised, th.violet, 0.10),
             outline=blend(th.panel_raised, th.violet, 0.26), width=1)
    ag_on = s["panel"] > 0.02
    sh.rrect(CARD_W - 190, TITLEBAR_H + 10, 168, 26, 7,
             fill=blend(th.panel_raised, th.violet, 0.16 if ag_on else 0.08),
             outline=blend(th.panel_raised, th.violet, 0.44 if ag_on else 0.24), width=1)

    # ---------- block library
    sh.rect(0, MAIN_Y0, LIB_W, MAIN_Y1 - MAIN_Y0, fill=th.panel)
    sh.line([(LIB_W, MAIN_Y0), (LIB_W, MAIN_Y1)], th.line, 1)
    sh.rrect(12, MAIN_Y0 + 14, LIB_W - 24, 24, 6, fill=th.bg_deep,
             outline=th.line, width=1)

    # ---------- dot grid (inside the clipped graph layer)
    gs = 26
    gxx = int(cx0) + (gs - int(cx0) % gs)
    while gxx < cx1 - 2:
        gy = MAIN_Y0 + gs
        while gy < MAIN_Y1 - 2:
            gl.dot(gxx, gy, 0.9, th.grid_dot)
            gy += gs
        gxx += gs

    # ---------- code panel shell
    if s["split"] > 0.02:
        px0, px1 = cx1, CARD_W - PANEL_W * s["panel"]
        sh.rect(px0, MAIN_Y0, px1 - px0, MAIN_Y1 - MAIN_Y0, fill=th.bg_deep)
        sh.line([(px0, MAIN_Y0), (px0, MAIN_Y1)], th.line, 1)
        sh.rect(px0, MAIN_Y0, px1 - px0, 30, fill=th.panel)
        sh.line([(px0, MAIN_Y0 + 30), (px1, MAIN_Y0 + 30)], th.line_soft, 1)
        if s["sync_pulse"] > 0:
            g = math.sin(math.pi * clamp(s["sync_pulse"])) * th.glow_strength
            sh.rrect(px1 - 96, MAIN_Y0 + 6, 84, 18, 5,
                     fill=blend(th.panel, th.green, 0.20 * g),
                     outline=blend(th.panel, th.green, 0.55 * g), width=1)
        # highlight band over the lines the MoE swap rewrote
        if 0.02 < s["code_moe"] < 1.0 or (moe_on and s["sync_pulse"] > 0):
            a0, a1 = MOE_DIFF_LINES
            g = max(math.sin(math.pi * clamp(s["code_moe"])),
                    math.sin(math.pi * clamp(s["sync_pulse"])))
            sh.rect(px0 + 4, MAIN_Y0 + 36 + a0 * 14.6,
                    px1 - px0 - 8, (a1 - a0) * 14.6,
                    fill=blend(th.bg_deep, th.green, 0.10 * g))

    # ---------- edges
    def port_out(nid):
        x, y, w, h = node_pos(nid, sc, tx, ty)
        return x + w, y + h / 2

    def port_in(nid):
        x, y, w, h = node_pos(nid, sc, tx, ty)
        return x, y + h / 2

    live = [e for e in EDGES if not (moe_on and (e[0] == "mlp" or e[1] == "mlp"))]
    for a, b, role in live:
        p = s["edges"].get((a, b), 0.0)
        if p <= 0.01 or s["nodes"].get(a, 0) < 0.3 or s["nodes"].get(b, 0) < 0.3:
            continue
        pts = edge_path(*port_out(a), *port_in(b))
        d = partial(pts, p)
        if len(d) > 1:
            gl.line(d, blend(th.panel, role_colour(th, role), 0.62), 1.7 * max(sc, 0.6))

    if s["moe"] > 0.02:
        ax, ay = port_out("attn")
        hx, hy = port_in("head")
        for i, (eid, _, _, _, ci, ri) in enumerate(MOE):
            ex, ey, ew, eh = node_rect(ci, ri, sc, tx, ty)
            active = i != 2
            rc = th.p_value if active else th.p_hidden
            al = 0.60 if active else 0.28
            for pts in (edge_path(ax, ay, ex, ey + eh / 2),
                        edge_path(ex + ew, ey + eh / 2, hx, hy)):
                d = partial(pts, s["moe"])
                if len(d) > 1:
                    gl.line(d, blend(th.panel, rc, al), 1.6 * max(sc, 0.6))

    # residual skip arc
    if s["nodes"].get("head", 0) > 0.5:
        sx, sy = port_out("ln1")
        ex, ey = port_in("head")
        top = ty + (GRAPH_CY - 104) * sc
        gl.line(bezier_pts((sx, sy), (sx + 90 * sc, top), (ex - 110 * sc, top), (ex, ey), 40),
                blend(th.panel, th.p_hidden, 0.28), 1.3 * max(sc, 0.6))

    # rejected connection
    if s["reject"] is not None:
        r = s["reject"]
        qx, qy = port_out("q")
        kx, ky = port_in("k")
        reach = ease_out(seg(r, 0.0, 0.52))
        recoil = ease_in_out(seg(r, 0.60, 1.0))
        f = reach * (1 - recoil)
        ex, ey = qx + (kx - qx) * f, qy + (ky - qy) * f
        bad = seg(r, 0.46, 0.58) * (1 - seg(r, 0.78, 1.0))
        col = blend(role_colour(th, "query"), th.danger, bad)
        gl.line(edge_path(qx, qy, ex, ey, 24), blend(th.panel, col, 0.78), 1.8 * max(sc, 0.6))
        if bad > 0.1:
            rr = (10 + 4 * (1 - bad)) * max(sc, 0.6)
            gl.circle(kx, ky, rr, outline=blend(th.panel, th.danger, bad), width=1.6)

    # ---------- nodes
    def draw_node(col_i, row_i, appear, role, status):
        x, y, w, h = node_rect(col_i, row_i, sc, tx, ty)
        y -= (1 - appear) * 16 * sc
        acc = role_colour(th, role)
        fill = blend(th.panel, th.surface, clamp(appear))
        edge = th.line
        if status == "active":
            edge = blend(th.line, th.green, 0.85)
        elif status == "done":
            edge = blend(th.line, th.green, 0.38)
        gl.rrect(x, y, w, h, 7 * max(sc, 0.5), fill=fill, outline=edge, width=1)
        gl.line([(x + 2, y + 1), (x + w - 2, y + 1)], blend(fill, th.text, 0.05), 1)
        gl.dot(x, y + h / 2, 3.0 * max(sc, 0.7), blend(th.panel, acc, 0.85))
        gl.dot(x + w, y + h / 2, 3.0 * max(sc, 0.7), blend(th.panel, acc, 0.85))
        if status == "active":
            gl.rrect(x - 2, y - 2, w + 4, h + 4, 8 * max(sc, 0.5),
                     outline=blend(th.panel, th.green, 0.45), width=1)
        return x, y, w, h, appear, status

    run_status = {}
    if s["run"] is not None:
        p = s["run"] * len(RUN_ORDER)
        for i, nid in enumerate(RUN_ORDER):
            if i < p - 1:
                run_status[nid] = "done"
            elif i <= p:
                run_status[nid] = "active"

    boxes = {}
    for nid, label, short, sub, ci, ri, role in NODES:
        if nid == "mlp" and moe_on:
            continue
        ap = s["nodes"].get(nid, 0.0)
        if nid == "mlp" and s["moe"] > 0.02:
            ap *= 1 - s["moe"]
        if ap <= 0.02:
            continue
        boxes[nid] = draw_node(ci, ri, ap, role, run_status.get(nid))

    if s["moe"] > 0.02:
        for i, (eid, label, short, sub, ci, ri) in enumerate(MOE):
            role = "value" if i != 2 else "hidden"
            boxes[eid] = draw_node(ci, ri, s["moe"], role, run_status.get(eid))

    # travelling tensor pulse
    if s["run"] is not None and 0 < s["run"] < 1:
        p = s["run"] * len(RUN_EDGES)
        i = min(int(p), len(RUN_EDGES) - 1)
        a, b = RUN_EDGES[i]
        if a in boxes and b in boxes:
            pts = edge_path(*port_out(a), *port_in(b))
            px, py = along(pts, p - i)
            gl.dot(px, py, 4.2 * max(sc, 0.6), blend(th.panel, th.green, 0.95))
            gl.circle(px, py, 7.5 * max(sc, 0.6),
                      outline=blend(th.panel, th.green, 0.35), width=1)

    # ---------- agent panel shell
    if s["panel"] > 0.02:
        px = CARD_W - PANEL_W * s["panel"]
        sh.rect(px, MAIN_Y0, PANEL_W, MAIN_Y1 - MAIN_Y0, fill=th.panel)
        sh.line([(px, MAIN_Y0), (px, MAIN_Y1)], th.line, 1)
        sh.rect(px, MAIN_Y0, PANEL_W, 30, fill=th.bg)
        sh.line([(px, MAIN_Y0 + 30), (CARD_W, MAIN_Y0 + 30)], th.line_soft, 1)
        sh.rrect(px + 12, MAIN_Y0 + 44, PANEL_W - 24, 46, 7, fill=th.bg_deep,
                 outline=blend(th.line, th.violet, 0.35), width=1)
        for i in range(len(PLAN)):
            rp = clamp(s["plan"] * len(PLAN) - i)
            if rp <= 0.02:
                continue
            ry = MAIN_Y0 + 108 + i * 30
            sh.rrect(px + 12, ry, PANEL_W - 24, 26, 6,
                     fill=blend(th.panel, th.panel_raised, rp),
                     outline=blend(th.panel, th.line, rp), width=1)
            if rp > 0.7:
                sh.circle(px + 25, ry + 13, 5.5,
                          outline=blend(th.panel_raised, th.green, 0.8), width=1.4)
        if s["applied"] > 0.02:
            ay = MAIN_Y0 + 108 + len(PLAN) * 30 + 12
            sh.rrect(px + 12, ay, PANEL_W - 24, 28, 7,
                     fill=blend(th.panel, th.green, 0.16 * s["applied"]),
                     outline=blend(th.panel, th.green, 0.5 * s["applied"]), width=1)

    # ---------- status bar
    sh.rect(0, MAIN_Y1, CARD_W, STATUS_H, fill=th.bg)
    sh.line([(0, MAIN_Y1), (CARD_W, MAIN_Y1)], th.line, 1)

    # ---------- caption pill
    cap = None
    for a, b, txt in CAPTIONS:
        if a <= t <= b:
            cap = (txt, min(seg(t, a, a + 0.26), 1 - seg(t, b - 0.26, b)))
    if cap:
        txt, fade = cap
        f = font("sb", 13)
        tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        wpx = tmp.textlength(txt, font=f) + 34
        bx = cx0 + ((cx1 - cx0) - wpx) / 2
        gl.rrect(bx, MAIN_Y1 - 46, wpx, 30, 8,
                 fill=blend(th.panel, th.bg_deep, 0.78 * fade),
                 outline=blend(th.panel, th.violet, 0.30 * fade), width=1)

    base = sh.resolve()
    d = ImageDraw.Draw(base)
    glimg = gl.resolve()
    gd = Offset(ImageDraw.Draw(glimg), gx0, MAIN_Y0)

    # ================================================== text pass
    text(d, (CARD_W / 2, TITLEBAR_H / 2 - 1), "NeuroBranch", font("sb", 12),
         th.faint, anchor="mm")
    text(d, (24, TITLEBAR_H + TOPBAR_H / 2), "GPT-like QA", font("sb", 13),
         th.text, anchor="lm")
    text(d, (128, TITLEBAR_H + TOPBAR_H / 2 + 1), "draft", font("m", 10),
         th.faint, anchor="lm")
    for i, name in enumerate(["Blocks", "PyTorch", "Split"]):
        text(d, (316 + (236 / 3) * (i + 0.5), TITLEBAR_H + TOPBAR_H / 2), name,
             font("sb", 11), th.text if i == s["view_tab"] else th.muted, anchor="mm")
    text(d, (CARD_W - 106, TITLEBAR_H + TOPBAR_H / 2), "Ask NeuroBranch",
         font("sb", 11), blend(th.text, th.violet, 0.5), anchor="mm")

    text(d, (22, MAIN_Y0 + 26), "Search 100+ cards", font("r", 10), th.faint, anchor="lm")
    ly = MAIN_Y0 + 54
    hot_label = None
    if s["lib_highlight"]:
        hot_label = NODE_BY_ID[s["lib_highlight"]][1]
    for fam, items in LIBRARY:
        text(d, (14, ly), fam, font("mb", 8), th.faint, anchor="lm")
        ly += 18
        for it in items:
            hot = it == hot_label
            if hot:
                d.rounded_rectangle([10, ly - 9, LIB_W - 10, ly + 9], radius=5,
                                    fill=blend(th.panel, th.green, 0.10),
                                    outline=blend(th.panel, th.green, 0.30))
            text(d, (18, ly), it, font("r", 11), th.text if hot else th.muted, anchor="lm")
            ly += 20
        ly += 6

    # node labels
    compact = sc < 0.74
    for nid, box in boxes.items():
        x, y, w, h, ap, status = box
        if nid in NODE_BY_ID:
            _, label, short, sub, _, _, role = NODE_BY_ID[nid]
        else:
            _, label, short, sub, _, _ = MOE_BY_ID[nid]
        fs = max(9, round(11 * sc))
        lab = label if not compact else label
        f = font("sb", fs)
        if gd.textlength(lab, font=f) > w - 14:
            lab = short
        col_t = blend(th.panel, th.text, clamp(ap))
        if status == "active":
            col_t = blend(th.text, th.green, 0.55)
        if compact:
            text(gd, (x + w / 2, y + h / 2), fit(gd, lab, f, w - 10), f, col_t, anchor="mm")
        else:
            text(gd, (x + w / 2, y + h / 2 - 6), fit(gd, lab, f, w - 12), f, col_t, anchor="mm")
            fsub = font("m", max(7, round(8.5 * sc)))
            text(gd, (x + w / 2, y + h / 2 + 7), fit(gd, sub, fsub, w - 12), fsub,
                 blend(th.panel, th.faint, clamp(ap)), anchor="mm")

    # router chip
    if s["moe"] > 0.3:
        rx = tx + (COL_X0 + 3.60 * COL_PITCH) * sc
        ry = ty + (GRAPH_CY - 96) * sc
        f = font("m", max(8, round(9 * sc)))
        lbl = "Router · top-2"
        wpx = gd.textlength(lbl, font=f) + 16
        gd.rounded_rectangle([rx - wpx / 2, ry - 10, rx + wpx / 2, ry + 10], radius=6,
                            fill=blend(th.panel, th.violet, 0.14),
                            outline=blend(th.panel, th.violet, 0.42))
        text(gd, (rx, ry), lbl, f, blend(th.text, th.violet, 0.30), anchor="mm")

    # one tensor-shape chip, on the node currently executing
    if s["run"] is not None and 0 < s["run"] < 1.0:
        act = [n for n, st in run_status.items() if st == "active" and n in SHAPES]
        if act:
            nid = act[0]
            x, y, w, h, _, _ = boxes[nid]
            f = font("m", max(8, round(9 * sc)))
            txt_ = SHAPES[nid]
            wpx = gd.textlength(txt_, font=f) + 12
            cxx, cyy = x + w / 2, y - 16
            gd.rounded_rectangle([cxx - wpx / 2, cyy - 9, cxx + wpx / 2, cyy + 9],
                                radius=5, fill=blend(th.panel, th.green, 0.13),
                                outline=blend(th.panel, th.green, 0.36))
            text(gd, (cxx, cyy), txt_, f, blend(th.text, th.green, 0.35), anchor="mm")

    # empty state
    if s["empty"] > 0.02:
        a = clamp(s["empty"])
        mx, my = (cx0 + cx1) / 2, (MAIN_Y0 + MAIN_Y1) / 2
        rise = (1 - ease_out(seg(t, 0.15, 0.95))) * 14
        text(gd, (mx, my - 16 + rise), "Blank starter", font("sb", 19),
             blend(th.panel, th.text, a * 0.92), anchor="mm")
        text(gd, (mx, my + 10 + rise),
             "Drag a typed card, or ask for an architecture in plain English",
             font("r", 12), blend(th.panel, th.muted, a * 0.9), anchor="mm")

    # agent panel text
    if s["panel"] > 0.35:
        px = CARD_W - PANEL_W * s["panel"]
        a = clamp((s["panel"] - 0.35) / 0.4)
        tc = lambda c, k=1.0: blend(th.panel, c, a * k)
        text(d, (px + 14, MAIN_Y0 + 15), "Ask NeuroBranch", font("sb", 11), tc(th.text), anchor="lm")
        text(d, (CARD_W - 16, MAIN_Y0 + 15), "Review", font("m", 9), tc(th.faint), anchor="rm")
        f = font("r", 11)
        shown = PROMPT[:s["prompt_chars"]]
        lines, cur = [], ""
        for wd in shown.split(" "):
            trial = (cur + " " + wd).strip()
            if d.textlength(trial, font=f) > PANEL_W - 44:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
        lines.append(cur)
        for i, ln in enumerate(lines[:2]):
            text(d, (px + 22, MAIN_Y0 + 60 + i * 15), ln, f, tc(th.text), anchor="lm")
        if s["prompt_chars"] < len(PROMPT) and int(t * 2.5) % 2 == 0:
            i = min(len(lines), 2) - 1
            lw = d.textlength(lines[i], font=f)
            d.line([(px + 23 + lw, MAIN_Y0 + 53 + i * 15),
                    (px + 23 + lw, MAIN_Y0 + 67 + i * 15)], fill=tc(th.violet), width=1)
        if s["agent_thinking"] > 0:
            for i in range(3):
                ph = pulse(t * 1.6 + i * 0.22, 1.0, 0.25, 1.0)
                d.ellipse([px + 20 + i * 12 - 3, MAIN_Y0 + 116 - 3,
                           px + 20 + i * 12 + 3, MAIN_Y0 + 116 + 3],
                          fill=blend(th.panel, th.violet, ph))
            text(d, (px + 62, MAIN_Y0 + 116), "planning…", font("m", 9), tc(th.faint), anchor="lm")
        for i, (name, note) in enumerate(PLAN):
            rp = clamp(s["plan"] * len(PLAN) - i)
            if rp <= 0.3:
                continue
            ry = MAIN_Y0 + 108 + i * 30 + 13
            text(d, (px + 38, ry), name, font("m", 9), blend(th.panel, th.text, rp * a), anchor="lm")
            text(d, (CARD_W - 18, ry), note, font("m", 8), blend(th.panel, th.faint, rp * a), anchor="rm")
            if rp > 0.7:
                d.line([(px + 22.5, ry), (px + 24.5, ry + 2.2), (px + 28, ry - 2.4)],
                       fill=blend(th.panel, th.green, a), width=1)
        if s["applied"] > 0.2:
            ay = MAIN_Y0 + 108 + len(PLAN) * 30 + 26
            text(d, (px + PANEL_W / 2, ay), "Apply plan", font("sb", 11),
                 blend(th.panel, th.green, s["applied"] * a), anchor="mm")

    # code panel text
    if s["split"] > 0.5:
        px0, px1 = cx1, CARD_W - PANEL_W * s["panel"]
        a = clamp((s["split"] - 0.5) / 0.4)
        text(d, (px0 + 16, MAIN_Y0 + 15), "generated.py", font("m", 10),
             blend(th.panel, th.muted, a), anchor="lm")
        if s["sync_pulse"] > 0:
            g = math.sin(math.pi * clamp(s["sync_pulse"]))
            text(d, (px1 - 54, MAIN_Y0 + 15), "graph · code", font("m", 9),
                 blend(th.panel, th.green, g * a), anchor="mm")
        fm = font("m", 9.5)
        syn = {"kw": th.syn_kw, "cls": th.syn_cls, "fn": th.syn_fn,
               "num": th.syn_num, "str": th.syn_str, "com": th.syn_com, "p": th.muted}
        listing = CODE_MOE if s["code_moe"] > 0.5 else CODE
        for i, parts in enumerate(listing):
            if i >= s["code_lines"]:
                break
            frac = clamp(s["code_lines"] - i)
            yy = MAIN_Y0 + 42 + i * 14.6
            if yy > MAIN_Y1 - 12:
                break
            text(d, (px0 + 10, yy), f"{i+1:>2}", font("m", 8),
                 blend(th.bg_deep, th.syn_com, a), anchor="lm")
            xx = px0 + 30
            budget = sum(len(p[0]) for p in parts) * frac
            used = 0
            for txt_, kind in parts:
                if used >= budget:
                    break
                take = txt_[:max(0, round(budget - used))]
                used += len(txt_)
                if not take:
                    continue
                text(d, (xx, yy), take, fm, blend(th.bg_deep, syn[kind], a), anchor="lm")
                xx += d.textlength(take, font=fm)

    # status bar text
    if s["run"] is not None and s["run"] >= 1.0:
        left, rc = "10 cards · 14 typed edges · routed MoE · run complete", th.green
    elif moe_on:
        left, rc = "10 cards · 14 typed edges · routed MoE · PyTorch synced", th.muted
    elif s["nodes"].get("head", 0) > 0.5:
        left, rc = "8 cards · 9 typed edges · validated · PyTorch synced", th.muted
    else:
        left, rc = "Blank starter · local workspace", th.muted
    text(d, (16, MAIN_Y1 + STATUS_H / 2), left, font("m", 9.5), rc, anchor="lm")
    d.ellipse([CARD_W - 140, MAIN_Y1 + STATUS_H / 2 - 3,
               CARD_W - 134, MAIN_Y1 + STATUS_H / 2 + 3], fill=blend(th.bg, th.green, 0.85))
    text(d, (CARD_W - 126, MAIN_Y1 + STATUS_H / 2), "Python 3 runtime",
         font("m", 9.5), th.faint, anchor="lm")

    if cap:
        txt, fade = cap
        f = font("sb", 13)
        wpx = gd.textlength(txt, font=f) + 34
        bx = cx0 + ((cx1 - cx0) - wpx) / 2
        text(gd, (bx + wpx / 2, MAIN_Y1 - 31), txt, f,
             blend(th.panel, th.text, 0.95 * fade), anchor="mm")

    # the clipped canvas layer goes on last so nothing bleeds into the chrome
    base.paste(glimg, (gx0, MAIN_Y0))

    # end-of-loop dissolve back to the empty canvas
    if s["fade"] < 0.999:
        base = Image.blend(paint_empty(th), base, s["fade"])

    return base


_EMPTY_CACHE = {}


def paint_empty(th):
    """The bare app shell, used as the dissolve target at the loop point."""
    if th.name in _EMPTY_CACHE:
        return _EMPTY_CACHE[th.name]
    im = paint(0.02, th)
    _EMPTY_CACHE[th.name] = im
    return im
