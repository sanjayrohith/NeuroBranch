"""Low-level drawing helpers, supersampled for antialiasing."""

import math
import os

from PIL import Image, ImageDraw, ImageFont

SS = 2  # supersample factor for shape layer

FONT_DIR = os.environ.get("NB_FONT_DIR", "/tmp/nb/ttf")
_FONT_CACHE = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _FONT_CACHE:
        name = {
            "r": "Inter-Regular.ttf",
            "sb": "Inter-SemiBold.ttf",
            "b": "Inter-Bold.ttf",
            "m": "JetBrainsMono-Regular.ttf",
            "mb": "JetBrainsMono-Bold.ttf",
        }[kind]
        _FONT_CACHE[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------- easing

def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2


def ease_out_back(t, s=1.35):
    t = clamp(t)
    u = t - 1
    return 1 + (s + 1) * u ** 3 + s * u ** 2


def seg(t, a, b):
    """Normalised progress of t through the window [a, b]."""
    if b <= a:
        return 1.0 if t >= b else 0.0
    return clamp((t - a) / (b - a))


# ---------------------------------------------------------------- colour

def blend(bg, fg, a):
    a = clamp(a)
    return tuple(round(bg[i] + (fg[i] - bg[i]) * a) for i in range(3))


# ---------------------------------------------------------------- shapes

class Shape:
    """Supersampled shape canvas drawn in 1x coordinates.

    An optional origin lets callers keep working in card-global coordinates
    while drawing into a smaller layer (used to clip the graph to the canvas).
    """

    def __init__(self, w, h, bg, origin=(0, 0)):
        self.w, self.h = w, h
        self.ox, self.oy = origin
        self.im = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    def _x(self, v):
        return (v - self.ox) * SS

    def _y(self, v):
        return (v - self.oy) * SS

    def rect(self, x, y, w, h, fill=None, outline=None, width=1):
        self.d.rectangle(
            [self._x(x), self._y(y), self._x(x + w) - 1, self._y(y + h) - 1],
            fill=fill, outline=outline, width=max(1, round(width * SS)),
        )

    def rrect(self, x, y, w, h, r, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(
            [self._x(x), self._y(y), self._x(x + w) - 1, self._y(y + h) - 1],
            radius=r * SS, fill=fill, outline=outline,
            width=max(1, round(width * SS)),
        )

    def line(self, pts, fill, width=1.0):
        p = [(self._x(x), self._y(y)) for x, y in pts]
        self.d.line(p, fill=fill, width=max(1, round(width * SS)), joint="curve")

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.d.ellipse(
            [self._x(cx - r), self._y(cy - r), self._x(cx + r), self._y(cy + r)],
            fill=fill, outline=outline, width=max(1, round(width * SS)),
        )

    def dot(self, cx, cy, r, fill):
        self.circle(cx, cy, r, fill=fill)

    def resolve(self):
        return self.im.resize((self.w, self.h), Image.LANCZOS)


# ---------------------------------------------------------------- bezier

def bezier_pts(p0, p1, p2, p3, n=36):
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        out.append((x, y))
    return out


def edge_path(x1, y1, x2, y2, n=36):
    """React Flow style horizontal bezier."""
    dx = max(28.0, abs(x2 - x1) * 0.45)
    return bezier_pts((x1, y1), (x1 + dx, y1), (x2 - dx, y2), (x2, y2), n)


def partial(pts, p):
    """First p (0..1) of a polyline, interpolating the final segment."""
    p = clamp(p)
    if p <= 0:
        return []
    if p >= 1:
        return pts
    total = len(pts) - 1
    f = total * p
    i = int(f)
    frac = f - i
    out = pts[:i + 1]
    if i < total:
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        out.append((ax + (bx - ax) * frac, ay + (by - ay) * frac))
    return out


def along(pts, p):
    """Point at fraction p along a polyline (by index, good enough here)."""
    p = clamp(p)
    total = len(pts) - 1
    f = total * p
    i = min(int(f), total - 1)
    frac = f - i
    ax, ay = pts[i]
    bx, by = pts[i + 1]
    return (ax + (bx - ax) * frac, ay + (by - ay) * frac)


# ---------------------------------------------------------------- text

def text(d, xy, s, f, fill, anchor="la"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def textw(d, s, f):
    return d.textlength(s, font=f)


def fit(d, s, f, maxw):
    """Truncate with an ellipsis to fit maxw."""
    if d.textlength(s, font=f) <= maxw:
        return s
    while s and d.textlength(s + "…", font=f) > maxw:
        s = s[:-1]
    return s + "…"


def pulse(t, period=1.0, lo=0.0, hi=1.0):
    v = 0.5 - 0.5 * math.cos(2 * math.pi * (t % period) / period)
    return lo + (hi - lo) * v


class Offset:
    """ImageDraw proxy that shifts card-global coordinates into a sub-layer."""

    def __init__(self, d, ox, oy):
        self._d, self._ox, self._oy = d, ox, oy

    def _p(self, box):
        return [box[0] - self._ox, box[1] - self._oy,
                box[2] - self._ox, box[3] - self._oy]

    def text(self, xy, *a, **k):
        self._d.text((xy[0] - self._ox, xy[1] - self._oy), *a, **k)

    def textlength(self, *a, **k):
        return self._d.textlength(*a, **k)

    def rounded_rectangle(self, box, **k):
        self._d.rounded_rectangle(self._p(box), **k)

    def ellipse(self, box, **k):
        self._d.ellipse(self._p(box), **k)

    def line(self, pts, **k):
        self._d.line([(x - self._ox, y - self._oy) for x, y in pts], **k)
