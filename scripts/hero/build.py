"""Build the NeuroBranch animated hero as an APNG (dark + light).

    python3 build.py            # both variants
    python3 build.py --still 8.5   # write a single frame for inspection

Output is written with a .png extension on purpose: GitHub's markdown renderer
displays animated PNGs only when the file is a .png (an .apng upload is
rejected outright by the attachment allowlist).
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apng import ApngWriter           # noqa: E402
from scene import CARD_H, CARD_W, DURATION, paint  # noqa: E402
from theme import DARK, LIGHT         # noqa: E402

PAD = 36                 # fully transparent padding for the shadow
RADIUS = 18              # rounded corner radius (brief asks for 16-20)
OUT_W, OUT_H = CARD_W + PAD * 2, CARD_H + PAD * 2
MASK_SS = 4              # supersample for a clean antialiased corner mask

SHADOW_BLUR = 9.0
SHADOW_DY = 6
SHADOW_INSET = 6


def build_mask():
    """Antialiased rounded-rect alpha mask for the card."""
    m = Image.new("L", (CARD_W * MASK_SS, CARD_H * MASK_SS), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, CARD_W * MASK_SS - 1, CARD_H * MASK_SS - 1],
        radius=RADIUS * MASK_SS, fill=255,
    )
    return m.resize((CARD_W, CARD_H), Image.LANCZOS)


def build_shadow(mask, theme):
    """Real Gaussian-blurred shadow, rendered straight into the alpha channel."""
    canvas = Image.new("L", (OUT_W, OUT_H), 0)
    src = mask.point(lambda v: min(255, round(v * (theme.shadow_alpha / 255.0) * 255 / 255)))
    src = mask.point(lambda v: round(v * theme.shadow_alpha / 255.0))
    shrunk = src.resize((CARD_W - SHADOW_INSET * 2, CARD_H - SHADOW_INSET * 2), Image.LANCZOS)
    canvas.paste(shrunk, (PAD + SHADOW_INSET, PAD + SHADOW_INSET + SHADOW_DY))
    return canvas.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))


def composite(card_rgb, mask, shadow):
    """Card 'over' shadow, straight (non-premultiplied) RGBA out.

    Where the card covers, alpha is 255 and colour is the card. Outside the
    rounded corners the pixel is black with the shadow's own graduated alpha,
    so corner pixels land at exactly alpha 0 and the shadow band carries real
    intermediate alpha rather than 1-bit transparency.
    """
    ma = np.zeros((OUT_H, OUT_W), dtype=np.float32)
    ma[PAD:PAD + CARD_H, PAD:PAD + CARD_W] = np.asarray(mask, dtype=np.float32) / 255.0
    sa = np.asarray(shadow, dtype=np.float32) / 255.0

    rgb = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)
    rgb[PAD:PAD + CARD_H, PAD:PAD + CARD_W] = np.asarray(card_rgb, dtype=np.float32)

    out_a = ma + sa * (1.0 - ma)
    safe = np.maximum(out_a, 1e-6)
    out_rgb = rgb * (ma[..., None] / safe[..., None])

    out = np.empty((OUT_H, OUT_W, 4), dtype=np.uint8)
    out[..., :3] = np.clip(out_rgb + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def render_variant(theme, fps, out_path, verbose=True, level=6):
    """Render and stream straight into the APNG; peak memory is two frames."""
    mask = build_mask()
    shadow = build_shadow(mask, theme)

    n = int(round(DURATION * fps))
    step_ms = round(1000 / fps)

    wr = ApngWriter(out_path, OUT_W, OUT_H, loops=0, level=level)
    for i in range(n):
        rgba = composite(paint(i / fps, theme), mask, shadow)
        wr.add(rgba, step_ms)
        if verbose and i % 50 == 0:
            print(f"  {theme.name}: {i}/{n}", flush=True)
    size = wr.close()
    return wr.count, n * step_ms, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "public"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--contact", action="store_true")
    args = ap.parse_args()

    mask = build_mask()

    if args.still is not None:
        for th in (DARK, LIGHT):
            sh = build_shadow(mask, th)
            arr = composite(paint(args.still, th), mask, sh)
            Image.fromarray(arr, "RGBA").save(f"/tmp/still_{th.name}.png")
            print(f"/tmp/still_{th.name}.png")
        return

    if args.contact:
        for th in (DARK,):
            sh = build_shadow(mask, th)
            times = [0.8, 3.4, 5.9, 6.9, 8.8, 11.0, 12.6, 14.6, 16.6, 18.6, 20.6, 21.9]
            sheet = Image.new("RGB", (OUT_W // 2 * 4, OUT_H // 2 * 3), (20, 20, 20))
            for i, t in enumerate(times):
                arr = composite(paint(t, th), mask, sh)
                im = Image.fromarray(arr, "RGBA").convert("RGB").resize(
                    (OUT_W // 2, OUT_H // 2), Image.LANCZOS)
                sheet.paste(im, ((i % 4) * (OUT_W // 2), (i // 4) * (OUT_H // 2)))
            sheet.save("/tmp/contact.png")
            print("/tmp/contact.png")
        return

    os.makedirs(args.out, exist_ok=True)
    for th, name in ((DARK, "neurobranch-hero-dark.png"),
                     (LIGHT, "neurobranch-hero-light.png")):
        path = os.path.normpath(os.path.join(args.out, name))
        nf, total, size = render_variant(th, fps=args.fps, out_path=path)
        print(f"{path}\n  {OUT_W}x{OUT_H}  frames={nf}  "
              f"duration={total/1000:.2f}s  size={size/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
