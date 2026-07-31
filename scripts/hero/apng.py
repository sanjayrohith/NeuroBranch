"""Minimal hand-rolled APNG writer.

Written by hand rather than through Pillow's APNG support so we get exact
control over colour type (6 = RGBA), bit depth (8), per-frame delays, the
loop count, and region-diffed frames.

Two things keep this practical for a 450-frame showcase:

* Region diffing. The showcase is mostly static UI, so each frame stores only
  the bounding box that actually changed (dispose NONE / blend SOURCE).
* Streaming. Frames are encoded and written as they arrive, so peak memory is
  two frames rather than the whole animation. acTL's frame count is patched in
  once the stream is finished.
"""

import struct
import zlib

import numpy as np

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _filter_scanlines(arr: np.ndarray) -> bytes:
    """PNG scanline filtering, vectorised over all rows at once.

    Every filter type is evaluated for the whole image, then the cheapest one
    per row is selected (the heuristic from the PNG spec). Doing it as five
    whole-array passes instead of a Python loop per row is what makes this
    fast enough to run over hundreds of frames.
    """
    h, w, c = arr.shape
    line = arr.reshape(h, w * c).astype(np.int16)

    left = np.zeros_like(line)
    left[:, c:] = line[:, :-c]
    up = np.zeros_like(line)
    up[1:] = line[:-1]
    upleft = np.zeros_like(line)
    upleft[1:, c:] = line[:-1, :-c]

    p = left + up - upleft
    pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - upleft)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft))

    cands = np.stack([
        line,
        line - left,
        line - up,
        line - ((left + up) >> 1),
        line - paeth,
    ]).astype(np.uint8)

    # cost: sum of |signed byte| per row, smaller is more compressible
    signed = cands.astype(np.int16)
    signed = np.where(signed > 127, signed - 256, signed)
    cost = np.abs(signed).sum(axis=2)
    best = cost.argmin(axis=0).astype(np.uint8)

    rows = cands[best, np.arange(h)]
    out = np.empty((h, w * c + 1), dtype=np.uint8)
    out[:, 0] = best
    out[:, 1:] = rows
    return out.tobytes()


class ApngWriter:
    """Streaming APNG writer. Feed frames with add(); call close() at the end."""

    def __init__(self, path, width, height, loops=0, level=6):
        self.path = path
        self.w, self.h = width, height
        self.level = level
        self.f = open(path, "wb")
        self.f.write(PNG_SIG)
        self.f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        self._actl_off = self.f.tell() + 8      # offset of acTL's data field
        self.f.write(_chunk(b"acTL", struct.pack(">II", 0, loops)))
        self.loops = loops
        self.seq = 0
        self.count = 0
        self.prev = None
        self._pending = None                    # (frame, delay_ms)

    # ---------------------------------------------------------------- api

    def add(self, rgba, delay_ms):
        """Queue a frame. Identical consecutive frames merge into one delay."""
        if self._pending is not None:
            pf, pd = self._pending
            if np.array_equal(pf, rgba):
                self._pending = (pf, pd + delay_ms)
                return
            self._emit(pf, pd)
        self._pending = (rgba, delay_ms)

    def close(self):
        if self._pending is not None:
            self._emit(*self._pending)
            self._pending = None
        self.f.write(_chunk(b"IEND", b""))
        # patch acTL now that the frame count is known
        data = struct.pack(">II", self.count, self.loops)
        self.f.seek(self._actl_off)
        self.f.write(data)
        self.f.write(struct.pack(">I", zlib.crc32(b"acTL" + data) & 0xFFFFFFFF))
        self.f.seek(0, 2)
        size = self.f.tell()
        self.f.close()
        return size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ internals

    def _fctl(self, fw, fh, fx, fy, delay):
        c = _chunk(b"fcTL", struct.pack(
            ">IIIIIHHBB", self.seq, fw, fh, fx, fy,
            delay & 0xFFFF, 1000,
            0,   # dispose_op NONE
            0,   # blend_op SOURCE
        ))
        self.seq += 1
        return c

    def _emit(self, rgba, delay):
        if self.prev is None:
            # first frame doubles as the default image, stored in IDAT
            self.f.write(self._fctl(self.w, self.h, 0, 0, delay))
            self.f.write(_chunk(b"IDAT", zlib.compress(_filter_scanlines(rgba), self.level)))
        else:
            diff = np.any(rgba != self.prev, axis=2)
            ys, xs = np.where(diff)
            if len(ys) == 0:
                x0, y0, x1, y1 = 0, 0, 1, 1
            else:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
            sub = np.ascontiguousarray(rgba[y0:y1, x0:x1])
            self.f.write(self._fctl(x1 - x0, y1 - y0, x0, y0, delay))
            payload = zlib.compress(_filter_scanlines(sub), self.level)
            self.f.write(_chunk(b"fdAT", struct.pack(">I", self.seq) + payload))
            self.seq += 1
        self.prev = rgba
        self.count += 1


def write_apng(path, frames, delays_ms, loops=0, level=6):
    """Non-streaming convenience wrapper."""
    h, w = frames[0].shape[:2]
    wr = ApngWriter(path, w, h, loops=loops, level=level)
    for fr, dl in zip(frames, delays_ms):
        wr.add(fr, dl)
    return wr.close()
