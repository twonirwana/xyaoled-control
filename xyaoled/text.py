"""Render text to the 32x16 panel: static, paginated, or hardware-scrolled.

The panel's text/bitmap format is 1 bit per pixel plus a single RGB colour.
"""
import argparse
import asyncio
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

from . import core
from .core import W, H, frame

BYTES_PER_ROW = W // 8                # 4 bytes for 32px width
BYTES_PER_FRAME = H * BYTES_PER_ROW   # 64 bytes per 32x16 frame

# Fixed bytes of the mono image payload (see PROTOCOL.md).
# Note: Bytes at index 3 and 5 (0x02, 0x00, 0x02) may represent width/module count
# on some firmware versions. If the image appears split or shifted, try changing
# those two 0x02 bytes to 0x01.
PL_HEAD = bytes([0x4C, 0x00, 0x00, 0x02, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00])
PL_MID = bytes([0x00, 0x00, 0xFF, 0x00, 0x3C, 0x00])
COLOR_CMD_PARAMS = bytes([0xFF, 0x00])

# 11-byte wrapper + 25-byte payload header + BYTES_PER_FRAME (64) per bitmap frame.
MAX_MONO_FRAMES = (10000 - 11 - 25) // BYTES_PER_FRAME  # 155

FONTS = [
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
]


@lru_cache(maxsize=16)
def _load_font(size, font_path):
    for fp in ([font_path] if font_path else []) + FONTS:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _to_grid(img):
    return [[img.getpixel((c, r)) for c in range(W)] for r in range(H)]


def render_centered(text, size, x_off, y_off, font_path):
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    font = _load_font(size, font_path)
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((W - tw) // 2 - bbox[0] + x_off, (H - th) // 2 - bbox[1] + y_off), text, fill=1, font=font)
    return _to_grid(img)


def _render_wide(text, size, font_path, pad_left=0):
    font = _load_font(size, font_path)
    probe = ImageDraw.Draw(Image.new("1", (8, 8)))
    bbox = probe.textbbox((0, 0), text, font=font)
    img = Image.new("1", (bbox[2] - bbox[0] + pad_left + 2, H), 0)
    d = ImageDraw.Draw(img)
    d.text((pad_left - bbox[0], (H - (bbox[3] - bbox[1])) // 2 - bbox[1]), text, fill=1, font=font)
    return img


def render_pages(text, size, font_path):
    """Split a long string into 32px pages at word boundaries."""
    d = ImageDraw.Draw(Image.new("1", (8, 8)))
    words = text.split() or [text]
    while size > 6:
        font = _load_font(size, font_path)
        if max(d.textbbox((0, 0), w, font=font)[2] for w in words) <= W - 2:
            break
        size -= 1
    font = _load_font(size, font_path)
    width = lambda s: d.textbbox((0, 0), s, font=font)[2]
    pages, cur = [], ""
    for w in text.split(" "):
        cand = (cur + " " + w).strip()
        if cur and width(cand) > W - 2:
            pages.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        pages.append(cur)
    return [render_centered(p, size, 0, 0, font_path) for p in pages]


def render_strip(text, size, font_path, lead=2, trail=W):
    """A wide N*32px strip the device scrolls itself."""
    img = _render_wide(text, size, font_path, pad_left=lead)
    n = max(1, (img.width + trail + W - 1) // W)
    canvas = Image.new("1", (n * W, H), 0)
    canvas.paste(img, (0, 0))
    return [_to_grid(canvas.crop((f * W, 0, f * W + W, H))) for f in range(n)]


def pack_bitmap(grid):
    """16 rows x 4 bytes (for 32px width), LSB-first: column = byte_index*8 + bit_index."""
    out = bytearray(BYTES_PER_FRAME)
    for r in range(H):
        for c in range(W):
            if grid[r][c]:
                out[r * BYTES_PER_ROW + (c // 8)] |= 1 << (c % 8)
    return bytes(out)


def build_mono(frames, rgb, seq, dur=0x3C, scroll=False):
    """frames: list of 16x32 grids. Returns [colour_cmd, image_cmd]."""
    n = len(frames)
    if not 1 <= n <= MAX_MONO_FRAMES:
        raise ValueError(f"mono image supports 1..{MAX_MONO_FRAMES} frames, got {n}")
    rgb = tuple(rgb)
    if len(rgb) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in rgb):
        raise ValueError(f"rgb must be three ints 0..255, got {rgb!r}")
    bitmap = b"".join(pack_bitmap(g) for g in frames)
    head = bytearray(PL_HEAD)
    head[9] = n
    head[11] = 0x00 if scroll else 0x03
    sub = 12 + n * BYTES_PER_FRAME
    head[0], head[1] = sub & 0xFF, (sub >> 8) & 0xFF
    mid = bytearray(PL_MID)
    mid[3] = 0x01 if scroll else 0x00
    mid[4] = dur & 0xFF
    payload = bytes(head) + bytes(rgb) + bytes(mid) + bitmap
    return [frame(0x0204, seq, COLOR_CMD_PARAMS), frame(0x0207, seq, payload)]


def _ascii(grid):
    for r in range(H):
        print("".join("#" if grid[r][c] else "." for c in range(W)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show text on a XYAO-LED 32x16 panel.")
    ap.add_argument("text", nargs="+")
    ap.add_argument("--color", type=core.parse_color, default="255,0,0", help="R,G,B (0..255)")
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--pages", action="store_true", help="split long text into pages")
    ap.add_argument("--scroll", action="store_true", help="hardware horizontal scroll")
    ap.add_argument("--dur", type=lambda x: int(x, 0), default=None, help="frame time / speed")
    ap.add_argument("--font", default=None)
    ap.add_argument("--no-black-first", action="store_true", help="don't pre-clear before scrolling")
    ap.add_argument("--clear", action="store_true", help="clear playlist before sending")
    ap.add_argument("--address", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    text = " ".join(a.text)
    rgb = a.color

    if a.scroll:
        frames = render_strip(text, a.size, a.font)
        dur = a.dur if a.dur is not None else 0x3C
    elif a.pages:
        frames = render_pages(text, a.size, a.font)
        dur = a.dur if a.dur is not None else 0x3C
    else:
        frames = [render_centered(text, a.size, 0, 0, a.font)]
        dur = a.dur if a.dur is not None else 0x3C

    for g in frames[:6]:
        _ascii(g)
        print()
    try:
        cmds = build_mono(frames, rgb, 2, dur=dur, scroll=a.scroll)
    except ValueError as e:
        ap.error(str(e))

    if a.scroll and not a.no_black_first:
        black = [[0] * W for _ in range(H)]
        cmds = build_mono([black], [0, 0, 0], 1, dur=0x05) + cmds

    if a.dry_run:
        print(f"{len(frames)} frame(s); image-cmd {len(cmds[-1])} bytes (not sent)")
        return
    try:
        asyncio.run(core.send(cmds, clear_first=a.clear, address=a.address,
                              on_notify=lambda _, d: print("notify:", bytes(d).hex(" "))))
    except core.XyaoError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()