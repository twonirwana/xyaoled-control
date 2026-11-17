"""Full-colour images and animations via an embedded GIF89a (mode 0x08)."""
import argparse
import asyncio
import io

from PIL import Image, ImageSequence, ImageOps

from . import core
from .core import W, H, frame

# 11 fixed header bytes (after the 2-byte sub-length) for a STATIC full-colour GIF.
STATIC_HDR_TAIL = bytes([0x00, 0x02, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x08, 0x00])

# The per-frame length field caps at 10000. Subtract the 11-byte frame wrapper
# (MAGIC + LEN + TYPE + SEQ) and the in-payload header to get the max GIF bytes per
# command: 13-byte header for the first (0x0207) frame, 5-byte header for each
# continuation (0x0209) frame.
MAX_0207_GIF = 10000 - 11 - 13
MAX_0209_GIF = 10000 - 11 - 5

# The total GIF size is announced in a u16 sub-length, so one transfer can
# never carry more than 65535 bytes of GIF data.
MAX_GIF_BYTES = 0xFFFF


def _fit(im, fit):
    im = im.convert("RGB")
    if fit == "stretch":
        return im.resize((W, H))
    im2 = ImageOps.contain(im, (W, H))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(im2, ((W - im2.width) // 2, (H - im2.height) // 2))
    return canvas


def make_gif(path, fit):
    src = Image.open(path)
    frames, durs = [], []
    # One pass: _fit() forces the frame to load, and only a loaded frame has a
    # reliable per-frame duration in info (WebP fills it in lazily on load).
    for f in ImageSequence.Iterator(src):
        frames.append(_fit(f, fit))
        durs.append(int(f.info.get("duration") or 100))
    pal = [fr.convert("P", palette=Image.ADAPTIVE, colors=256) for fr in frames]
    buf = io.BytesIO()
    pal[0].save(buf, format="GIF", save_all=True, append_images=pal[1:],
                duration=durs, loop=0, version="GIF89a")
    data = bytearray(buf.getvalue())
    data[:6] = b"GIF89a"
    return bytes(data), len(pal)


def build_cmds(gif, nframes, seq):
    """Static colour image (single 0x0207) or animation (0x0207 + one/more 0x0209).

    The device accumulates GIF bytes until it has the total announced in the 0x0207
    sub-length, so large animations are chunked across as many 0x0209 continuation
    frames as needed (each within the 10000-byte per-frame limit).
    """
    if len(gif) > MAX_GIF_BYTES:
        raise ValueError(
            f"GIF is {len(gif)} bytes; the protocol caps one transfer at {MAX_GIF_BYTES}. "
            "Use fewer frames or a simpler image."
        )
    color = frame(0x0204, seq, bytes([0xFF, 0x00]))
    if nframes <= 1:
        payload = len(gif).to_bytes(2, "little") + STATIC_HDR_TAIL + gif
        return [color, frame(0x0207, seq, payload)]
    # Animation: header byte 3 = 0x03, byte 11 = 0x03, frame count at byte 9.
    # Bytes 5..8 are don't-care. Animations MUST always have at least one 0x0209.
    hdr = bytes([0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, nframes & 0xFF, 0x00, 0x03, 0x00])
    split = MAX_0207_GIF if len(gif) > MAX_0207_GIF else max(1, len(gif) // 2)
    cmds = [color, frame(0x0207, seq, len(gif).to_bytes(2, "little") + hdr + gif[:split])]
    rest = gif[split:]
    for i in range(0, len(rest), MAX_0209_GIF):
        chunk = rest[i:i + MAX_0209_GIF]
        cmds.append(frame(0x0209, seq, len(chunk).to_bytes(2, "little") + bytes([0x00, 0x02, 0x00]) + chunk))
    return cmds


def main(argv=None):
    ap = argparse.ArgumentParser(description="Full-colour image / animation on a XYAO-LED panel.")
    ap.add_argument("image", help="any image, or an animated GIF for animation")
    ap.add_argument("--fit", choices=["contain", "stretch"], default="contain")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--address", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    gif, n = make_gif(a.image, a.fit)
    try:
        cmds = build_cmds(gif, n, 2)
    except ValueError as e:
        ap.error(str(e))
    print(f"{n} frame(s), GIF {len(gif)} bytes, {len(cmds)} commands")
    if a.dry_run:
        print("(not sent)")
        return
    try:
        asyncio.run(core.send(cmds, clear_first=a.clear, address=a.address,
                              on_notify=lambda _, d: print("notify:", bytes(d).hex(" "))))
    except core.XyaoError as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    main()