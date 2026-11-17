"""Pure frame builders for the XYAO-LED BLE wire protocol (no I/O).

Vendored from the ``xyaoled`` library in this repository so the integration is
self-contained (HACS installs only ``custom_components/``). KEEP THE FRAME LOGIC
IN SYNC with ``xyaoled/core.py`` and ``xyaoled/gif.py``; ``tools/check_vendor_sync.py``
verifies the two produce byte-identical frames. See PROTOCOL.md for the full format.
"""
from __future__ import annotations

# Panel geometry
W, H = 64, 16

# BLE identifiers
NAME_PREFIX = "XyaoLED"
SERVICE_UUID = "0000ae00-0000-1000-8000-00805f9b34fb"
WRITE_CHAR_UUID = "0000ae01-0000-1000-8000-00805f9b34fb"
NOTIFY_CHAR_UUID = "0000ae02-0000-1000-8000-00805f9b34fb"

# Every command frame starts with this constant header (magic + protocol constant).
MAGIC = bytes.fromhex("99aa002eff88")

# Known-good handshake frame (command type 0x0000). The device validates a token
# bound to the embedded timestamp and accepts a previously captured frame as-is.
# If a device rejects it, capture your own and set it in the integration options.
DEFAULT_INIT_HEX = "99aa002eff881f00000001001a06030b1a2f030100a1a9df647ff700000000"

# Fixed bytes of the mono image payload; frame count (offset 9), mode (offset 11),
# colour (16-18) and scroll flag / speed (22-23) vary.
PL_HEAD = bytes([0x8C, 0x00, 0x00, 0x02, 0x00, 0x02, 0x01, 0x00, 0x00, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00])
PL_MID = bytes([0x00, 0x00, 0xFF, 0x00, 0x3C, 0x00])
COLOR_CMD_PARAMS = bytes([0xFF, 0x00])

# 11 fixed header bytes (after the 2-byte sub-length) for a static full-colour GIF.
STATIC_HDR_TAIL = bytes([0x00, 0x02, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x08, 0x00])

# The per-frame length field caps at 10000. Subtract the 11-byte frame wrapper and
# the in-payload header (13 bytes for the first 0x0207 frame, 5 bytes for each 0x0209
# continuation) to get the max GIF bytes carried per command.
MAX_0207_GIF = 10000 - 11 - 13
MAX_0209_GIF = 10000 - 11 - 5

# The total GIF size is announced in a u16 sub-length, so one transfer can
# never carry more than 65535 bytes of GIF data.
MAX_GIF_BYTES = 0xFFFF

# Command type carrying image data (triggers a completion notify), and the device
# notification prefix that marks an image/animation transfer as finished.
IMAGE_TYPE = 0x0207
NOTIFY_IMAGE_DONE = b"\x88\xff\x05\x01"

# A mono image is a single 0x0207 frame, so it must fit the 10000-byte cap:
# 11-byte wrapper + 25-byte payload header + 128 bytes per bitmap frame.
MAX_MONO_FRAMES = (10000 - 11 - 25) // 128  # 77


def validate_init(data: bytes) -> None:
    """Structurally validate a captured init frame; raises ValueError.

    A handshake frame is MAGIC + total_len(u16) + type 0x0000 + seq + token,
    so we can catch copy-paste accidents before ever talking to the device.
    """
    if len(data) < 11:
        raise ValueError(f"init frame is {len(data)} bytes, expected at least 11")
    if not data.startswith(MAGIC):
        raise ValueError(f"init frame must start with the magic header {MAGIC.hex()}")
    total = int.from_bytes(data[6:8], "little")
    if total != len(data):
        raise ValueError(f"embedded length {total} does not match frame length {len(data)}")
    if int.from_bytes(data[8:10], "little") != 0x0000:
        raise ValueError("not a handshake frame (command type must be 0x0000)")


def frame(type_le: int, seq: int, params: bytes) -> bytes:
    """Build a command frame: MAGIC + total_len(u16) + type(u16) + seq + params."""
    rest = type_le.to_bytes(2, "little") + bytes([seq & 0xFF]) + bytes(params)
    total = len(MAGIC) + 2 + len(rest)
    return MAGIC + total.to_bytes(2, "little") + rest


def power_cmd(on: bool, seq: int = 1) -> bytes:
    """Panel power (command type 0x0011)."""
    return frame(0x0011, seq, bytes([0x01 if on else 0x02, 0x01, 0x01]))


def brightness_cmd(level: int, seq: int = 1) -> bytes:
    """Brightness 0..100 (command type 0x0012), params `01 [level]`.

    Layout confirmed on hardware with tools/brightness_probe.py.
    """
    level = max(0, min(100, int(level)))
    return frame(0x0012, seq, bytes([0x01, level]))


def clear_cmd(seq: int = 1) -> bytes:
    """Clear screen + playlist (command type 0x0005)."""
    return frame(0x0005, seq, bytes([0x01, 0x01, 0x00, 0x00, 0x00, 0x00]))


def pack_bitmap(grid: list[list[int]]) -> bytes:
    """16 rows x 8 bytes, LSB-first: column = byte_index*8 + bit_index."""
    out = bytearray(H * 8)
    for r in range(H):
        for c in range(W):
            if grid[r][c]:
                out[r * 8 + (c // 8)] |= 1 << (c % 8)
    return bytes(out)


def mono_cmds(grids: list[list[list[int]]], rgb: tuple[int, int, int], seq: int,
              dur: int = 0x3C, scroll: bool = False) -> list[bytes]:
    """1-bit image in a single colour. grids: list of 16x32 grids -> [colour, image]."""
    n = len(grids)
    if not 1 <= n <= MAX_MONO_FRAMES:
        raise ValueError(f"mono image supports 1..{MAX_MONO_FRAMES} frames, got {n}")
    rgb = tuple(rgb)
    if len(rgb) != 3 or not all(isinstance(v, int) and 0 <= v <= 255 for v in rgb):
        raise ValueError(f"rgb must be three ints 0..255, got {rgb!r}")
    bitmap = b"".join(pack_bitmap(g) for g in grids)
    head = bytearray(PL_HEAD)
    head[9] = n
    head[11] = 0x00 if scroll else 0x03
    sub = 12 + n * 128
    head[0], head[1] = sub & 0xFF, (sub >> 8) & 0xFF
    mid = bytearray(PL_MID)
    mid[3] = 0x01 if scroll else 0x00
    mid[4] = dur & 0xFF
    payload = bytes(head) + bytes(rgb) + bytes(mid) + bitmap
    return [frame(0x0204, seq, COLOR_CMD_PARAMS), frame(0x0207, seq, payload)]


def gif_cmds(gif: bytes, nframes: int, seq: int) -> list[bytes]:
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
    color = frame(0x0204, seq, COLOR_CMD_PARAMS)
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


def parse_status(data: bytes) -> dict | None:
    """Parse the device status notify sent after the handshake.

    `88 ff 00 05 01 02 [W:u16][H:u16] 04 [power] ...`

    The brightness also appears in the tail, but its offset varies between
    observed captures, so it is intentionally not parsed (the integration
    tracks brightness optimistically instead).
    """
    if len(data) < 12 or data[0] != 0x88 or data[1] != 0xFF or data[2] != 0x00 or data[3] != 0x05:
        return None
    return {
        "width": int.from_bytes(data[6:8], "little"),
        "height": int.from_bytes(data[8:10], "little"),
        "power": data[11] != 0x02,
    }
