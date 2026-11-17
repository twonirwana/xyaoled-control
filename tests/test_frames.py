"""Unit tests for the wire-protocol frame builders (no BLE, no hardware)."""
import pytest

from xyaoled import core, gif, text
from xyaoled.core import H, W


def blank():
    return [[0] * W for _ in range(H)]


# ------------------------------------------------------------------ frame layout

def test_frame_layout():
    f = core.frame(0x0207, 5, b"\xaa\xbb")
    assert f.startswith(core.MAGIC)
    assert int.from_bytes(f[6:8], "little") == len(f)  # total-length field
    assert int.from_bytes(f[8:10], "little") == 0x0207  # command type
    assert f[10] == 5                                   # sequence byte
    assert f[11:] == b"\xaa\xbb"                        # params


def test_simple_commands():
    assert core.clear_cmd()[8:10] == bytes([0x05, 0x00])
    assert core.power_cmd(True)[11] == 0x01
    assert core.power_cmd(False)[11] == 0x02


@pytest.mark.parametrize("level,expected", [(-5, 0), (0, 0), (42, 42), (100, 100), (250, 100)])
def test_brightness_clamped(level, expected):
    assert core.brightness_cmd(level)[12] == expected


# ------------------------------------------------------------------- mono images

def test_pack_bitmap_bit_positions():
    g = blank()
    g[0][0] = g[0][9] = g[15][63] = 1
    packed = text.pack_bitmap(g)
    assert len(packed) == H * 8
    assert packed[0] == 0b00000001          # row 0, col 0 -> byte 0 bit 0
    assert packed[1] == 0b00000010          # row 0, col 9 -> byte 1 bit 1
    assert packed[15 * 8 + 7] == 0b10000000  # row 15, col 63 -> last byte bit 7


def test_build_mono_header_fields():
    frames = [blank(), blank(), blank()]
    color_cmd, img_cmd = text.build_mono(frames, (10, 20, 30), 2, dur=0x22, scroll=True)
    assert int.from_bytes(color_cmd[8:10], "little") == 0x0204
    payload = img_cmd[11:]
    assert int.from_bytes(payload[0:2], "little") == 12 + 3 * 128  # sub-length
    assert payload[9] == 3                        # frame count
    assert payload[11] == 0x00                    # mode byte: scroll
    assert payload[16:19] == bytes([10, 20, 30])  # colour
    assert payload[19 + 3] == 0x01                # scroll flag in PL_MID
    assert payload[19 + 4] == 0x22                # duration/speed
    assert len(payload) == 25 + 3 * 128


def test_build_mono_static_mode_byte():
    payload = text.build_mono([blank()], (1, 2, 3), 2, scroll=False)[1][11:]
    assert payload[11] == 0x03
    assert payload[19 + 3] == 0x00


@pytest.mark.parametrize("nframes", [0, text.MAX_MONO_FRAMES + 1])
def test_build_mono_rejects_bad_frame_count(nframes):
    with pytest.raises(ValueError, match="frames"):
        text.build_mono([blank()] * nframes, (1, 2, 3), 2)


@pytest.mark.parametrize("rgb", [(255, 0), (0, 0, 0, 0), (256, 0, 0), (-1, 0, 0)])
def test_build_mono_rejects_bad_rgb(rgb):
    with pytest.raises(ValueError, match="rgb"):
        text.build_mono([blank()], rgb, 2)


def test_build_mono_max_frames_fits_device_cap():
    cmds = text.build_mono([blank()] * text.MAX_MONO_FRAMES, (1, 2, 3), 2)
    assert len(cmds[1]) <= 10000


# --------------------------------------------------------------------- GIF frames

@pytest.mark.parametrize("size", [300, 9976, 15000, 30000, 65535])
def test_gif_chunks_reassemble(size):
    blob = bytes((i * 13) & 0xFF for i in range(size))
    cmds = gif.build_cmds(blob, 8, 2)
    first_payload = cmds[1][11:]
    assert int.from_bytes(first_payload[0:2], "little") == size  # announced total
    data = first_payload[13:]                                    # after 13-byte header
    for cont in cmds[2:]:
        assert int.from_bytes(cont[8:10], "little") == 0x0209
        data += cont[11:][5:]                                    # after 5-byte header
    assert data == blob                                          # reassembles losslessly
    assert all(len(c) <= 10000 for c in cmds)


def test_gif_static_single_frame():
    blob = bytes(500)
    cmds = gif.build_cmds(blob, 1, 2)
    assert len(cmds) == 2  # colour + one 0x0207, no continuations
    assert cmds[1][11:][2:13] == gif.STATIC_HDR_TAIL


def test_gif_animation_always_has_continuation():
    cmds = gif.build_cmds(bytes(500), 8, 2)
    assert len(cmds) >= 3
    assert int.from_bytes(cmds[2][8:10], "little") == 0x0209


def test_gif_rejects_oversized():
    with pytest.raises(ValueError, match="65535"):
        gif.build_cmds(bytes(gif.MAX_GIF_BYTES + 1), 8, 2)


# ------------------------------------------------------------- init-frame checks

def test_validate_init_accepts_bundled_default():
    core.validate_init(bytes.fromhex(core.DEFAULT_INIT_HEX))  # must not raise


@pytest.mark.parametrize("bad,reason", [
    (b"\x99\xaa", "at least 11"),
    (bytes(31), "magic header"),
    (bytes.fromhex(core.DEFAULT_INIT_HEX) + b"\x00", "does not match"),
    (core.frame(0x0011, 1, b"\x01\x01\x01"), "command type"),
])
def test_validate_init_rejects_malformed(bad, reason):
    with pytest.raises(ValueError, match=reason):
        core.validate_init(bad)


def test_get_init_rejects_bad_env(monkeypatch):
    monkeypatch.setenv("XYAO_INIT_HEX", "keinhex")
    with pytest.raises(core.XyaoError, match="Invalid XYAO_INIT_HEX"):
        core.get_init()
    monkeypatch.setenv("XYAO_INIT_HEX", bytes(31).hex())
    with pytest.raises(core.XyaoError, match="magic header"):
        core.get_init()
    monkeypatch.delenv("XYAO_INIT_HEX")
    assert core.get_init() == bytes.fromhex(core.DEFAULT_INIT_HEX)


# ------------------------------------------------------------------ input parsing

def test_parse_color_valid():
    assert core.parse_color("255,0,128") == (255, 0, 128)


@pytest.mark.parametrize("bad", ["255,0", "1,2,3,4", "300,0,0", "-1,0,0", "rot", ""])
def test_parse_color_invalid(bad):
    with pytest.raises(ValueError):
        core.parse_color(bad)


def test_parse_status():
    from check_vendor_sync import _load
    import os
    ven = _load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "custom_components", "xyaoled", "protocol.py"), "ven_status")
    frame = bytes([0x88, 0xFF, 0x00, 0x05, 0x01, 0x02, 64, 0, 16, 0, 0x04, 0x01])
    status = ven.parse_status(frame)
    assert status == {"width": 32, "height": 16, "power": True}
    assert ven.parse_status(frame[:-1] + bytes([0x02]))["power"] is False
    assert ven.parse_status(b"\x88\xff\x05\x01") is None  # image-done, not status
    assert ven.parse_status(b"") is None
