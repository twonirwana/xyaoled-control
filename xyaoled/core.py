"""
Core protocol + BLE transport for XYAO-LED style 32x16 BLE pixel matrices.

Reverse-engineered, unofficial. No affiliation with the vendor. For use with
hardware you own. See PROTOCOL.md for the wire format.
"""
from __future__ import annotations

import asyncio
import os

from bleak import BleakClient, BleakScanner

# Panel geometry
W, H = 32, 16

# BLE identifiers (16-bit shorts inside the standard base UUID)
NAME_PREFIX = "XyaoLED"          # advertised device-name prefix used for auto-discovery
SERVICE_PREFIX = "0000ae00"      # control service
WRITE_CHAR_PREFIX = "0000ae01"   # write commands here (write-without-response)
NOTIFY_CHAR_PREFIX = "0000ae02"  # status/ack notifications

# Every command frame starts with this constant header (magic + protocol constant).
MAGIC = bytes.fromhex("99aa002eff88")

# Known-good handshake (command type 0x0000).
DEFAULT_INIT_HEX = "99aa002eff881f00000001001a06030b1a2f030100a1a9df647ff700000000"


class XyaoError(Exception):
    """Device/transport error. The CLI mains map this to a clean exit message."""


def validate_init(data: bytes) -> None:
    if len(data) < 11:
        raise ValueError(f"init frame is {len(data)} bytes, expected at least 11")
    if not data.startswith(MAGIC):
        raise ValueError(f"init frame must start with the magic header {MAGIC.hex()}")
    total = int.from_bytes(data[6:8], "little")
    if total != len(data):
        raise ValueError(f"embedded length {total} does not match frame length {len(data)}")
    if int.from_bytes(data[8:10], "little") != 0x0000:
        raise ValueError("not a handshake frame (command type must be 0x0000)")


def get_init() -> bytes:
    hex_str = os.environ.get("XYAO_INIT_HEX", DEFAULT_INIT_HEX)
    try:
        data = bytes.fromhex(hex_str)
        validate_init(data)
    except ValueError as e:
        raise XyaoError(f"Invalid XYAO_INIT_HEX: {e}") from e
    return data


def parse_color(s: str) -> tuple[int, int, int]:
    try:
        rgb = tuple(int(p) for p in s.split(","))
    except ValueError:
        rgb = ()
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(f"expected R,G,B with values 0..255, got {s!r}")
    return rgb


def frame(type_le: int, seq: int, params: bytes) -> bytes:
    rest = type_le.to_bytes(2, "little") + bytes([seq & 0xFF]) + bytes(params)
    total = len(MAGIC) + 2 + len(rest)
    return MAGIC + total.to_bytes(2, "little") + rest


def clear_cmd(seq: int = 1) -> bytes:
    return frame(0x0005, seq, bytes([0x01, 0x01, 0x00, 0x00, 0x00, 0x00]))


def power_cmd(on: bool, seq: int = 1) -> bytes:
    return frame(0x0011, seq, bytes([0x01 if on else 0x02, 0x01, 0x01]))


def brightness_cmd(level: int, seq: int = 1) -> bytes:
    return frame(0x0012, seq, bytes([0x01, max(0, min(100, int(level)))]))


def checksum(data: bytes) -> int:
    return sum(data) & 0xFF


async def resolve_address(address: str | None = None) -> str:
    address = address or os.environ.get("XYAO_ADDRESS")
    if address:
        return address
    print(f"Scanning for {NAME_PREFIX}...")
    devices = await BleakScanner.discover(timeout=8.0)
    for d in devices:
        if d.name and d.name.startswith(NAME_PREFIX):
            print(f"Found {d.name} @ {d.address}")
            return d.address
    raise XyaoError(
        f"No '{NAME_PREFIX}*' device found. Make sure it is on and not connected to "
        f"the phone app, or set XYAO_ADDRESS / pass --address."
    )


NOTIFY_HANDSHAKE = b"\x88\xff\x00\x05"
NOTIFY_IMAGE_DONE = b"\x88\xff\x05\x01"
IMAGE_TYPE = 0x0207


def _frame_type(cmd: bytes) -> int:
    return int.from_bytes(cmd[8:10], "little") if len(cmd) >= 10 else -1


async def _wait(event: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout)
    except asyncio.TimeoutError:
        pass


async def send(cmds, clear_first=False, address=None, do_init=True, on_notify=None):
    addr = await resolve_address(address)
    async with BleakClient(addr, timeout=20.0) as c:
        write_char = notify_char = None
        for s in c.services:
            if s.uuid.lower().startswith(SERVICE_PREFIX):
                for ch in s.characteristics:
                    if ch.uuid.lower().startswith(WRITE_CHAR_PREFIX):
                        write_char = ch
                    if ch.uuid.lower().startswith(NOTIFY_CHAR_PREFIX):
                        notify_char = ch
        if write_char is None:
            raise XyaoError("Control characteristic (ae01) not found on this device.")

        handshaked = asyncio.Event()
        image_done = asyncio.Event()

        def _notify(char, data):
            b = bytes(data)
            if b.startswith(NOTIFY_HANDSHAKE):
                handshaked.set()
            elif b.startswith(NOTIFY_IMAGE_DONE):
                image_done.set()
            if on_notify:
                on_notify(char, data)

        have_notify = notify_char is not None
        if have_notify:
            await c.start_notify(notify_char, _notify)
            await asyncio.sleep(0.3)

        chunk = min(512, (getattr(c, "mtu_size", 515) or 515) - 3)

        if do_init:
            await c.write_gatt_char(write_char, get_init(), response=False)
            if have_notify:
                await _wait(handshaked, 3.0)
            else:
                await asyncio.sleep(1.0)
        if clear_first:
            await c.write_gatt_char(write_char, clear_cmd(), response=False)
            await asyncio.sleep(0.5)

        expect_done = have_notify and any(_frame_type(cmd) == IMAGE_TYPE for cmd in cmds)
        image_done.clear()
        for cmd in cmds:
            for i in range(0, len(cmd), chunk):
                await c.write_gatt_char(write_char, cmd[i:i + chunk], response=False)
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.15)
        if expect_done:
            await _wait(image_done, 8.0)
        else:
            await asyncio.sleep(0.3)