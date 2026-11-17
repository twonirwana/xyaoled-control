"""Connection manager and high-level operations for one XYAO LED panel."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from bleak import BleakClient
from bleak.exc import BleakError
from bleak_retry_connector import BleakClientWithServiceCache, establish_connection

from homeassistant.components import bluetooth
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from . import protocol, render
from .const import (
    DEFAULT_COLOR,
    DEFAULT_SPEED,
    MODE_AUTO,
    MODE_PAGES,
    MODE_SCROLL,
    MODE_STATIC,
)

_LOGGER = logging.getLogger(__name__)

# Seconds to keep the BLE connection open after the last command. The panel
# only accepts one central, so we let go quickly to not lock out the phone app.
IDLE_DISCONNECT = 30.0


async def _wait(event: asyncio.Event, timeout: float) -> None:
    """Wait for a device notification, but never hang: fall back to the timeout."""
    try:
        await asyncio.wait_for(event.wait(), timeout)
    except asyncio.TimeoutError:
        pass


class XyaoLedPanel:
    """One 32x16 panel; serializes commands and manages the BLE session."""

    def __init__(self, hass: HomeAssistant, address: str, init_hex: str | None = None) -> None:
        self._hass = hass
        self.address = address
        self._init_frame = bytes.fromhex(init_hex or protocol.DEFAULT_INIT_HEX)
        self._client: BleakClient | None = None
        self._write_char = None
        self._lock = asyncio.Lock()
        self._disconnect_handle: asyncio.TimerHandle | None = None

        # Last known device state (optimistic, refreshed from the handshake notify).
        self.power: bool = True
        self.brightness: int | None = None  # 0..100
        self.width: int = protocol.W
        self.height: int = protocol.H
        self._listeners: list[Callable[[], None]] = []
        self._last_available: bool | None = None

        # Configurable defaults (options flow) used by notify and the clock.
        self.default_color: tuple[int, int, int] = DEFAULT_COLOR
        self.default_speed: int = DEFAULT_SPEED

        # Upscaled PNG of the last content sent, shown by the image entity.
        self.preview: bytes | None = None
        self.preview_updated = None

        # Set from device notifications so we can wait on them instead of sleeping.
        self._handshaked = asyncio.Event()
        self._image_done = asyncio.Event()

    # ------------------------------------------------------------- state plumbing

    def register_listener(self, cb: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(cb)

        def _unsub() -> None:
            self._listeners.remove(cb)

        return _unsub

    def _fire(self) -> None:
        for cb in self._listeners:
            cb()

    def _set_preview(self, png: bytes) -> None:
        self.preview = png
        self.preview_updated = dt_util.utcnow()
        self._fire()

    @property
    def available(self) -> bool:
        if self._client is not None and self._client.is_connected:
            return True
        return bluetooth.async_address_present(self._hass, self.address, connectable=True)

    @callback
    def check_available(self, *_args: object) -> None:
        """Push availability changes to entities (wired to bluetooth callbacks)."""
        avail = self.available
        if avail != self._last_available:
            self._last_available = avail
            self._fire()

    def _on_notify(self, _char, data: bytearray) -> None:
        b = bytes(data)
        _LOGGER.debug("<- notify: %s", b.hex(" "))
        status = protocol.parse_status(b)
        if status:
            _LOGGER.debug("<- status parsed: %s", status)
            self._handshaked.set()
            self.power = status["power"]
            self.width = status["width"] or self.width
            self.height = status["height"] or self.height
            self._fire()
        elif b.startswith(protocol.NOTIFY_IMAGE_DONE):
            self._image_done.set()

    # ----------------------------------------------------------------- connection

    def _on_disconnect(self, _client: BleakClient) -> None:
        self._client = None
        self._write_char = None
        self._hass.loop.call_soon_threadsafe(self._fire)

    async def _connect(self) -> tuple[BleakClient, object]:
        """Return a connected (client, write_char) pair, connecting if needed.

        Callers must use the returned pair rather than re-reading self._client:
        the disconnected-callback may null the attributes at any moment.
        """
        if self._client is not None and self._client.is_connected:
            return self._client, self._write_char
        ble_device = bluetooth.async_ble_device_from_address(
            self._hass, self.address, connectable=True
        )
        if ble_device is None:
            raise HomeAssistantError(
                f"XYAO panel {self.address} not found. Is it powered on, in range "
                "of a Bluetooth adapter/proxy, and not connected to the phone app?"
            )
        client = await establish_connection(
            BleakClientWithServiceCache,
            ble_device,
            self.address,
            disconnected_callback=self._on_disconnect,
        )
        write_char = notify_char = None
        for service in client.services:
            if service.uuid.lower() == protocol.SERVICE_UUID:
                for ch in service.characteristics:
                    if ch.uuid.lower() == protocol.WRITE_CHAR_UUID:
                        write_char = ch
                    elif ch.uuid.lower() == protocol.NOTIFY_CHAR_UUID:
                        notify_char = ch
        if write_char is None:
            await client.disconnect()
            raise HomeAssistantError(
                "Control characteristic (ae01) not found - is this really a XYAO panel?"
            )
        if notify_char is not None:
            await client.start_notify(notify_char, self._on_notify)
            await asyncio.sleep(0.3)

        _LOGGER.debug(
            "Connected to %s, MTU %s", self.address, getattr(client, "mtu_size", "?")
        )
        # Handshake: without a valid init frame the device ignores all commands.
        _LOGGER.debug("-> init: %s", self._init_frame.hex(" "))
        self._handshaked.clear()
        await client.write_gatt_char(write_char, self._init_frame, response=False)
        await _wait(self._handshaked, 3.0)

        self._client = client
        self._write_char = write_char
        self._fire()
        return client, write_char

    def _schedule_disconnect(self) -> None:
        if self._disconnect_handle is not None:
            self._disconnect_handle.cancel()
        self._disconnect_handle = self._hass.loop.call_later(
            IDLE_DISCONNECT,
            lambda: self._hass.async_create_task(self.disconnect()),
        )

    async def disconnect(self) -> None:
        if self._disconnect_handle is not None:
            self._disconnect_handle.cancel()
            self._disconnect_handle = None
        client, self._client, self._write_char = self._client, None, None
        if client is not None and client.is_connected:
            try:
                await client.disconnect()
            except BleakError:
                pass

    async def _execute(self, cmds: list[bytes], clear_first: bool = False) -> None:
        """Connect (incl. handshake) if needed, then write the command frames."""
        async with self._lock:
            try:
                client, write_char = await self._connect()
                if cmds and self.brightness == 0:
                    _LOGGER.warning(
                        "Panel brightness is 0 - the content will be invisible. "
                        "Turn it up via the panel's light entity."
                    )
                chunk = min(512, (getattr(client, "mtu_size", 515) or 515) - 3)
                _LOGGER.debug(
                    "Sending %d frame(s), chunk size %d%s",
                    len(cmds), chunk, ", clear first" if clear_first else "",
                )
                if clear_first:
                    _LOGGER.debug("-> clear: %s", protocol.clear_cmd().hex(" "))
                    await client.write_gatt_char(
                        write_char, protocol.clear_cmd(), response=False
                    )
                    await asyncio.sleep(0.5)  # clear sends no completion notify
                expect_done = any(
                    int.from_bytes(cmd[8:10], "little") == protocol.IMAGE_TYPE
                    for cmd in cmds
                )
                self._image_done.clear()
                for cmd in cmds:
                    _LOGGER.debug(
                        "-> frame type 0x%04x, %d bytes: %s%s",
                        int.from_bytes(cmd[8:10], "little"), len(cmd),
                        cmd[:48].hex(" "), "..." if len(cmd) > 48 else "",
                    )
                    for i in range(0, len(cmd), chunk):
                        await client.write_gatt_char(
                            write_char, cmd[i : i + chunk], response=False
                        )
                        await asyncio.sleep(0.02)  # pace writes so the buffer keeps up
                    await asyncio.sleep(0.15)       # small gap between frames
                if expect_done:
                    # image/animation transfer ends with a completion notify
                    await _wait(self._image_done, 8.0)
            except BleakError as err:
                await self.disconnect()
                raise HomeAssistantError(f"BLE communication with panel failed: {err}") from err
            except Exception:
                # The panel accepts only one central: never leave a connection
                # dangling without an idle timeout, it would lock out the app.
                await self.disconnect()
                raise
            self._schedule_disconnect()

    # ----------------------------------------------------------------- operations

    async def set_power(self, on: bool) -> None:
        await self._execute([protocol.power_cmd(on)])
        self.power = on
        self._fire()

    async def set_brightness(self, level: int) -> None:
        """level 0..100."""
        await self._execute([protocol.brightness_cmd(level)])
        self.brightness = max(0, min(100, level))
        self._fire()

    async def clear(self) -> None:
        await self._execute([], clear_first=True)
        self._set_preview(await self._hass.async_add_executor_job(render.blank_preview_png))

    async def display_text(
        self,
        message: str,
        rgb: tuple[int, int, int],
        mode: str = MODE_AUTO,
        size: int = 14,
        speed: int = DEFAULT_SPEED,
        font: str | None = None,
        clear: bool = True,
    ) -> None:
        def _render() -> tuple[list, bool]:
            resolved = mode
            if resolved == MODE_AUTO:
                wide = render.text_pixel_width(message, size, font) > protocol.W - 2
                resolved = MODE_SCROLL if wide else MODE_STATIC
            if resolved == MODE_SCROLL:
                return render.render_strip(message, size, font), True
            if resolved == MODE_PAGES:
                return render.render_pages(message, size, font), False
            return [render.render_centered(message, size, font)], False

        grids, scroll = await self._hass.async_add_executor_job(_render)
        try:
            cmds = protocol.mono_cmds(grids, rgb, 2, dur=speed, scroll=scroll)
        except ValueError as err:
            raise HomeAssistantError(
                f"Message is too long for a single transfer ({err}). "
                "Shorten the text or increase the font size."
            ) from err
        if scroll and clear:
            # A static black frame first hides the previous screen scrolling out.
            cmds = protocol.mono_cmds([render.blank_grid()], (0, 0, 0), 1, dur=0x05) + cmds
        await self._execute(cmds, clear_first=clear)
        self._set_preview(
            await self._hass.async_add_executor_job(render.preview_png, grids[0], tuple(rgb))
        )

    async def display_image(self, path: str, fit: str = "contain", clear: bool = True) -> None:
        gif, nframes = await self._hass.async_add_executor_job(render.make_gif, path, fit)
        try:
            cmds = protocol.gif_cmds(gif, nframes, 2)
        except ValueError as err:
            raise HomeAssistantError(f"Image is too large for the panel: {err}") from err
        await self._execute(cmds, clear_first=clear)
        self._set_preview(
            await self._hass.async_add_executor_job(render.preview_png_from_gif, gif)
        )

    async def display_pixel_art(
        self,
        source: str,
        rgb: tuple[int, int, int],
        threshold: int = 128,
        invert: bool = False,
        fit: str = "contain",
        clear: bool = True,
    ) -> None:
        def _grid() -> list[list[int]]:
            if source in render.PATTERNS:
                return render.PATTERNS[source]()
            return render.mono_from_image(source, threshold, invert, fit)

        grid = await self._hass.async_add_executor_job(_grid)
        cmds = protocol.mono_cmds([grid], rgb, 2, scroll=False)
        await self._execute(cmds, clear_first=clear)
        self._set_preview(
            await self._hass.async_add_executor_job(render.preview_png, grid, tuple(rgb))
        )
