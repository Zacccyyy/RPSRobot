"""
ble_bridge.py
=============
Bluetooth Low Energy bridge for the RPS Robot ESP32.

Replaces serial_bridge.py — same public interface so the rest of
the app (hardware_test_mode.py, robot_output.py, main.py) requires
no changes beyond swapping SerialBridge -> BLEBridge.

Protocol (unchanged from serial):
    Outgoing (Python -> ESP32):  CMD|<action>\\n
    Incoming (ESP32 -> Python):  ACK|<action>\\n  or  ERR|<message>\\n

BLE Service layout (must match ESP32 firmware exactly):
    Service UUID:        6E400001-B5A3-F393-E0A9-E50E24DCCA9E  (Nordic UART)
    TX Characteristic:   6E400002-B5A3-F393-E0A9-E50E24DCCA9E  (write to ESP32)
    RX Characteristic:   6E400003-B5A3-F393-E0A9-E50E24DCCA9E  (notify from ESP32)

The Nordic UART Service (NUS) is the standard way to emulate a serial
port over BLE. It is supported by the NimBLE-Arduino library on ESP32
and by the nRF Connect app for testing.

Dependencies:
    pip install bleak

bleak is pure Python, async, and works on macOS, Windows, and Linux.
On macOS it uses CoreBluetooth. On Windows it uses WinRT BLE APIs.

Usage:
    import asyncio
    from ble_bridge import BLEBridge

    bridge = BLEBridge()

    # Scan for devices
    devices = asyncio.run(bridge.scan_devices())

    # Connect
    asyncio.run(bridge.connect(devices[0].address))

    # Send command
    asyncio.run(bridge.send_command("ROCK"))

    # Read response (non-blocking)
    response = bridge.read_response()

    # Disconnect
    asyncio.run(bridge.disconnect())

For use inside the synchronous OpenCV loop, BLEBridge manages its own
background asyncio event loop in a daemon thread so all async operations
are exposed as regular synchronous methods.
"""

import time
import threading
import asyncio
from collections import deque

# Nordic UART Service UUIDs
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_UUID      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Python writes here
NUS_RX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Python reads here

# Device name the ESP32 advertises
ESP32_DEVICE_NAME = "RPS Robot"

# Scan duration in seconds
SCAN_TIMEOUT = 5.0

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


HARDWARE_COMMANDS = {
    "ROCK":     "CMD|ROCK",
    "PAPER":    "CMD|PAPER",
    "SCISSORS": "CMD|SCISSORS",
    "OPEN":     "CMD|OPEN",
    "CLOSE":    "CMD|CLOSE",
    "PING":     "CMD|PING",
}


class BLEBridge:
    """
    Synchronous wrapper around bleak (async BLE library).

    Maintains a background asyncio event loop in a daemon thread so
    that BLE operations never block the main OpenCV rendering loop.

    Public interface matches SerialBridge exactly for drop-in replacement.
    """

    def __init__(self, log_limit=50):
        self.log_limit = log_limit

        self._client        = None
        self._device_name   = None
        self._device_addr   = None
        self._connected     = False
        self._read_buffer   = ""
        self._response_queue = deque(maxlen=100)
        self._lock          = threading.Lock()

        self.last_command_sent = None
        self.last_command_time = None
        self.last_response     = None
        self.last_response_time = None
        self.command_log       = deque(maxlen=log_limit)

        # Background event loop
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="BLEEventLoop"
        )
        self._thread.start()

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout=10.0):
        """
        Submit a coroutine to the background loop and wait for the result.
        Returns the result or raises the exception.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Device discovery ──────────────────────────────────────────────

    def scan_devices(self, timeout=SCAN_TIMEOUT):
        """
        Scan for BLE devices and return a list of discovered devices.
        Filters to show RPS Robot first, then all others.

        Returns list of dicts: [{"name": str, "address": str}, ...]
        """
        if not BLEAK_AVAILABLE:
            print("[BLE] bleak not installed. Run: pip install bleak")
            return []

        try:
            devices = self._run_async(
                self._async_scan(timeout), timeout=timeout + 2
            )
            return devices
        except Exception as e:
            print(f"[BLE] Scan error: {e}")
            return []

    async def _async_scan(self, timeout):
        # Scan with return_adv=True so we can check service UUIDs
        # macOS hides device names until after first connection,
        # so we identify the ESP32 by its Nordic UART Service UUID instead
        NUS_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
        discovered = await BleakScanner.discover(
            timeout=timeout, return_adv=True)
        result = []
        for addr, (device, adv) in discovered.items():
            name    = device.name or ""
            uuids   = [str(u).lower() for u in adv.service_uuids]
            is_esp  = NUS_UUID in uuids or ESP32_DEVICE_NAME in name
            display = "RPS Robot (ESP32)" if is_esp else (name or "Unknown")
            result.append({
                "name":    display,
                "address": device.address,
                "is_esp":  is_esp,
            })
        # Sort: ESP32 first, then alphabetically
        result.sort(key=lambda x: (0 if x["is_esp"] else 1, x["name"]))
        return result

    # ── Connection ────────────────────────────────────────────────────

    @property
    def is_connected(self):
        return self._connected

    @property
    def port_name(self):
        """Returns device name + address for UI display."""
        if not self._connected:
            return None
        return f"{self._device_name} ({self._device_addr})"

    def connect(self, address, device_name=""):
        """
        Connect to a BLE device by address.
        Returns True on success, False on failure.
        """
        if not BLEAK_AVAILABLE:
            print("[BLE] bleak not installed.")
            return False

        self.disconnect()

        try:
            self._run_async(self._async_connect(address, device_name), timeout=15)
            return self._connected
        except Exception as e:
            print(f"[BLE] Connect error: {e}")
            return False

    async def _async_connect(self, address, device_name):
        try:
            client = BleakClient(address, disconnected_callback=self._on_disconnect)
            await client.connect(timeout=10.0)
            # Subscribe to RX notifications
            await client.start_notify(NUS_RX_UUID, self._on_notify)
            with self._lock:
                self._client       = client
                self._device_addr  = address
                self._device_name  = device_name or address
                self._connected    = True
                self._read_buffer  = ""
            print(f"[BLE] Connected to {device_name} ({address})")
        except Exception as e:
            print(f"[BLE] Async connect failed: {e}")
            self._connected = False

    def _on_disconnect(self, client):
        """Called by bleak when the connection drops."""
        print("[BLE] Disconnected from device")
        with self._lock:
            self._connected   = False
            self._client      = None
            self._device_addr = None
            self._device_name = None

    def disconnect(self):
        """Disconnect from the BLE device."""
        if self._client is not None:
            try:
                self._run_async(self._async_disconnect(), timeout=5)
            except Exception:
                pass
        with self._lock:
            self._connected   = False
            self._client      = None
            self._device_addr = None
            self._device_name = None

    async def _async_disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()

    # ── Send ──────────────────────────────────────────────────────────

    def send_command(self, action):
        """
        Send a command to the ESP32.
        Same interface as SerialBridge.send_command().
        Returns True if sent, False on error.
        """
        if not self._connected or self._client is None:
            return False

        wire_text  = HARDWARE_COMMANDS.get(action, f"CMD|{action}")
        wire_bytes = (wire_text + "\n").encode("utf-8")

        try:
            self._run_async(
                self._async_write(wire_bytes), timeout=5
            )
            now = time.monotonic()
            with self._lock:
                self.last_command_sent = wire_text
                self.last_command_time = now
                self.command_log.append(("TX", wire_text, now))
            print(f"[BLE] TX -> {wire_text}")
            return True
        except Exception as e:
            print(f"[BLE] Write error: {e}")
            return False

    async def _async_write(self, data):
        await self._client.write_gatt_char(NUS_TX_UUID, data, response=False)

    # ── Receive ───────────────────────────────────────────────────────

    def _on_notify(self, sender, data):
        """
        Called by bleak in the BLE event loop thread when ESP32 sends data.
        Buffers incoming bytes and queues complete lines.
        """
        text = data.decode("utf-8", errors="replace")
        lines_to_print = []
        with self._lock:
            self._read_buffer += text
            while "\n" in self._read_buffer:
                line, self._read_buffer = self._read_buffer.split("\n", 1)
                line = line.strip()
                if line:
                    now = time.monotonic()
                    self.last_response      = line
                    self.last_response_time = now
                    self.command_log.append(("RX", line, now))
                    self._response_queue.append(line)
                    lines_to_print.append(line)
        for line in lines_to_print:
            print(f"[BLE] RX <- {line}")

    def read_response(self):
        """
        Non-blocking read. Call once per frame.
        Returns the next complete response line, or None.
        Same interface as SerialBridge.read_response().
        """
        with self._lock:
            if self._response_queue:
                return self._response_queue.popleft()
        return None

    # ── Status ────────────────────────────────────────────────────────

    def list_ports(self):
        """
        Scan for BLE devices. Returns list of address strings for
        compatibility with SerialBridge.list_ports() interface.
        """
        devices = self.scan_devices()
        return [f"{d['name']} | {d['address']}" for d in devices]

    def get_status_summary(self):
        """Returns a dict the UI can display. Matches SerialBridge interface."""
        if not BLEAK_AVAILABLE:
            return {
                "bleak_installed": False,
                "connected": False,
                "port": None,
                "last_tx": None,
                "last_rx": None,
            }
        return {
            "bleak_installed": True,
            "connected": self._connected,
            "port": self.port_name,
            "last_tx": self.last_command_sent,
            "last_rx": self.last_response,
        }
