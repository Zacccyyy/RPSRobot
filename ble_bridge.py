"""
ble_bridge.py
=============
Bluetooth Low Energy bridge for the RPS Robot ESP32.

This module is a drop-in replacement for serial_bridge.py — the public
interface is identical so the rest of the app (hardware_test_mode.py,
robot_output.py, main.py) only needs to swap SerialBridge -> BLEBridge.

Communication protocol (same as serial):
    Outgoing (Python -> ESP32):  CMD|<action>\\n
    Incoming (ESP32 -> Python):  ACK|<action>\\n  or  ERR|<message>\\n

BLE service layout (must match the ESP32 firmware exactly):
    Service UUID:       6E400001-B5A3-F393-E0A9-E50E24DCCA9E  (Nordic UART)
    TX Characteristic:  6E400002-B5A3-F393-E0A9-E50E24DCCA9E  (Python writes here)
    RX Characteristic:  6E400003-B5A3-F393-E0A9-E50E24DCCA9E  (ESP32 notifies here)

The Nordic UART Service (NUS) emulates a serial port over BLE.  It is
supported by the NimBLE-Arduino library on ESP32 and by the nRF Connect
app for manual testing.

Dependencies:
    pip install bleak

bleak is pure Python, async, and works on macOS (CoreBluetooth),
Windows (WinRT BLE), and Linux (BlueZ).

Because the rest of the game runs synchronously inside an OpenCV loop,
BLEBridge manages its own background asyncio event loop in a daemon thread
and exposes everything as regular synchronous methods.
"""

import time
import threading
import asyncio
from collections import deque

# Nordic UART Service UUIDs (standard BLE UART emulation protocol).
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_UUID      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Python writes to this characteristic
NUS_RX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # ESP32 sends notifications on this one

# The BLE advertisement name the ESP32 firmware broadcasts.
ESP32_DEVICE_NAME = "RPS Robot"

# How long to scan for nearby BLE devices before giving up (seconds).
SCAN_TIMEOUT = 5.0

# Try to import bleak.  If it isn't installed we set a flag and fail gracefully
# instead of crashing at import time.
try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False


# Maps human-readable action names to the wire format string the ESP32 expects.
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
    Synchronous wrapper around the bleak async BLE library.

    The game loop is synchronous but bleak is async, so this class runs
    a private asyncio event loop in a background daemon thread.  All
    public methods submit coroutines to that loop and block until they
    complete, which keeps the public API simple and synchronous.

    Public interface matches SerialBridge exactly for drop-in replacement.
    """

    def __init__(self, log_limit=50):
        # How many TX/RX entries to keep in the command log.
        self.log_limit   = log_limit
        self.bridge_type = "BLE"   # used by HardwareTestController to know which mode is active

        # BLE connection state.
        self._client      = None
        self._device_name = None
        self._device_addr = None
        self._connected   = False

        # Buffer for partial lines arriving from the ESP32.
        self._read_buffer    = ""
        # Queue of complete response lines waiting to be consumed by read_response().
        self._response_queue = deque(maxlen=100)

        # Thread lock protecting all shared state between the main thread
        # and the background BLE event loop thread.
        self._lock = threading.Lock()

        # Last TX/RX tracking for the UI status display.
        self.last_command_sent  = None
        self.last_command_time  = None
        self.last_response      = None
        self.last_response_time = None
        self.command_log        = deque(maxlen=log_limit)

        # Create and start the background asyncio event loop in a daemon thread.
        # Using a dedicated loop (rather than asyncio.run) lets us submit
        # coroutines from the main thread at any time.
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="BLEEventLoop",
        )
        self._thread.start()

    def _run_loop(self):
        """Entry point for the background daemon thread — runs the event loop forever."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_async(self, coro, timeout=10.0):
        """
        Submit a coroutine to the background event loop and wait for its result.

        This is how synchronous callers can run async BLE operations.
        Raises the coroutine's exception if it fails within the timeout.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ── Device discovery ──────────────────────────────────────────────────────

    def scan_devices(self, timeout=SCAN_TIMEOUT):
        """
        Scan for nearby BLE devices and return a list of discovered ones.

        The ESP32 running the NUS firmware is identified by its service UUID
        (more reliable than name matching, because macOS hides device names
        until after the first connection).  ESP32 devices are sorted first.

        Returns a list of dicts: [{"name": str, "address": str, "is_esp": bool}, ...]
        Returns an empty list if bleak isn't installed or scanning fails.
        """
        if not BLEAK_AVAILABLE:
            print("[BLE] bleak not installed. Run: pip install bleak")
            return []

        try:
            # Allow extra time for the async scan to complete.
            devices = self._run_async(
                self._async_scan(timeout), timeout=timeout + 2
            )
            return devices
        except Exception as e:
            print(f"[BLE] Scan error: {e}")
            return []

    async def _async_scan(self, timeout):
        """
        Async implementation of device discovery.

        return_adv=True gives us the advertisement data, including the list
        of service UUIDs — that's how we identify the ESP32 on macOS where
        device names aren't available until after connection.
        """
        NUS_UUID   = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)

        result = []
        for addr, (device, adv) in discovered.items():
            name    = device.name or ""
            uuids   = [str(u).lower() for u in adv.service_uuids]
            # Identify as the ESP32 if it advertises the NUS UUID or has the right name.
            is_esp  = NUS_UUID in uuids or ESP32_DEVICE_NAME in name
            display = "RPS Robot (ESP32)" if is_esp else (name or "Unknown")
            result.append({
                "name":    display,
                "address": device.address,
                "is_esp":  is_esp,
            })

        # Put the ESP32 at the top, then sort the rest alphabetically.
        result.sort(key=lambda x: (0 if x["is_esp"] else 1, x["name"]))
        return result

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def is_connected(self):
        """True if currently connected to a BLE device."""
        return self._connected

    @property
    def port_name(self):
        """Device name + address string for UI display.  None if not connected."""
        if not self._connected:
            return None
        return f"{self._device_name} ({self._device_addr})"

    def connect(self, address, device_name=""):
        """
        Connect to a BLE device by its hardware address.

        Any existing connection is dropped first.
        Returns True on success, False on failure.
        """
        if not BLEAK_AVAILABLE:
            print("[BLE] bleak not installed.")
            return False

        # Always disconnect cleanly before attempting a new connection.
        self.disconnect()

        try:
            # Allow 15 seconds — BLE connection setup can be slow.
            self._run_async(self._async_connect(address, device_name), timeout=15)
            return self._connected
        except Exception as e:
            print(f"[BLE] Connect error: {e}")
            return False

    async def _async_connect(self, address, device_name):
        """
        Async implementation of the connection sequence.

        After connecting we subscribe to RX characteristic notifications
        so that incoming data from the ESP32 will call _on_notify()
        automatically without us having to poll.
        """
        try:
            client = BleakClient(address, disconnected_callback=self._on_disconnect)
            await client.connect(timeout=10.0)
            # Subscribe to notifications from the ESP32's TX (our RX) characteristic.
            await client.start_notify(NUS_RX_UUID, self._on_notify)
            with self._lock:
                self._client      = client
                self._device_addr = address
                self._device_name = device_name or address
                self._connected   = True
                self._read_buffer = ""
            print(f"[BLE] Connected to {device_name} ({address})")
        except Exception as e:
            print(f"[BLE] Async connect failed: {e}")
            self._connected = False

    def _on_disconnect(self, client):
        """
        Called automatically by bleak when the BLE connection drops unexpectedly.
        Clears the connection state so the rest of the app knows we're offline.
        """
        print("[BLE] Disconnected from device")
        with self._lock:
            self._connected   = False
            self._client      = None
            self._device_addr = None
            self._device_name = None

    def disconnect(self):
        """Disconnect from the BLE device gracefully."""
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
        """Async implementation: disconnect if the client is still connected."""
        if self._client and self._client.is_connected:
            await self._client.disconnect()

    # ── Send ──────────────────────────────────────────────────────────────────

    def send_command(self, action):
        """
        Send a command to the ESP32 over BLE.

        Looks up the action in HARDWARE_COMMANDS; falls back to "CMD|<action>"
        if the action isn't in the map.  Appends a newline because the ESP32
        firmware reads line-by-line.

        Returns True if the write succeeded, False otherwise.
        Same interface as SerialBridge.send_command().
        """
        if not self._connected or self._client is None:
            return False

        wire_text  = HARDWARE_COMMANDS.get(action, f"CMD|{action}")
        wire_bytes = (wire_text + "\n").encode("utf-8")

        try:
            self._run_async(self._async_write(wire_bytes), timeout=5)
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
        """Write raw bytes to the ESP32's RX (our TX) characteristic."""
        # response=False means "write without response" (faster, fire-and-forget).
        await self._client.write_gatt_char(NUS_TX_UUID, data, response=False)

    # ── Receive ───────────────────────────────────────────────────────────────

    def _on_notify(self, sender, data):
        """
        Called by bleak (in the BLE event loop thread) whenever the ESP32 sends data.

        Incoming bytes are appended to a line buffer.  Whenever a newline is
        found, the complete line is extracted and pushed onto the response queue
        for the main thread to consume via read_response().
        """
        text = data.decode("utf-8", errors="replace")

        # We need to print OUTSIDE the lock to avoid holding it during I/O.
        lines_to_print = []

        with self._lock:
            self._read_buffer += text
            # Extract all complete lines from the buffer.
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
        Non-blocking read.  Call once per frame from the main loop.

        Returns the next complete response line from the ESP32, or None if
        nothing has arrived since the last call.
        Same interface as SerialBridge.read_response().
        """
        with self._lock:
            if self._response_queue:
                return self._response_queue.popleft()
        return None

    # ── Status helpers (for UI) ───────────────────────────────────────────────

    def list_ports(self):
        """
        Scan for BLE devices and return a list of display strings.

        Format: "<name> | <address>" for each found device.
        Provides compatibility with SerialBridge.list_ports() interface.
        """
        devices = self.scan_devices()
        return [f"{d['name']} | {d['address']}" for d in devices]

    def get_status_summary(self):
        """
        Return a dict describing the current BLE connection state.

        Used by the UI to show status information.
        Matches the SerialBridge.get_status_summary() interface.
        """
        if not BLEAK_AVAILABLE:
            return {
                "bleak_installed": False,
                "connected":       False,
                "port":            None,
                "last_tx":         None,
                "last_rx":         None,
            }
        return {
            "bleak_installed": True,
            "connected":       self._connected,
            "port":            self.port_name,
            "last_tx":         self.last_command_sent,
            "last_rx":         self.last_response,
        }
