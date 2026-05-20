"""
serial_bridge.py
================
USB Serial bridge for the ESP32 robot hand.

Protocol (pipe-delimited plain text — easy to parse on both sides):
    Outgoing (Python → ESP32):  CMD|<action>\n
    Incoming (ESP32 → Python):  ACK|<action>\n   or   ERR|<message>\n

Usage:
    bridge = SerialBridge()
    ports  = bridge.list_ports()           # e.g. ["/dev/cu.usbserial-0001"]
    bridge.connect("/dev/cu.usbserial-0001")
    bridge.send_command("ROCK")            # sends  CMD|ROCK\n
    response = bridge.read_response()      # non-blocking, returns str or None
    bridge.disconnect()
"""

import time
from collections import deque

# Try to import pyserial.  If it isn't installed we set a flag and fail
# gracefully instead of crashing at import time.
try:
    import serial
    import serial.tools.list_ports
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False


# Maps human-readable action names to the wire format the ESP32 expects.
HARDWARE_COMMANDS = {
    "ROCK":     "CMD|ROCK",
    "PAPER":    "CMD|PAPER",
    "SCISSORS": "CMD|SCISSORS",
    "OPEN":     "CMD|OPEN",
    "CLOSE":    "CMD|CLOSE",
    "PING":     "CMD|PING",
}


class SerialBridge:
    """
    Thin wrapper around pyserial for the RPS robot.

    - Non-blocking reads (timeout=0) so the OpenCV loop never stalls.
    - Keeps a small log of recent commands and responses for the UI overlay.
    """

    def __init__(self, baud_rate=115200, log_limit=50):
        """
        baud_rate : must match the ESP32 firmware setting (default 115200).
        log_limit : max entries to keep in the command_log deque.
        """
        self.baud_rate   = baud_rate
        self.log_limit   = log_limit
        self.bridge_type = "SERIAL"  # used by HardwareTestController to know the active mode

        # Internal serial port object (None when disconnected).
        self._serial = None
        self._port_name   = None
        self._read_buffer = ""  # partial line buffer — data arrives in chunks

        # Last TX/RX tracking for the UI status display.
        self.last_command_sent  = None
        self.last_command_time  = None
        self.last_response      = None
        self.last_response_time = None

        # Rolling log of (direction, message, timestamp) tuples.
        self.command_log = deque(maxlen=log_limit)

    # ── Port discovery ────────────────────────────────────────────────────────

    @staticmethod
    def list_ports():
        """
        Return a list of available serial port device paths.

        On macOS these typically look like:
            /dev/cu.usbserial-0001
            /dev/cu.usbmodem14101

        Returns an empty list if pyserial is not installed.
        """
        if not PYSERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    # ── Connection ────────────────────────────────────────────────────────────

    @property
    def is_connected(self):
        """True if the serial port is open and ready."""
        return self._serial is not None and self._serial.is_open

    @property
    def port_name(self):
        """The active port path, or None if not connected."""
        return self._port_name if self.is_connected else None

    def connect(self, port) -> bool:
        """
        Open a serial connection to the given port.
        Disconnects any existing connection first.
        Returns True on success, False on failure.
        """
        if not PYSERIAL_AVAILABLE:
            print("[SerialBridge] pyserial is not installed.")
            return False

        # Drop any existing connection cleanly before opening a new one.
        self.disconnect()

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=0,        # non-blocking reads (returns immediately)
                write_timeout=1.0,
            )
            self._port_name   = port
            self._read_buffer = ""
            print(f"[SerialBridge] Connected to {port} @ {self.baud_rate}")
            return True

        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Failed to connect to {port}: {exc}")
            self._serial    = None
            self._port_name = None
            return False

    def disconnect(self):
        """Close the serial connection if it's open and clear all state."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass  # ignore errors on close — we're disconnecting anyway

        self._serial      = None
        self._port_name   = None
        self._read_buffer = ""

    # ── Send ──────────────────────────────────────────────────────────────────

    def send_command(self, action) -> bool:
        """
        Send a command string to the ESP32.

        action: one of the keys in HARDWARE_COMMANDS (e.g. "ROCK"), or any
                custom string — it will be wrapped as CMD|<action> automatically.

        Returns True if the write succeeded, False on error.
        """
        if not self.is_connected:
            return False

        # Look up the formatted command, or build one if it's a custom action.
        wire_text  = HARDWARE_COMMANDS.get(action, f"CMD|{action}")
        wire_bytes = (wire_text + "\n").encode("utf-8")

        try:
            self._serial.write(wire_bytes)
            now = time.monotonic()
            self.last_command_sent = wire_text
            self.last_command_time = now
            self.command_log.append(("TX", wire_text, now))
            print(f"[SerialBridge] TX -> {wire_text}")
            return True

        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Write error: {exc}")
            self.disconnect()  # assume the connection is dead
            return False

    # ── Receive (non-blocking) ────────────────────────────────────────────────

    def read_response(self) -> str | None:
        """
        Non-blocking read — call once per frame from the game loop.

        Data from the ESP32 arrives as a stream of bytes.  We accumulate it
        in _read_buffer and return the first complete line when a newline appears.

        Returns the first complete line received, or None if nothing is ready.
        Partial data is kept in the buffer for the next call.
        """
        if not self.is_connected:
            return None

        # Read however many bytes are waiting in the OS buffer right now.
        try:
            available = self._serial.in_waiting
            if available > 0:
                raw = self._serial.read(available)
                self._read_buffer += raw.decode("utf-8", errors="replace")
        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Read error: {exc}")
            self.disconnect()
            return None

        # Check if a complete line has arrived yet.
        if "\n" not in self._read_buffer:
            return None

        # Split off exactly the first complete line; keep the rest in the buffer.
        line, self._read_buffer = self._read_buffer.split("\n", 1)
        line = line.strip()

        if line:
            now = time.monotonic()
            self.last_response      = line
            self.last_response_time = now
            self.command_log.append(("RX", line, now))
            print(f"[SerialBridge] RX <- {line}")

        return line if line else None

    # ── Status helpers (for UI) ───────────────────────────────────────────────

    def get_status_summary(self) -> dict:
        """
        Return a dict the UI can display directly in the hardware test screen.
        """
        if not PYSERIAL_AVAILABLE:
            return {
                "pyserial_installed": False,
                "connected":          False,
                "port":               None,
                "last_tx":            None,
                "last_rx":            None,
            }

        return {
            "pyserial_installed": True,
            "connected":          self.is_connected,
            "port":               self.port_name,
            "last_tx":            self.last_command_sent,
            "last_rx":            self.last_response,
        }
