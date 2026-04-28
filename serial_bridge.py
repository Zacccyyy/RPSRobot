"""
Serial bridge for ESP32 hardware integration.

Protocol:
    Outgoing (Python -> ESP32):  CMD|<action>\n
    Incoming (ESP32 -> Python):  ACK|<action>\n   or   ERR|<message>\n

Pipe-delimited plain text — easy to parse on both sides.

Usage:
    bridge = SerialBridge()
    ports  = bridge.list_ports()          # e.g. ["/dev/cu.usbserial-0001"]
    bridge.connect("/dev/cu.usbserial-0001")
    bridge.send_command("ROCK")           # sends  CMD|ROCK\n
    response = bridge.read_response()     # non-blocking, returns str or None
    bridge.disconnect()
"""

import time
from collections import deque

try:
    import serial
    import serial.tools.list_ports

    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False


# Commands the hardware test mode can send.
HARDWARE_COMMANDS = {
    "ROCK":  "CMD|ROCK",
    "PAPER": "CMD|PAPER",
    "SCISSORS": "CMD|SCISSORS",
    "OPEN":  "CMD|OPEN",
    "CLOSE": "CMD|CLOSE",
    "PING":  "CMD|PING",
}


class SerialBridge:
    """
    Thin wrapper around pyserial for the RPS robot.

    - Non-blocking reads (timeout=0) so the OpenCV loop never stalls.
    - Keeps a small log of recent commands and responses for the UI.
    """

    def __init__(self, baud_rate=115200, log_limit=50):
        self.baud_rate = baud_rate
        self.log_limit = log_limit

        self._serial = None
        self._port_name = None
        self._read_buffer = ""

        self.last_command_sent = None
        self.last_command_time = None
        self.last_response = None
        self.last_response_time = None
        self.command_log = deque(maxlen=log_limit)

    # ------------------------------------------------------------------
    # Port discovery
    # ------------------------------------------------------------------

    @staticmethod
    def list_ports():
        """
        Returns a list of available serial port device paths.

        On macOS these typically look like:
            /dev/cu.usbserial-0001
            /dev/cu.usbmodem14101

        If pyserial is not installed, returns an empty list.
        """
        if not PYSERIAL_AVAILABLE:
            return []

        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self):
        return self._serial is not None and self._serial.is_open

    @property
    def port_name(self):
        return self._port_name if self.is_connected else None

    def connect(self, port):
        """
        Open a serial connection to the given port.

        Returns True on success, False on failure.
        """
        if not PYSERIAL_AVAILABLE:
            print("[SerialBridge] pyserial is not installed.")
            return False

        self.disconnect()

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=0,           # non-blocking reads
                write_timeout=1.0,
            )
            self._port_name = port
            self._read_buffer = ""
            print(f"[SerialBridge] Connected to {port} @ {self.baud_rate}")
            return True

        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Failed to connect to {port}: {exc}")
            self._serial = None
            self._port_name = None
            return False

    def disconnect(self):
        """Close the serial connection if open."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

        self._serial = None
        self._port_name = None
        self._read_buffer = ""

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    def send_command(self, action):
        """
        Send a command string to the ESP32.

        action:  one of the keys in HARDWARE_COMMANDS, e.g. "ROCK"
                 or any custom string — will be wrapped as CMD|<action>.

        Returns True if sent, False on error.
        """
        if not self.is_connected:
            return False

        wire_text = HARDWARE_COMMANDS.get(action, f"CMD|{action}")
        wire_bytes = (wire_text + "\n").encode("utf-8")

        try:
            self._serial.write(wire_bytes)
            self.last_command_sent = wire_text
            self.last_command_time = time.monotonic()
            self.command_log.append(("TX", wire_text, self.last_command_time))
            print(f"[SerialBridge] TX -> {wire_text}")
            return True

        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Write error: {exc}")
            self.disconnect()
            return False

    # ------------------------------------------------------------------
    # Receive (non-blocking)
    # ------------------------------------------------------------------

    def read_response(self):
        """
        Non-blocking read.  Call once per frame.

        Returns the first complete line received, or None if nothing
        is ready yet.  Partial data is buffered internally.
        """
        if not self.is_connected:
            return None

        try:
            available = self._serial.in_waiting
            if available > 0:
                raw = self._serial.read(available)
                self._read_buffer += raw.decode("utf-8", errors="replace")
        except (serial.SerialException, OSError) as exc:
            print(f"[SerialBridge] Read error: {exc}")
            self.disconnect()
            return None

        if "\n" not in self._read_buffer:
            return None

        line, self._read_buffer = self._read_buffer.split("\n", 1)
        line = line.strip()

        if line:
            self.last_response = line
            self.last_response_time = time.monotonic()
            self.command_log.append(("RX", line, self.last_response_time))
            print(f"[SerialBridge] RX <- {line}")

        return line if line else None

    # ------------------------------------------------------------------
    # Status helpers (for UI)
    # ------------------------------------------------------------------

    def get_status_summary(self):
        """
        Returns a dict the UI can display directly.
        """
        if not PYSERIAL_AVAILABLE:
            return {
                "pyserial_installed": False,
                "connected": False,
                "port": None,
                "last_tx": None,
                "last_rx": None,
            }

        return {
            "pyserial_installed": True,
            "connected": self.is_connected,
            "port": self.port_name,
            "last_tx": self.last_command_sent,
            "last_rx": self.last_response,
        }
