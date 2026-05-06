"""
Hardware Integration Test Mode
===============================
Diagnostic screen for testing the ESP32 connection.
Supports BOTH BLE and USB Serial - press B to switch between them at runtime.

Access: press D during gameplay -> press H

Key map:
    R / P / S   send ROCK / PAPER / SCISSORS
    O           send OPEN  (reset/open hand)
    C           send CLOSE (close fist)
    T           send PING  (connection test)

    [ / ]       cycle through available devices/ports
    Enter       connect to selected device/port
    X           disconnect
    B           toggle between BLE and USB Serial mode

    ESC         exit hardware test, return to Diagnostic gameplay
"""

KEY_ENTER = {10, 13}
KEY_ESC   = 27


class HardwareTestController:
    """
    Manages state for the hardware test overlay.
    Supports runtime switching between BLE and Serial bridges via B key.
    """

    def __init__(self, bridge, ble_available=True, serial_available=True):
        self.bridge           = bridge
        self.ble_available    = ble_available
        self.serial_available = serial_available
        self._mode            = "BLE" if hasattr(bridge, "_async_scan") else "SERIAL"

        self.available_ports     = []
        self.selected_port_index = 0
        self.status_message      = "Press [ ] to select device, Enter to connect  |  B = switch BLE/Serial"
        self._is_scanning        = False

        self.refresh_ports()

    # ── Bridge switching ──────────────────────────────────────────────

    def switch_to_ble(self):
        if not self.ble_available:
            self.status_message = "BLE not available - run: pip install bleak"
            return
        try:
            from ble_bridge import BLEBridge
            self.bridge.disconnect()
            self.bridge  = BLEBridge()
            self._mode   = "BLE"
            self.available_ports     = []
            self.selected_port_index = 0
            self.status_message = "Switched to BLE - scanning for devices..."
            self.refresh_ports()
        except Exception as e:
            self.status_message = f"BLE switch failed: {e}"

    def switch_to_serial(self):
        if not self.serial_available:
            self.status_message = "Serial not available - run: pip install pyserial"
            return
        try:
            from serial_bridge import SerialBridge
            self.bridge.disconnect()
            self.bridge  = SerialBridge()
            self._mode   = "SERIAL"
            self.available_ports     = []
            self.selected_port_index = 0
            self.status_message = "Switched to USB Serial - scanning for ports..."
            self.refresh_ports()
        except Exception as e:
            self.status_message = f"Serial switch failed: {e}"

    # ── Port/device discovery ─────────────────────────────────────────

    def refresh_ports(self):
        self._is_scanning    = True
        self.status_message  = "Scanning..."
        self.available_ports = self.bridge.list_ports()
        self._is_scanning    = False

        if not self.available_ports:
            mode_hint = "BLE devices" if self._mode == "BLE" else "serial ports"
            self.status_message = f"No {mode_hint} found. Press [ ] to rescan."
            return

        if self.selected_port_index >= len(self.available_ports):
            self.selected_port_index = len(self.available_ports) - 1

        self.status_message = f"Found {len(self.available_ports)} device(s). Select with [ ]"

    @property
    def selected_port(self):
        if not self.available_ports:
            return None
        return self.available_ports[self.selected_port_index]

    # ── Key handling ──────────────────────────────────────────────────

    def handle_key(self, key):
        if key == KEY_ESC:
            return "exit"

        if key == ord("["):
            self._cycle_port(-1)
        elif key == ord("]"):
            self._cycle_port(1)
        elif key in KEY_ENTER:
            self._try_connect()
        elif key in (ord("x"), ord("X")):
            self._disconnect()
        elif key in (ord("b"), ord("B")):
            if self._mode == "BLE":
                self.switch_to_serial()
            else:
                self.switch_to_ble()
        else:
            command = self._key_to_command(key)
            if command:
                self._send(command)

        return None

    # ── Per-frame update ──────────────────────────────────────────────

    def update(self):
        self.bridge.read_response()

    # ── UI data ───────────────────────────────────────────────────────

    def get_display_state(self):
        summary = self.bridge.get_status_summary()
        return {
            "mode":                self._mode,
            "is_ble":              self._mode == "BLE",
            "ble_available":       self.ble_available,
            "serial_available":    self.serial_available,
            "bleak_installed":     self.ble_available,
            "pyserial_installed":  self.serial_available,
            "connected":           summary["connected"],
            "port":                summary["port"],
            "last_tx":             summary["last_tx"],
            "last_rx":             summary["last_rx"],
            "available_ports":     self.available_ports,
            "selected_port":       self.selected_port,
            "selected_port_index": self.selected_port_index,
            "status_message":      self.status_message,
            "is_scanning":         self._is_scanning,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _cycle_port(self, direction):
        self.refresh_ports()
        if not self.available_ports:
            self.status_message = "No devices found"
            return
        self.selected_port_index = (
            (self.selected_port_index + direction) % len(self.available_ports)
        )
        self.status_message = f"Selected: {self.selected_port}"

    def _try_connect(self):
        self.refresh_ports()
        if not self.available_ports:
            self.status_message = "No devices available"
            return

        entry = self.selected_port
        if entry is None:
            return

        if self._mode == "BLE" and " | " in entry:
            name, address = entry.split(" | ", 1)
            self.status_message = f"Connecting to {name}..."
            ok = self.bridge.connect(address, device_name=name)
        else:
            self.status_message = f"Connecting to {entry}..."
            ok = self.bridge.connect(entry)

        self.status_message = f"Connected: {entry}" if ok else f"FAILED: {entry}"

    def _disconnect(self):
        self.bridge.disconnect()
        self.status_message = "Disconnected"

    def _send(self, action):
        if not self.bridge.is_connected:
            self.status_message = "Not connected - press Enter to connect first"
            return
        ok = self.bridge.send_command(action)
        self.status_message = f"Sent: CMD|{action}" if ok else f"Send failed: {action}"

    @staticmethod
    def _key_to_command(key):
        mapping = {
            ord("r"): "ROCK",     ord("R"): "ROCK",
            ord("p"): "PAPER",    ord("P"): "PAPER",
            ord("s"): "SCISSORS", ord("S"): "SCISSORS",
            ord("o"): "OPEN",     ord("O"): "OPEN",
            ord("c"): "CLOSE",    ord("C"): "CLOSE",
            ord("t"): "PING",     ord("T"): "PING",
        }
        return mapping.get(key)
