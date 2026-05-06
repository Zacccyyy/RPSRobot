"""
Hardware Integration Test Mode
===============================
Diagnostic screen for testing the ESP32 connection.
Supports both BLE (ble_bridge.BLEBridge) and USB serial (serial_bridge.SerialBridge).

Access: press D during gameplay -> press H

Key map:
    R / P / S   send ROCK / PAPER / SCISSORS
    O           send OPEN  (reset/open hand)
    C           send CLOSE (close fist)
    T           send PING  (connection test)

    [ / ]       cycle through available devices/ports
    Enter       connect to selected device/port
    X           disconnect
    F5          re-scan for devices

    ESC         exit hardware test, return to Diagnostic gameplay
"""

KEY_ENTER = {10, 13}
KEY_ESC   = 27
KEY_F5    = 0xF5   # mapped in main.py as needed


class HardwareTestController:
    """
    Manages state for the hardware test overlay.
    Works with both BLEBridge and SerialBridge via duck typing.
    """

    def __init__(self, bridge):
        self.bridge = bridge

        self.available_ports    = []
        self.selected_port_index = 0
        self.status_message     = "Press [ ] to select device, Enter to connect"
        self._is_scanning       = False

        self.refresh_ports()

    # ── Port/device discovery ─────────────────────────────────────────

    def refresh_ports(self):
        """Scan for available devices/ports."""
        self._is_scanning = True
        self.status_message = "Scanning..."
        self.available_ports = self.bridge.list_ports()
        self._is_scanning = False

        if not self.available_ports:
            self.status_message = "No devices found. Press F5 to rescan."
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
        """
        Process a key press.
        Returns "exit" if ESC pressed, otherwise None.
        """
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
        elif key == KEY_F5 or key == ord("r") and not self.bridge.is_connected:
            self.refresh_ports()
        else:
            command = self._key_to_command(key)
            if command:
                self._send(command)

        return None

    # ── Per-frame update ──────────────────────────────────────────────

    def update(self):
        """Call once per frame to drain the read buffer."""
        self.bridge.read_response()

    # ── UI data ───────────────────────────────────────────────────────

    def get_display_state(self):
        """Returns everything the UI renderer needs."""
        summary = self.bridge.get_status_summary()

        # Detect bridge type
        is_ble = hasattr(self.bridge, "_async_scan")

        return {
            "is_ble":               is_ble,
            "bleak_installed":      summary.get("bleak_installed", True),
            "pyserial_installed":   summary.get("pyserial_installed", True),
            "connected":            summary["connected"],
            "port":                 summary["port"],
            "last_tx":              summary["last_tx"],
            "last_rx":              summary["last_rx"],
            "available_ports":      self.available_ports,
            "selected_port":        self.selected_port,
            "selected_port_index":  self.selected_port_index,
            "status_message":       self.status_message,
            "is_scanning":          self._is_scanning,
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

        # BLE entries look like "RPS Robot | AA:BB:CC:DD:EE:FF"
        # Serial entries are just "/dev/cu.usbserial-0001"
        is_ble = hasattr(self.bridge, "_async_scan")

        if is_ble and " | " in entry:
            name, address = entry.split(" | ", 1)
            self.status_message = f"Connecting to {name}..."
            ok = self.bridge.connect(address, device_name=name)
        else:
            self.status_message = f"Connecting to {entry}..."
            ok = self.bridge.connect(entry)

        if ok:
            self.status_message = f"Connected: {entry}"
        else:
            self.status_message = f"FAILED to connect to {entry}"

    def _disconnect(self):
        self.bridge.disconnect()
        self.status_message = "Disconnected"

    def _send(self, action):
        if not self.bridge.is_connected:
            self.status_message = "Not connected - press Enter to connect first"
            return
        ok = self.bridge.send_command(action)
        if ok:
            self.status_message = f"Sent: CMD|{action}"
        else:
            self.status_message = f"Send failed for {action}"

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
