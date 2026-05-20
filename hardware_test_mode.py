"""
hardware_test_mode.py
=====================
Diagnostic screen for testing the ESP32 robot hand connection.

Supports both BLE and USB Serial — press B to switch between them at runtime.

How to reach this screen:
    Press D during gameplay → then press H

Key map:
    R / P / S   send ROCK / PAPER / SCISSORS command to the robot
    O           send OPEN  (open the hand)
    C           send CLOSE (close the fist)
    T           send PING  (just test if the connection is alive)

    [ / ]       cycle through available devices / ports
    Enter       connect to the selected device
    X           disconnect
    B           toggle between BLE and USB Serial mode

    ESC         exit hardware test and return to Diagnostic gameplay
"""

# Key codes recognised by handle_key().
KEY_ENTER = {10, 13}  # both LF and CR count as Enter
KEY_ESC   = 27


class HardwareTestController:
    """
    Manages all state for the hardware test overlay.

    Tracks which bridge (BLE or Serial) is active, which port/device is
    selected, and routes key presses to the appropriate action.
    """

    def __init__(self, bridge, ble_available=True, serial_available=True):
        """
        bridge          : the active bridge object (SerialBridge or BLEBridge)
        ble_available   : True if the bleak package is installed
        serial_available: True if the pyserial package is installed
        """
        self.bridge           = bridge
        self.ble_available    = ble_available
        self.serial_available = serial_available

        # Read the bridge type from the object itself (avoids hard-coding).
        self._mode = getattr(bridge, "bridge_type", "SERIAL")

        self.available_ports     = []    # list of port strings / device strings
        self.selected_port_index = 0     # which entry in available_ports is highlighted
        self.status_message      = "Press [ ] to select device, Enter to connect  |  B = switch BLE/Serial"
        self._is_scanning        = False  # True while a port scan is in progress

        # Do an initial scan so the list is populated on first open.
        self.refresh_ports()

    # ── Bridge switching ──────────────────────────────────────────────────────

    def switch_to_ble(self):
        """
        Replace the current bridge with a new BLEBridge instance.

        Disconnects the old bridge first, then recreates the port list.
        Does nothing (with a message) if bleak isn't installed.
        """
        if not self.ble_available:
            self.status_message = "BLE not available - run: pip install bleak"
            return
        try:
            from ble_bridge import BLEBridge
            self.bridge.disconnect()
            self.bridge              = BLEBridge()
            self._mode               = "BLE"
            self.available_ports     = []
            self.selected_port_index = 0
            self.status_message      = "Switched to BLE - scanning for devices..."
            self.refresh_ports()
        except Exception as e:
            self.status_message = f"BLE switch failed: {e}"

    def switch_to_serial(self):
        """
        Replace the current bridge with a new SerialBridge instance.

        Disconnects the old bridge first, then recreates the port list.
        Does nothing (with a message) if pyserial isn't installed.
        """
        if not self.serial_available:
            self.status_message = "Serial not available - run: pip install pyserial"
            return
        try:
            from serial_bridge import SerialBridge
            self.bridge.disconnect()
            self.bridge              = SerialBridge()
            self._mode               = "SERIAL"
            self.available_ports     = []
            self.selected_port_index = 0
            self.status_message      = "Switched to USB Serial - scanning for ports..."
            self.refresh_ports()
        except Exception as e:
            self.status_message = f"Serial switch failed: {e}"

    # ── Port / device discovery ───────────────────────────────────────────────

    def refresh_ports(self):
        """
        Ask the current bridge to list available ports or devices and update
        available_ports.  Clamps selected_port_index to a valid position if
        the list shrank.
        """
        self._is_scanning   = True
        self.status_message = "Scanning..."

        self.available_ports = self.bridge.list_ports()
        self._is_scanning    = False

        if not self.available_ports:
            # Tell the user what kind of thing we were looking for.
            mode_hint = "BLE devices" if self._mode == "BLE" else "serial ports"
            self.status_message = f"No {mode_hint} found. Press [ ] to rescan."
            return

        # Keep the index in bounds if the list got shorter.
        if self.selected_port_index >= len(self.available_ports):
            self.selected_port_index = len(self.available_ports) - 1

        self.status_message = f"Found {len(self.available_ports)} device(s). Select with [ ]"

    @property
    def selected_port(self):
        """Return the currently highlighted port string, or None if the list is empty."""
        if not self.available_ports:
            return None
        return self.available_ports[self.selected_port_index]

    # ── Key handling ──────────────────────────────────────────────────────────

    def handle_key(self, key):
        """
        Process a single keypress from the diagnostic screen.

        Returns "exit" when ESC is pressed (tells the caller to close this screen).
        Returns None for all other keys.
        """
        if key == KEY_ESC:
            return "exit"

        if key == ord("["):
            self._cycle_port(-1)          # move selection left / up
        elif key == ord("]"):
            self._cycle_port(1)           # move selection right / down
        elif key in KEY_ENTER:
            self._try_connect()           # connect to the selected port
        elif key in (ord("x"), ord("X")):
            self._disconnect()
        elif key in (ord("b"), ord("B")):
            # Toggle between BLE and Serial depending on what's active now.
            if self._mode == "BLE":
                self.switch_to_serial()
            else:
                self.switch_to_ble()
        else:
            # Any other key might map to a robot command (R, P, S, O, C, T).
            command = self._key_to_command(key)
            if command:
                self._send(command)

        return None

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self):
        """
        Called once per frame by the game loop.
        Flushes any pending data from the bridge's receive buffer so incoming
        ACK/ERR responses don't pile up unread.
        """
        self.bridge.read_response()

    # ── UI data ───────────────────────────────────────────────────────────────

    def get_display_state(self):
        """
        Return a dict of everything the renderer needs to draw the test screen.
        The renderer reads these values and doesn't touch the bridge directly.
        """
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cycle_port(self, direction):
        """
        Move the selection cursor by `direction` (+1 or -1), wrapping around.
        Refreshes the port list first in case new devices appeared.
        """
        self.refresh_ports()
        if not self.available_ports:
            self.status_message = "No devices found"
            return
        # Modulo wraps around at both ends of the list.
        self.selected_port_index = (
            (self.selected_port_index + direction) % len(self.available_ports)
        )
        self.status_message = f"Selected: {self.selected_port}"

    def _try_connect(self):
        """
        Attempt to connect to the currently selected port or BLE device.

        For BLE entries the format is "Name | Address" — we split it to extract
        the address that bleak needs.
        """
        self.refresh_ports()
        if not self.available_ports:
            self.status_message = "No devices available"
            return

        entry = self.selected_port
        if entry is None:
            return

        # BLE entries look like "RPS Robot (ESP32) | AA:BB:CC:DD:EE:FF"
        if self._mode == "BLE" and " | " in entry:
            name, address = entry.split(" | ", 1)
            self.status_message = f"Connecting to {name}..."
            ok = self.bridge.connect(address, device_name=name)
        else:
            self.status_message = f"Connecting to {entry}..."
            ok = self.bridge.connect(entry)

        self.status_message = f"Connected: {entry}" if ok else f"FAILED: {entry}"

    def _disconnect(self):
        """Disconnect from the current device and update the status message."""
        self.bridge.disconnect()
        self.status_message = "Disconnected"

    def _send(self, action):
        """
        Send a robot command over the active bridge.

        Does nothing (with a message) if we're not currently connected.
        action is a string like "ROCK" — the bridge wraps it in CMD|... format.
        """
        if not self.bridge.is_connected:
            self.status_message = "Not connected - press Enter to connect first"
            return
        ok = self.bridge.send_command(action)
        self.status_message = f"Sent: CMD|{action}" if ok else f"Send failed: {action}"

    @staticmethod
    def _key_to_command(key):
        """
        Map a key code to a robot command string.

        Both upper and lower case are supported for convenience.
        Returns None if the key doesn't correspond to any command.
        """
        mapping = {
            ord("r"): "ROCK",     ord("R"): "ROCK",
            ord("p"): "PAPER",    ord("P"): "PAPER",
            ord("s"): "SCISSORS", ord("S"): "SCISSORS",
            ord("o"): "OPEN",     ord("O"): "OPEN",
            ord("c"): "CLOSE",    ord("C"): "CLOSE",
            ord("t"): "PING",     ord("T"): "PING",
        }
        return mapping.get(key)
