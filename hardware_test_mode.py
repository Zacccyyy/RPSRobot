"""
Hardware Integration Test Mode

Diagnostic-only screen for manually testing the ESP32 serial connection.
Not exposed in the main menu — only accessible via the 'H' key while
already in Diagnostic display mode during gameplay.

Key map (while in hardware test mode):
    r  = send ROCK
    p  = send PAPER
    s  = send SCISSORS
    o  = send OPEN  (reset / open hand)
    c  = send CLOSE (close fist)
    t  = send PING  (connection test)

    [ / ]  = cycle through available serial ports
    Enter  = connect to selected port
    x      = disconnect

    Esc    = exit hardware test mode, return to Diagnostic gameplay
"""


KEY_ENTER = {10, 13}
KEY_ESC = 27


class HardwareTestController:
    """
    Manages the state needed by the hardware test overlay.

    Owns no serial connection itself — it delegates to a SerialBridge
    instance that is passed in.
    """

    def __init__(self, serial_bridge):
        self.bridge = serial_bridge

        # Port selection state
        self.available_ports = []
        self.selected_port_index = 0

        # Feedback for the UI
        self.status_message = "Press [ ] to select port, Enter to connect"

        self.refresh_ports()

    def refresh_ports(self):
        """Re-scan available serial ports."""
        self.available_ports = self.bridge.list_ports()

        if not self.available_ports:
            self.selected_port_index = 0
            return

        # Keep current index in bounds after a refresh
        if self.selected_port_index >= len(self.available_ports):
            self.selected_port_index = len(self.available_ports) - 1

    @property
    def selected_port(self):
        if not self.available_ports:
            return None
        return self.available_ports[self.selected_port_index]

    # ------------------------------------------------------------------
    # Key handling  (called from main.py)
    # ------------------------------------------------------------------

    def handle_key(self, key):
        """
        Process a key press while in hardware test mode.

        Returns:
            "exit"  — user pressed Esc, caller should leave HW test mode
            None    — handled normally, stay in HW test mode
        """
        if key == KEY_ESC:
            return "exit"

        # --- Port cycling ---
        if key == ord("["):
            self._cycle_port(-1)
            return None

        if key == ord("]"):
            self._cycle_port(1)
            return None

        # --- Connect / Disconnect ---
        if key in KEY_ENTER:
            self._try_connect()
            return None

        if key == ord("x") or key == ord("X"):
            self._disconnect()
            return None

        # --- Manual commands ---
        command = self._key_to_command(key)
        if command is not None:
            self._send(command)
            return None

        return None

    # ------------------------------------------------------------------
    # Per-frame update  (called from main.py each loop iteration)
    # ------------------------------------------------------------------

    def update(self):
        """
        Call once per frame so the bridge can drain its read buffer.
        """
        self.bridge.read_response()

    # ------------------------------------------------------------------
    # UI data
    # ------------------------------------------------------------------

    def get_display_state(self):
        """
        Returns everything the UI renderer needs to draw the
        hardware test screen.
        """
        summary = self.bridge.get_status_summary()

        return {
            "pyserial_installed": summary["pyserial_installed"],
            "connected": summary["connected"],
            "port": summary["port"],
            "last_tx": summary["last_tx"],
            "last_rx": summary["last_rx"],
            "available_ports": self.available_ports,
            "selected_port": self.selected_port,
            "selected_port_index": self.selected_port_index,
            "status_message": self.status_message,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cycle_port(self, direction):
        self.refresh_ports()

        if not self.available_ports:
            self.status_message = "No serial ports found"
            return

        self.selected_port_index = (
            (self.selected_port_index + direction) % len(self.available_ports)
        )
        self.status_message = f"Selected: {self.selected_port}"

    def _try_connect(self):
        self.refresh_ports()

        if not self.available_ports:
            self.status_message = "No serial ports available"
            return

        port = self.selected_port
        ok = self.bridge.connect(port)

        if ok:
            self.status_message = f"Connected to {port}"
        else:
            self.status_message = f"FAILED to connect to {port}"

    def _disconnect(self):
        self.bridge.disconnect()
        self.status_message = "Disconnected"

    def _send(self, action):
        if not self.bridge.is_connected:
            self.status_message = "Not connected — press Enter to connect first"
            return

        ok = self.bridge.send_command(action)

        if ok:
            self.status_message = f"Sent: CMD|{action}"
        else:
            self.status_message = f"Send failed for {action}"

    @staticmethod
    def _key_to_command(key):
        """Map a key press to a command string, or None."""
        mapping = {
            ord("r"): "ROCK",
            ord("R"): "ROCK",
            ord("p"): "PAPER",
            ord("P"): "PAPER",
            ord("s"): "SCISSORS",
            ord("S"): "SCISSORS",
            ord("o"): "OPEN",
            ord("O"): "OPEN",
            ord("c"): "CLOSE",
            ord("C"): "CLOSE",
            ord("t"): "PING",
            ord("T"): "PING",
        }
        return mapping.get(key)
