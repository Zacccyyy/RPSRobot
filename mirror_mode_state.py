"""
mirror_mode_state.py — Live finger mirroring mode.
Extracts raw finger curl values from MediaPipe landmarks and sends
them to the ESP32 hand via BLE at 10fps.

Curl extraction uses raw landmark geometry (no ML model required):
  curl = 1.0 - clamp(tip_to_mcp_distance / hand_size, 0, 1)
  0.0 = fully open, 1.0 = fully closed
"""
import time
import math


# MediaPipe landmark indices
# Format: (tip, mcp) per finger
FINGER_LANDMARKS = [
    (4,  2),   # Thumb:  tip=4,  ip=2
    (8,  5),   # Index:  tip=8,  mcp=5
    (12, 9),   # Middle: tip=12, mcp=9
    (16, 13),  # Ring:   tip=16, mcp=13
]
WRIST_IDX      = 0
HAND_REF_MCP   = 9   # Middle finger MCP — used for hand size normalisation


def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def extract_finger_curls(landmarks):
    """
    landmarks: MediaPipe NormalizedLandmarkList OR list of 21 (x, y, z) tuples.
    Returns [thumb, index, middle, ring] curl values as ints 0-100.
    Returns None if landmarks are invalid.
    """
    # Normalise: MediaPipe returns a NormalizedLandmarkList (protobuf), not a list.
    if hasattr(landmarks, 'landmark'):
        lm = [(p.x, p.y, p.z) for p in landmarks.landmark]
    else:
        lm = landmarks

    if not lm or len(lm) < 21:
        return None

    wrist    = lm[WRIST_IDX]
    hand_ref = lm[HAND_REF_MCP]
    hand_size = _dist(wrist, hand_ref)
    if hand_size < 1e-6:
        return None

    curls = []
    for tip_idx, mcp_idx in FINGER_LANDMARKS:
        tip = lm[tip_idx]
        mcp = lm[mcp_idx]
        tip_to_mcp = _dist(tip, mcp)
        # Extended finger: tip far from mcp → low curl
        # Curled finger:   tip close to mcp → high curl
        curl = 1.0 - min(1.0, tip_to_mcp / hand_size)
        curls.append(int(curl * 100))

    return curls  # [thumb, index, middle, ring] each 0-100


class MirrorModeState:
    """
    Manages live finger mirroring.
    Call update() every frame with current hand landmarks.
    Sends CMD|MIRROR|t|i|m|r to the BLE bridge at 10fps.
    """
    TARGET_FPS    = 10
    SEND_INTERVAL = 1.0 / TARGET_FPS

    def __init__(self, ble_bridge=None):
        self.ble_bridge    = ble_bridge
        self._last_send    = 0.0
        self._last_curls   = None
        self.active        = False

    def start(self):
        self.active     = True
        self._last_send = 0.0
        print("[MirrorMode] Started")

    def stop(self):
        self.active = False
        print("[MirrorMode] Stopped")

    def update(self, landmarks):
        """
        Call every frame with MediaPipe landmarks (list of 21 (x,y,z) tuples).
        Sends BLE command at 10fps if curls have changed.
        Returns current curl values [t, i, m, r] or None.
        """
        if not self.active:
            return None

        curls = extract_finger_curls(landmarks)
        if curls is None:
            return None

        self._last_curls = curls

        now = time.monotonic()
        if now - self._last_send >= self.SEND_INTERVAL:
            self._last_send = now
            if self.ble_bridge is not None:
                t, i, m, r = curls
                cmd = f"CMD|MIRROR|{t}|{i}|{m}|{r}"
                # Use the internal queue directly — bypass GESTURE_CMD_MAP
                import asyncio
                if (self.ble_bridge._loop and self.ble_bridge._connected
                        and self.ble_bridge._send_queue is not None):
                    asyncio.run_coroutine_threadsafe(
                        self.ble_bridge._send_queue.put(cmd),
                        self.ble_bridge._loop
                    )

        return curls

    def get_curls(self):
        return self._last_curls
