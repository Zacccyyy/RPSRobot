"""
test_camera.py
==============
Minimal standalone script to verify that the webcam is working.

Run this before the main app if you are having camera issues:
    cd ~/rps_hand_counter
    source .venv/bin/activate
    python test_camera.py

Press Q to quit.
"""

import cv2

# Open the default camera (index 0 = the first/built-in webcam).
cap = cv2.VideoCapture(0)

# Make sure the camera actually opened before trying to read from it.
if not cap.isOpened():
    print("Could not open camera.")
    raise SystemExit

# Keep reading and displaying frames until the user presses Q.
while True:
    ret, frame = cap.read()

    # ret is False if the camera disconnects or stops sending frames.
    if not ret:
        print("Could not read frame.")
        break

    # Show the live frame in a window titled "Camera Test".
    cv2.imshow("Camera Test", frame)

    # waitKey(1) waits 1 ms for a key press.
    # & 0xFF masks to the lower 8 bits -- needed on some Linux setups
    # where waitKey returns a larger integer.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up: release the camera hardware and close the display window.
cap.release()
cv2.destroyAllWindows()
