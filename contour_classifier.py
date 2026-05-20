"""
contour_classifier.py
=====================
Classifies RPS gestures from the actual camera image using contour analysis.

This approach works when the palm faces the camera front-on. It bypasses
MediaPipe landmark inaccuracy by working directly with pixel data.

PIPELINE:
    1. MediaPipe landmarks  -> padded hand bounding box
    2. HSV skin segmentation within that bounding box
    3. Find the largest contour (the hand silhouette)
    4. Compute convex hull and convexity defects (V-shaped gaps between fingers)
    5. Count defects with sufficient depth and angle:
        - Rock:     0 defects  (fist — no gaps between fingers)
        - Scissors: 1 defect   (one V-gap between the two extended finger groups)
        - Paper:    3-4 defects (gaps between all four spread fingers)

This technique has been used in gesture recognition for decades and is robust
to camera angle, lighting variation, and hand size.
"""

import cv2
import numpy as np
import math


def _get_hand_bbox(landmarks, frame_h, frame_w, padding=0.25):
    """
    Compute a padded bounding box around the hand from MediaPipe landmarks.

    We add 25% padding on each side so the full hand fits even when
    landmark positions are slightly off. The landmarks don't need to be
    precise — just good enough to find the hand region.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, clipped to the frame boundary.
    """
    # Get the pixel coordinate of every landmark
    xs = [lm.x * frame_w for lm in landmarks.landmark]
    ys = [lm.y * frame_h for lm in landmarks.landmark]

    # Bounding box before padding
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Compute padding proportional to the hand's current size in frame
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * padding
    pad_y = h * padding

    # Expand the box, then clamp to the frame so we don't read outside the image
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))

    return x1, y1, x2, y2


def _create_skin_mask(roi_bgr):
    """
    Create a binary mask that marks skin-coloured pixels within the hand region.

    Uses two HSV ranges combined with OR to handle both lighter and darker
    skin tones. After thresholding, morphological closing fills small holes,
    and opening removes small noise blobs.

    Returns:
        Binary mask (uint8) the same size as roi_bgr, with 255 where skin
        was detected and 0 elsewhere.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Range 1: covers lighter / medium skin tones (hue 0-25)
    lower1 = np.array([0, 25, 50],   dtype=np.uint8)
    upper1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1  = cv2.inRange(hsv, lower1, upper1)

    # Range 2: covers reddish / darker skin tones (hue 160-180, wrapping around red)
    lower2 = np.array([160, 25, 50],   dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2  = cv2.inRange(hsv, lower2, upper2)

    # Combine both ranges — a pixel is skin if it matches either range
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup: closing fills finger-gap holes, opening removes noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return mask


def _create_landmark_mask(landmarks, frame_h, frame_w, bbox):
    """
    Fallback mask: fill the convex hull of all 21 landmark points.

    Used when skin colour detection fails (e.g. unusual lighting or skin tone
    outside the HSV ranges). It's less accurate than the skin mask but still
    gives us a rough hand outline to work with.

    Returns:
        Binary mask (uint8), same size as the ROI, with the hand region filled.
    """
    x1, y1, x2, y2 = bbox
    roi_w = x2 - x1
    roi_h = y2 - y1

    # All 21 landmark indices — we use the full set to form the boundary polygon
    boundary_ids = list(range(21))

    # Convert each landmark's normalised coordinates to ROI-relative pixel coords
    points = []
    for idx in boundary_ids:
        lm = landmarks.landmark[idx]
        # Translate to pixel space, then subtract the ROI's top-left corner
        px = int(lm.x * frame_w) - x1
        py = int(lm.y * frame_h) - y1
        # Clamp so we never write outside the ROI array
        px = max(0, min(roi_w - 1, px))
        py = max(0, min(roi_h - 1, py))
        points.append([px, py])

    # Need at least 3 points to form any polygon
    if len(points) < 3:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    # Fill the convex hull of all landmark points with white
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def _count_defects(contour, min_depth_ratio=0.15):
    """
    Count convexity defects that look like genuine finger gaps.

    A defect is a V-shaped indentation where the hand contour dips inward
    below the convex hull — this is what the gap between two fingers looks like.
    We filter out shallow noise defects using two criteria:
        1. depth  >= min_depth_ratio * hand_size  (deep enough to be a real gap)
        2. angle  < 90 degrees at the defect tip  (V-shaped, not a flat ridge)

    Parameters:
        contour         — the hand contour from OpenCV
        min_depth_ratio — minimum depth as a fraction of hand height (default 0.15)

    Returns:
        (count, defect_list) where defect_list holds dicts with geometry info
        for each significant defect (used for debug drawing).
    """
    # Can't compute defects on a tiny or empty contour
    if contour is None or len(contour) < 5:
        return 0, []

    # The hull index array is needed as input to convexityDefects
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    if hull_indices is None or len(hull_indices) < 3:
        return 0, []

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        # OpenCV can raise on degenerate contours — treat as no defects
        return 0, []

    if defects is None:
        return 0, []

    # Use the hand's bounding box to set a depth threshold scaled to hand size
    _, _, hand_w, hand_h = cv2.boundingRect(contour)
    hand_size = max(hand_w, hand_h, 1)    # avoid dividing by zero on tiny contours
    min_depth = hand_size * min_depth_ratio

    significant_defects = []

    # Iterate over every defect OpenCV found
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0   # OpenCV stores depth * 256; divide to get pixels

        # Ignore shallow defects — they are just noise on the hand outline
        if depth < min_depth:
            continue

        # The three points of the defect triangle:
        # start and end are on the hull, far is the deepest dip point
        start = tuple(contour[s][0])
        end   = tuple(contour[e][0])
        far   = tuple(contour[f][0])

        # Compute the angle at the defect tip (the 'far' point).
        # Real finger gaps form a sharp V, so the angle should be < 90 degrees.
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2   + (end[1] - far[1])**2)

        if b > 0 and c > 0:
            # Law of cosines to find the angle at 'far'
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
            cos_angle = max(-1.0, min(1.0, cos_angle))   # clamp for floating-point safety
            angle = math.degrees(math.acos(cos_angle))
        else:
            angle = 180   # degenerate — treat as not a real gap

        # Only count the defect if it has a sharp enough angle to be a finger gap
        if angle < 90:
            significant_defects.append({
                "start": start,
                "end":   end,
                "far":   far,
                "depth": depth,
                "angle": angle,
            })

    return len(significant_defects), significant_defects


def classify_contour(frame_bgr, hand_landmarks, draw_debug=False):
    """
    Classify an RPS gesture by analysing the hand contour in the camera frame.

    Parameters:
        frame_bgr:      BGR camera frame (already mirrored if needed)
        hand_landmarks: MediaPipe hand landmarks (used only for the bounding box)
        draw_debug:     if True, draws the contour, hull, and defect points on
                        the frame in place (useful for tuning and diagnostics)

    Returns:
        dict with keys:
            "gesture"           — "Rock", "Paper", "Scissors", or "Unknown"
            "command"           — corresponding command string for the game engine
            "reason"            — short description of which rule fired
            "defect_count"      — number of significant defects found
            "contour_area_ratio"— how much of the hull the hand contour fills (0-1)
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2   = _get_hand_bbox(hand_landmarks, frame_h, frame_w)

    roi_w = x2 - x1
    roi_h = y2 - y1

    # Reject bounding boxes that are too small to contain a readable hand shape
    if roi_w < 30 or roi_h < 30:
        return {
            "gesture":            "Unknown",
            "command":            "CMD_UNKNOWN",
            "reason":             "roi_too_small",
            "defect_count":       0,
            "contour_area_ratio": 0.0,
        }

    # Crop the region of interest from the full frame
    roi = frame_bgr[y1:y2, x1:x2]

    # Try skin colour segmentation first
    skin_mask   = _create_skin_mask(roi)
    skin_pixels = cv2.countNonZero(skin_mask)
    roi_pixels  = roi_w * roi_h

    # If skin detection finds less than 15% of the ROI, fall back to the
    # landmark polygon mask — the lighting is probably unusual
    if skin_pixels < roi_pixels * 0.15:
        skin_mask = _create_landmark_mask(
            hand_landmarks, frame_h, frame_w, (x1, y1, x2, y2)
        )

    # Find all contours in the mask and pick the largest one (the hand)
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "gesture":            "Unknown",
            "command":            "CMD_UNKNOWN",
            "reason":             "no_contour",
            "defect_count":       0,
            "contour_area_ratio": 0.0,
        }

    hand_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(hand_contour)

    # Reject contours that are too small to be a real hand
    if contour_area < 500:
        return {
            "gesture":            "Unknown",
            "command":            "CMD_UNKNOWN",
            "reason":             "contour_too_small",
            "defect_count":       0,
            "contour_area_ratio": 0.0,
        }

    # Compute the area ratio: contour area vs the area of its convex hull.
    # Rock is a tight fist (~0.90+), Paper is spread wide (~0.75-0.85),
    # Scissors falls between (~0.70-0.80). Used to disambiguate edge cases.
    hull_points  = cv2.convexHull(hand_contour)
    hull_area    = cv2.contourArea(hull_points)
    area_ratio   = contour_area / max(hull_area, 1)   # max(..., 1) avoids div-by-zero

    # Count the significant finger-gap defects
    defect_count, defect_list = _count_defects(hand_contour)

    # Draw debug overlays on the frame if requested
    if draw_debug:
        # Shift contour and hull points back to full-frame coordinates
        offset_contour = hand_contour.copy()
        offset_contour[:, :, 0] += x1
        offset_contour[:, :, 1] += y1
        cv2.drawContours(frame_bgr, [offset_contour], -1, (0, 255, 0), 2)   # green

        offset_hull = hull_points.copy()
        offset_hull[:, :, 0] += x1
        offset_hull[:, :, 1] += y1
        cv2.drawContours(frame_bgr, [offset_hull], -1, (255, 200, 0), 2)   # blue

        # Mark each defect's deepest point with a red dot
        for defect in defect_list:
            far_pt = (defect["far"][0] + x1, defect["far"][1] + y1)
            cv2.circle(frame_bgr, far_pt, 8, (0, 0, 255), -1)

    # =========================================================================
    # Classification rules based on defect count and area ratio
    # =========================================================================

    if defect_count == 0:
        # No gaps between fingers -> closed fist -> Rock
        gesture = "Rock"
        reason  = f"contour 0_defects ar={area_ratio:.2f}"

    elif defect_count == 1:
        # One gap -> two finger groups separated -> Scissors
        gesture = "Scissors"
        reason  = f"contour 1_defect ar={area_ratio:.2f}"

    elif defect_count >= 2:
        # 2 defects is ambiguous: could be scissors-with-thumb or an early Paper.
        # Use area ratio to decide: Paper spreads more, so its ratio is lower.
        if defect_count >= 3:
            # 3+ gaps unambiguously means multiple spread fingers -> Paper
            gesture = "Paper"
            reason  = f"contour {defect_count}_defects ar={area_ratio:.2f}"
        elif area_ratio < 0.78:
            # 2 defects but hand fills less of the hull -> fingers spread -> Paper
            gesture = "Paper"
            reason  = f"contour 2_defects_open ar={area_ratio:.2f}"
        else:
            # 2 defects and hand fills most of the hull -> fingers close -> Scissors
            gesture = "Scissors"
            reason  = f"contour 2_defects_tight ar={area_ratio:.2f}"

    else:
        # This branch is unreachable given the logic above, but kept as a safety net
        gesture = "Unknown"
        reason  = f"contour unexpected d={defect_count}"

    # Map gesture name to the command string the game engine expects
    command = {
        "Rock":     "CMD_ROCK",
        "Paper":    "CMD_PAPER",
        "Scissors": "CMD_SCISSORS",
    }.get(gesture, "CMD_UNKNOWN")

    return {
        "gesture":            gesture,
        "command":            command,
        "reason":             reason,
        "defect_count":       defect_count,
        "contour_area_ratio": round(area_ratio, 3),
    }
