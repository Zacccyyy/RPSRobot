"""
Contour-Based Front-On Gesture Classifier

Uses the actual camera image to classify RPS gestures when the palm
faces the camera. This bypasses MediaPipe landmark inaccuracy by
working with real pixel data.

Pipeline:
    1. MediaPipe landmarks → hand bounding box (rough, good enough)
    2. Skin colour segmentation in HSV within the bounding box
    3. Find the largest contour (the hand)
    4. Compute convex hull and convexity defects
    5. Count defects with sufficient depth:
        - Rock:     0 defects (no gaps between fingers)
        - Scissors: 1 defect  (one V-gap between finger groups)
        - Paper:    3-4 defects (gaps between all fingers)

This approach has been used in gesture recognition for decades and is
robust to camera angle, lighting variation, and hand size.
"""

import cv2
import numpy as np
import math


def _get_hand_bbox(landmarks, frame_h, frame_w, padding=0.25):
    """
    Get a padded bounding box around the hand from landmarks.
    The landmarks don't need to be precise — just enough for a bbox.
    """
    xs = [lm.x * frame_w for lm in landmarks.landmark]
    ys = [lm.y * frame_h for lm in landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * padding
    pad_y = h * padding

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))

    return x1, y1, x2, y2


def _create_skin_mask(roi_bgr):
    """
    Create a binary mask of skin-coloured pixels using HSV thresholds.
    Uses two HSV ranges to handle different skin tones and lighting.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Range 1: lighter skin tones.
    lower1 = np.array([0, 25, 50], dtype=np.uint8)
    upper1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Range 2: reddish/darker skin tones.
    lower2 = np.array([160, 25, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _create_landmark_mask(landmarks, frame_h, frame_w, bbox):
    """
    Fallback: create a mask from the landmark polygon itself.
    Used when skin detection fails (e.g., unusual lighting).
    """
    x1, y1, x2, y2 = bbox
    roi_w = x2 - x1
    roi_h = y2 - y1

    # Use the outer boundary landmarks to form a hand polygon.
    # Wrist + finger tips + sides of hand.
    boundary_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20]

    points = []
    for idx in boundary_ids:
        lm = landmarks.landmark[idx]
        px = int(lm.x * frame_w) - x1
        py = int(lm.y * frame_h) - y1
        px = max(0, min(roi_w - 1, px))
        py = max(0, min(roi_h - 1, py))
        points.append([px, py])

    if len(points) < 3:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def _count_defects(contour, min_depth_ratio=0.15):
    """
    Count convexity defects with sufficient depth.

    A defect is a V-shaped indentation in the convex hull — the gap
    between two extended fingers. We filter by depth to ignore noise.

    min_depth_ratio: minimum defect depth as fraction of hand height.
    """
    if contour is None or len(contour) < 5:
        return 0, []

    hull_indices = cv2.convexHull(contour, returnPoints=False)

    if hull_indices is None or len(hull_indices) < 3:
        return 0, []

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return 0, []

    if defects is None:
        return 0, []

    # Hand size reference.
    _, _, hand_w, hand_h = cv2.boundingRect(contour)
    hand_size = max(hand_w, hand_h, 1)
    min_depth = hand_size * min_depth_ratio

    significant_defects = []

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0  # defect depth in pixels

        if depth < min_depth:
            continue

        # Get the three points of the defect triangle.
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Filter by angle: real finger gaps have angle < 90 degrees.
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        if b > 0 and c > 0:
            cos_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
        else:
            angle = 180

        if angle < 90:
            significant_defects.append({
                "start": start,
                "end": end,
                "far": far,
                "depth": depth,
                "angle": angle,
            })

    return len(significant_defects), significant_defects


def classify_contour(frame_bgr, hand_landmarks, draw_debug=False):
    """
    Classify RPS gesture using contour analysis of the actual image.

    Parameters:
        frame_bgr:      the camera frame (BGR, already mirrored)
        hand_landmarks:  MediaPipe hand landmarks (for bbox only)
        draw_debug:      if True, draw contour/hull/defects on frame

    Returns:
        {
            "gesture": "Rock" | "Paper" | "Scissors" | "Unknown",
            "command": str,
            "reason": str,
            "defect_count": int,
            "contour_area_ratio": float,
        }
    """
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _get_hand_bbox(hand_landmarks, frame_h, frame_w)

    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w < 30 or roi_h < 30:
        return {
            "gesture": "Unknown",
            "command": "CMD_UNKNOWN",
            "reason": "roi_too_small",
            "defect_count": 0,
            "contour_area_ratio": 0.0,
        }

    roi = frame_bgr[y1:y2, x1:x2]

    # Try skin detection first.
    skin_mask = _create_skin_mask(roi)
    skin_pixels = cv2.countNonZero(skin_mask)
    roi_pixels = roi_w * roi_h

    # If skin detection finds less than 15% of ROI, fall back to landmark mask.
    if skin_pixels < roi_pixels * 0.15:
        skin_mask = _create_landmark_mask(
            hand_landmarks, frame_h, frame_w, (x1, y1, x2, y2)
        )

    # Find the largest contour.
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return {
            "gesture": "Unknown",
            "command": "CMD_UNKNOWN",
            "reason": "no_contour",
            "defect_count": 0,
            "contour_area_ratio": 0.0,
        }

    hand_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(hand_contour)

    if contour_area < 500:
        return {
            "gesture": "Unknown",
            "command": "CMD_UNKNOWN",
            "reason": "contour_too_small",
            "defect_count": 0,
            "contour_area_ratio": 0.0,
        }

    # Convex hull area vs contour area ratio.
    # Rock ≈ 0.90+ (hand fills the hull), Paper ≈ 0.75-0.85, Scissors ≈ 0.70-0.80.
    hull_points = cv2.convexHull(hand_contour)
    hull_area = cv2.contourArea(hull_points)
    area_ratio = contour_area / max(hull_area, 1)

    # Count significant defects.
    defect_count, defect_list = _count_defects(hand_contour)

    # Draw debug visualisation.
    if draw_debug:
        # Draw contour in green, hull in blue.
        offset_contour = hand_contour.copy()
        offset_contour[:, :, 0] += x1
        offset_contour[:, :, 1] += y1
        cv2.drawContours(frame_bgr, [offset_contour], -1, (0, 255, 0), 2)

        offset_hull = hull_points.copy()
        offset_hull[:, :, 0] += x1
        offset_hull[:, :, 1] += y1
        cv2.drawContours(frame_bgr, [offset_hull], -1, (255, 200, 0), 2)

        # Draw defect points.
        for defect in defect_list:
            far_pt = (defect["far"][0] + x1, defect["far"][1] + y1)
            cv2.circle(frame_bgr, far_pt, 8, (0, 0, 255), -1)

    # =========================================================
    # Classification based on defect count
    # =========================================================

    if defect_count == 0:
        gesture = "Rock"
        reason = f"contour 0_defects ar={area_ratio:.2f}"
    elif defect_count == 1:
        gesture = "Scissors"
        reason = f"contour 1_defect ar={area_ratio:.2f}"
    elif defect_count >= 2:
        # 2 defects is ambiguous — could be scissors with thumb, or paper.
        # Use area ratio to disambiguate: paper has lower ratio (more gaps).
        if defect_count >= 3:
            gesture = "Paper"
            reason = f"contour {defect_count}_defects ar={area_ratio:.2f}"
        elif area_ratio < 0.78:
            gesture = "Paper"
            reason = f"contour 2_defects_open ar={area_ratio:.2f}"
        else:
            gesture = "Scissors"
            reason = f"contour 2_defects_tight ar={area_ratio:.2f}"
    else:
        gesture = "Unknown"
        reason = f"contour unexpected d={defect_count}"

    command = {
        "Rock": "CMD_ROCK",
        "Paper": "CMD_PAPER",
        "Scissors": "CMD_SCISSORS",
    }.get(gesture, "CMD_UNKNOWN")

    return {
        "gesture": gesture,
        "command": command,
        "reason": reason,
        "defect_count": defect_count,
        "contour_area_ratio": round(area_ratio, 3),
    }
