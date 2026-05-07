"""
core/lip_features.py
Rich lip feature extraction — replaces the 2-number (vert, horiz) approach.

What we extract per frame:
  - 6 normalized scalar features  (shape descriptors)
  - 20-point inner lip contour    (for shape similarity)
  - Normalized by FACE WIDTH      (handles distance to camera)

Feature vector per frame (8 values, all in [0,1] range):
  [0] vert_ratio      vertical opening / face_width
  [1] horiz_ratio     horizontal width  / face_width
  [2] aspect_ratio    vert / horiz  (lip "roundness")
  [3] left_ratio      left corner x from centre / face_width
  [4] right_ratio     right corner x from centre / face_width
  [5] upper_curl      upper lip curvature (how much it rises)
  [6] lower_curl      lower lip curvature
  [7] area_ratio      rough lip opening area / face_width²

A "word sample" = sequence of these 8-D vectors over the utterance duration.
DTW then compares sequences instead of single peaks.
"""
import math
import numpy as np
from typing import Optional, List, Tuple

# ── MediaPipe face landmark indices ──────────────────────────────────────────
# Outer lip
OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
             291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
# Inner lip (used for area + contour)
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Key single points
LIP_TOP       = 13    # upper lip centre
LIP_BOTTOM    = 14    # lower lip centre
LIP_LEFT      = 78    # left corner
LIP_RIGHT     = 308   # right corner
LIP_TOP_L     = 82    # upper lip left
LIP_TOP_R     = 312   # upper lip right
LIP_BOT_L     = 87    # lower lip left
LIP_BOT_R     = 317   # lower lip right

# Face width reference (outer eye corners)
EYE_LEFT_OUT  = 33
EYE_RIGHT_OUT = 263

FEATURE_DIM = 8   # size of one frame's feature vector


def _pt(lm, idx, w, h):
    """Return (x_px, y_px) for landmark index."""
    return lm[idx].x * w, lm[idx].y * h


def extract_frame_features(
    face_result,
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    """
    Extract an 8-D feature vector from one frame.
    Returns None if no face detected.
    """
    if not face_result or not face_result.face_landmarks:
        return None

    lm = face_result.face_landmarks[0]

    # ── Face width for normalisation ─────────────────────────────────────────
    el_x, el_y = _pt(lm, EYE_LEFT_OUT,  frame_w, frame_h)
    er_x, er_y = _pt(lm, EYE_RIGHT_OUT, frame_w, frame_h)
    face_w = math.hypot(er_x - el_x, er_y - el_y)
    if face_w < 10:
        return None

    # ── Key lip points ────────────────────────────────────────────────────────
    tx, ty   = _pt(lm, LIP_TOP,    frame_w, frame_h)
    bx, by   = _pt(lm, LIP_BOTTOM, frame_w, frame_h)
    lx, ly   = _pt(lm, LIP_LEFT,   frame_w, frame_h)
    rx, ry   = _pt(lm, LIP_RIGHT,  frame_w, frame_h)
    tlx, tly = _pt(lm, LIP_TOP_L,  frame_w, frame_h)
    trx, try_= _pt(lm, LIP_TOP_R,  frame_w, frame_h)
    blx, bly = _pt(lm, LIP_BOT_L,  frame_w, frame_h)
    brx, bry = _pt(lm, LIP_BOT_R,  frame_w, frame_h)

    mid_x = (lx + rx) / 2
    mid_y = (lx + rx) / 2  # horizontal midline

    # ── Feature computation ───────────────────────────────────────────────────
    vert  = math.hypot(tx - bx, ty - by)
    horiz = math.hypot(lx - rx, ly - ry)

    # 0: vertical openness (normalised)
    f0 = vert / face_w

    # 1: horizontal width (normalised)
    f1 = horiz / face_w

    # 2: aspect ratio — tall narrow = "O/U", wide flat = "E/I"
    f2 = vert / (horiz + 1e-6)

    # 3: lip corners relative to face centre
    face_cx = (el_x + er_x) / 2
    f3 = abs(lx - face_cx) / face_w   # left corner pull
    f4 = abs(rx - face_cx) / face_w   # right corner pull

    # 5: upper lip curl — how much mid-upper rises vs corners
    upper_mid_y  = ty
    upper_side_y = (tly + try_) / 2
    f5 = (upper_side_y - upper_mid_y) / (face_w + 1e-6)   # +ve = M-shape

    # 6: lower lip curl
    lower_mid_y  = by
    lower_side_y = (bly + bry) / 2
    f6 = (lower_mid_y - lower_side_y) / (face_w + 1e-6)   # +ve = U-shape

    # 7: area proxy (vert × horiz ellipse area, normalised)
    f7 = (math.pi * vert * horiz / 4) / (face_w ** 2 + 1e-6)

    feat = np.array([f0, f1, f2, f3, f4, f5, f6, f7], dtype=np.float32)
    return feat


# ── Contour extraction (for visualisation) ────────────────────────────────────

def extract_lip_contour(face_result, frame_w, frame_h) -> Optional[np.ndarray]:
    """Return (N,2) array of inner lip pixel coords, or None."""
    if not face_result or not face_result.face_landmarks:
        return None
    lm = face_result.face_landmarks[0]
    pts = np.array([[lm[i].x * frame_w, lm[i].y * frame_h]
                    for i in INNER_LIP], dtype=np.float32)
    return pts


# ── Lip bounding box ──────────────────────────────────────────────────────────

def lip_bbox(face_result, frame_w, frame_h, pad=16):
    if not face_result or not face_result.face_landmarks:
        return None
    lm = face_result.face_landmarks[0]
    xs = [lm[i].x * frame_w for i in OUTER_LIP]
    ys = [lm[i].y * frame_h for i in OUTER_LIP]
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        min(frame_w, int(max(xs)) + pad),
        min(frame_h, int(max(ys)) + pad),
    )