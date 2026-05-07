"""
core/vision_tracker.py
MediaPipe face + hand tracker.
Now delegates all lip-feature math to lip_features.py.
"""
import os
import math
import glob
import urllib.request
import urllib.error
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from core.lip_features import (
    extract_frame_features, extract_lip_contour, lip_bbox,
)

logger = logging.getLogger(__name__)

FINGERTIPS = [8, 12, 16, 20]
PIP_JOINTS = [6, 10, 14, 18]
THUMB_TIP  = 4
THUMB_IP   = 3

MODEL_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


@dataclass
class GestureState:
    finger_up: bool = False
    fist:      bool = False
    open_palm: bool = False
    thumbs_up: bool = False


@dataclass
class LipFeatures:
    valid: bool = False
    vert: float = 0.0
    horiz: float = 0.0
    feat: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    contour: Optional[np.ndarray] = None


class VisionTracker:

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._download_models()
        self._init_detectors()

    def _download_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for filename, url in MODEL_URLS.items():
            dest = os.path.join(self.model_dir, filename)
            if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
                continue
            logger.info(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
            except urllib.error.URLError as e:
                logger.error(f"Failed to download {filename}: {e}")

    def _init_detectors(self):
        face_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(self.model_dir, "face_landmarker.task")
            ),
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=os.path.join(self.model_dir, "hand_landmarker.task")
            ),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_detector = vision.FaceLandmarker.create_from_options(face_opts)
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    def process_frame(self, rgb_frame: np.ndarray):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return (
            self.face_detector.detect(mp_img),
            self.hand_detector.detect(mp_img),
        )

    def extract_lip_features(self, face_result, frame_w: int, frame_h: int) -> LipFeatures:
        feat = extract_frame_features(face_result, frame_w, frame_h)
        if feat is None:
            return LipFeatures()

        bbox = lip_bbox(face_result, frame_w, frame_h)
        contour = extract_lip_contour(face_result, frame_w, frame_h)

        if not face_result or not face_result.face_landmarks:
            return LipFeatures()

        lm = face_result.face_landmarks[0]
        top = lm[13]
        bottom = lm[14]
        left = lm[78]
        right = lm[308]

        vert = math.hypot((top.x - bottom.x) * frame_w, (top.y - bottom.y) * frame_h)
        horiz = math.hypot((left.x - right.x) * frame_w, (left.y - right.y) * frame_h)

        return LipFeatures(
            valid=True,
            vert=vert,
            horiz=horiz,
            feat=feat,
            bbox=bbox,
            contour=contour,
        )

    def get_lip_features(self, face_result, frame_w: int, frame_h: int):
        lip = self.extract_lip_features(face_result, frame_w, frame_h)
        return lip.feat, lip.bbox, lip.contour

    def get_gesture(self, hand_result) -> GestureState:
        state = GestureState()
        if not hand_result or not hand_result.hand_landmarks:
            return state
        lm = hand_result.hand_landmarks[0]
        folded = sum(
            1 for tip, pip in zip(FINGERTIPS, PIP_JOINTS)
            if lm[tip].y > lm[pip].y
        )
        index_up      = lm[8].y < lm[6].y
        others_folded = all(lm[t].y > lm[p].y for t, p in zip(FINGERTIPS[1:], PIP_JOINTS[1:]))
        state.finger_up = index_up and others_folded
        state.fist      = folded >= 4
        state.open_palm = (4 - folded) >= 4
        state.thumbs_up = (lm[THUMB_TIP].y < lm[THUMB_IP].y) and folded >= 3
        return state

    def draw_overlay(self, frame, lip, gesture=None, *,
                     mode_label, color=(0, 230, 120), top_k_labels=None):
        out = frame.copy()

        if isinstance(lip, LipFeatures):
            bbox = lip.bbox
            contour = lip.contour
            feat = lip.feat
        else:
            bbox = lip
            contour = None
            feat = None

        if contour is not None:
            pts = contour.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if feat is not None:
                cv2.putText(out,
                    f"V:{int(feat[0]*100)}% H:{int(feat[1]*100)}% AR:{int(feat[2]*100)}%",
                    (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if top_k_labels:
            label_str = "  |  ".join(f"{w} {int(c*100)}%" for w, c in top_k_labels)
            cv2.putText(out, label_str, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 80), 1, cv2.LINE_AA)

        cv2.rectangle(out, (0, 0), (340, 44), (0, 0, 0), -1)
        cv2.putText(out, mode_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
        return out

    def close(self):
        try:
            self.face_detector.close()
            self.hand_detector.close()
        except Exception:
            pass


def discover_camera(max_idx: int = 6):
    device_paths = sorted(glob.glob("/dev/video*"))
    if device_paths:
        indices = []
        for path in device_paths:
            try:
                indices.append(int(path.rsplit("video", 1)[1]))
            except (IndexError, ValueError):
                continue
    else:
        indices = list(range(max_idx))

    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return idx
        cap.release()
    return None


def open_camera(camera_idx=None, width=640, height=480):
    if camera_idx is None or camera_idx == 0:
        camera_idx = discover_camera()
    if camera_idx is None:
        return None
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap