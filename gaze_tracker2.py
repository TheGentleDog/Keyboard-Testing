# hi whats up
"""
Gaze Tracker  –  EyeTrax-style fullscreen edition
==================================================
•  Fullscreen 1920 × 1080 gaze canvas (pure black background)
•  Camera PiP in the bottom-right corner
•  16-point calibration (also 5, 9, 25) with animated shrink-dot
•  Linear regression gaze mapping
•  Kalman Filter  +  EMA hybrid smoother
•  Animated gaze cursor with fade trail

Requirements
------------
    pip install mediapipe opencv-python numpy scipy

Controls
--------
    Q  – quit
    R  – recalibrate
    H  – toggle camera PiP
    D  – toggle debug landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import math
import argparse
import csv
import json
from pathlib import Path

try:
    import pyautogui
    pyautogui.FAILSAFE  = False   # don't crash if cursor hits screen corner
    pyautogui.PAUSE     = 0       # no delay between calls
    _PYAUTOGUI_SCREEN_W, _PYAUTOGUI_SCREEN_H = pyautogui.size()
    _PYAUTOGUI_OK = True
except ImportError:
    _PYAUTOGUI_SCREEN_W, _PYAUTOGUI_SCREEN_H = 1920, 1080
    _PYAUTOGUI_OK = False
    print("[Warn] pyautogui not installed — mouse control disabled. pip install pyautogui")

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080
CAMERA_FRAME_W, CAMERA_FRAME_H = 1920, 1080
CAMERA_HFOV_DEG = 60.0
REAL_IPD_CM = 6.3

CALIB_25 = [
    (x, y)
    for y in [0.08, 0.29, 0.50, 0.71, 0.92]
    for x in [0.08, 0.29, 0.50, 0.71, 0.92]
]
CALIB_16 = [
    (x, y)
    for y in [0.12, 0.37, 0.63, 0.88]
    for x in [0.12, 0.37, 0.63, 0.88]
]
CALIB_9 = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]
CALIB_5 = [
    (0.50, 0.50),
    (0.10, 0.10), (0.90, 0.10),
    (0.10, 0.90), (0.90, 0.90),
]

# Colour palette (BGR)
C_BG          = (10,  10,  10)
C_DOT_OUTER   = (255, 255, 255)
C_DOT_INNER   = (0,   80,  255)
C_DOT_READY   = (0,   220, 80)
C_CURSOR      = (0,   255, 220)
C_CURSOR_RING = (0,   160, 255)
C_TRAIL_START = (0,   60,  180)
C_TRAIL_END   = (0,   240, 160)
C_TEXT        = (200, 200, 200)
C_ACCENT      = (0,   220, 120)
C_WARN        = (0,   100, 255)
C_GRID        = (28,  28,  28)


# ──────────────────────────────────────────────────────────────
#  1.  EMA Smoother
# ──────────────────────────────────────────────────────────────
class EMASmoother:
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value = None

    def update(self, m):
        m = np.asarray(m, float)
        self.value = m.copy() if self.value is None else self.alpha * m + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self): self.value = None


# ──────────────────────────────────────────────────────────────
#  2.  Kalman Filter (position + velocity, 2-D)
# ──────────────────────────────────────────────────────────────
class KalmanFilter2D:
    def __init__(self, process_noise=5e-3, measurement_noise=8.0):
        dt = 1.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4)
        self.x = np.zeros((4, 1))
        self.initialized = False

    def update(self, z):
        z = np.asarray(z, float).reshape(2, 1)
        if not self.initialized:
            self.x[:2, 0] = z.ravel(); self.initialized = True
            return z.ravel().copy()
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2, 0].copy()

    def reset(self):
        self.P = np.eye(4); self.x = np.zeros((4,1)); self.initialized = False


# ──────────────────────────────────────────────────────────────
#  3.  Hybrid Smoother (Kalman → EMA)
# ──────────────────────────────────────────────────────────────
class HybridSmoother:
    def __init__(self, pn=5e-3, mn=8.0, alpha=0.3):
        self.kalman = KalmanFilter2D(pn, mn)
        self.ema    = EMASmoother(alpha)

    def update(self, m): return self.ema.update(self.kalman.update(m))
    def reset(self): self.kalman.reset(); self.ema.reset()


# ──────────────────────────────────────────────────────────────
#  4.  Gaze Feature Extractor (with blink detection)
# ──────────────────────────────────────────────────────────────
class GazeFeatureExtractor:
    L_CORNERS = [33,  133]
    R_CORNERS = [362, 263]
    L_IRIS    = 468
    R_IRIS    = 473
    GLABELLA  = 9    # Approximate Face Mesh point at the glabella / between eyebrows

    # Eye landmarks for EAR (Eye Aspect Ratio) blink detection
    # Left eye: outer corner, upper1, upper2, inner corner, lower2, lower1
    L_EYE = [33, 160, 158, 133, 153, 144]
    # Right eye: outer corner, upper1, upper2, inner corner, lower2, lower1
    R_EYE = [362, 385, 387, 263, 380, 373]

    EAR_THRESHOLD = 0.20  # Below this = blink detected

    def _ear(self, lms, eye_indices, w, h):
        """Calculate Eye Aspect Ratio for blink detection."""
        def p(i): return np.array([lms[i].x * w, lms[i].y * h])
        try:
            pts = [p(i) for i in eye_indices]
            # EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
            vertical1 = np.linalg.norm(pts[1] - pts[5])
            vertical2 = np.linalg.norm(pts[2] - pts[4])
            horizontal = np.linalg.norm(pts[0] - pts[3])
            return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
        except:
            return 1.0  # Assume open if error

    def is_blinking(self, lms, w, h):
        """Returns True if either eye is blinking."""
        left_ear = self._ear(lms, self.L_EYE, w, h)
        right_ear = self._ear(lms, self.R_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear < self.EAR_THRESHOLD

    def extract(self, lms, w, h):
        try:
            def p(i): return np.array([lms[i].x * w, lms[i].y * h])
            def norm(iris, c0, c1):
                ctr = (p(c0) + p(c1)) / 2.0
                return (iris - ctr) / (np.linalg.norm(p(c1)-p(c0)) / 2.0 + 1e-6)
            nl = norm(p(self.L_IRIS), *self.L_CORNERS)
            nr = norm(p(self.R_IRIS), *self.R_CORNERS)
            return np.array([nl[0], nl[1], nr[0], nr[1]], float)
        except Exception:
            return None

    def iris_px(self, lms, w, h):
        def p(i): return (int(lms[i].x*w), int(lms[i].y*h))
        return p(self.L_IRIS), p(self.R_IRIS)

    def glabella_px(self, lms, w, h):
        return np.array([lms[self.GLABELLA].x * w, lms[self.GLABELLA].y * h], float)

    def interocular_px(self, lms, w, h):
        def p(i): return np.array([lms[i].x * w, lms[i].y * h], float)
        left = (p(self.L_CORNERS[0]) + p(self.L_CORNERS[1])) / 2.0
        right = (p(self.R_CORNERS[0]) + p(self.R_CORNERS[1])) / 2.0
        return float(np.linalg.norm(right - left))


# ──────────────────────────────────────────────────────────────
#  5.  Linear Regression Gaze Model
# ──────────────────────────────────────────────────────────────
class GazeRegressionModel:
    def __init__(self): self.Wx = self.Wy = None; self.fitted = False

    def _design(self, F):
        N = F.shape[0]
        return np.hstack([np.ones((N, 1)), F])

    def fit(self, F, T):
        A = self._design(F)
        self.Wx,*_ = np.linalg.lstsq(A, T[:,0], rcond=None)
        self.Wy,*_ = np.linalg.lstsq(A, T[:,1], rcond=None)
        self.fitted = True

    def predict(self, f):
        if not self.fitted: return np.array([0.5, 0.5])
        a = self._design(f.reshape(1,-1))
        return np.clip([float(a@self.Wx), float(a@self.Wy)], 0, 1)


# ──────────────────────────────────────────────────────────────
#  6.  Calibration Manager
# ──────────────────────────────────────────────────────────────
class CalibrationManager:
    HOLD   = 20   # stabilisation frames before collecting
    SHRINK = 25   # collection animation frames

    def __init__(self, num_points=16, spp=60):
        if num_points == 25:
            self.pts = CALIB_25
        elif num_points == 16:
            self.pts = CALIB_16
        elif num_points == 9:
            self.pts = CALIB_9
        else:
            self.pts = CALIB_5
        self.n     = len(self.pts)
        self.spp   = spp
        self.idx   = 0
        self.buf   = []
        self.feats = []
        self.tgts  = []
        self.done  = False
        self.model = GazeRegressionModel()
        self._fp   = 0   # frames on current point

    @property
    def target_px(self):
        fx, fy = self.pts[self.idx]
        return int(fx * SCREEN_W), int(fy * SCREEN_H)

    def add_sample(self, feat):
        if self.done: return
        self._fp += 1
        if self._fp <= self.HOLD: return
        tx, ty = self.pts[self.idx]
        self.buf.append(feat); self.feats.append(feat); self.tgts.append([tx, ty])
        if len(self.buf) >= self.spp:
            self.buf = []; self._fp = 0; self.idx += 1
            if self.idx >= self.n:
                self.model.fit(np.array(self.feats), np.array(self.tgts))
                print(f"[Calibration] Complete — {len(self.feats)} samples over {self.n} points.")
                self.done = True

    def progress(self):
        done = self.idx * self.spp + max(0, self._fp - self.HOLD)
        return min(done / (self.n * self.spp), 1.0)

    def dot_state(self):
        if self._fp <= self.HOLD:
            return 'hold', self._fp / self.HOLD
        return 'collect', min(self._fp - self.HOLD, self.spp) / self.spp


# ──────────────────────────────────────────────────────────────
#  Drawing helpers
# ──────────────────────────────────────────────────────────────
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))

def draw_grid(canvas):
    for c in range(1, 6):
        x = int(c * SCREEN_W / 6)
        cv2.line(canvas, (x,0), (x,SCREEN_H), C_GRID, 1)
    for r in range(1, 4):
        y = int(r * SCREEN_H / 4)
        cv2.line(canvas, (0,y), (SCREEN_W,y), C_GRID, 1)

def draw_calib_dot(canvas, cx, cy, phase, t, R=32):
    now = time.time()
    if phase == 'hold':
        pulse = 0.5 + 0.5 * math.sin(now * 6)
        rr = int(R + 6*pulse)
        cv2.circle(canvas, (cx,cy), rr,  lerp_color((60,60,60), C_DOT_OUTER, pulse), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx,cy), R,   C_DOT_OUTER, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx,cy), 10,  C_DOT_INNER, -1, cv2.LINE_AA)
    else:
        inner = max(2, int(R*(1-t)))
        cv2.circle(canvas, (cx,cy), R, C_DOT_READY, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx,cy), inner, C_DOT_READY, -1, cv2.LINE_AA)
        angle = int(360*t)
        if angle > 0:
            cv2.ellipse(canvas, (cx,cy), (R+8,R+8), -90, 0, angle, C_ACCENT, 3, cv2.LINE_AA)

def draw_crosshair(canvas, cx, cy, size=14, color=(55,55,55)):
    cv2.line(canvas, (cx-size,cy), (cx+size,cy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx,cy-size), (cx,cy+size), color, 1, cv2.LINE_AA)

def draw_gaze_cursor(canvas, gx, gy, history):
    n = len(history)
    for i,(px,py) in enumerate(history):
        t = i / max(n-1, 1)
        col = lerp_color(C_TRAIL_START, C_TRAIL_END, t)
        cv2.circle(canvas, (px,py), max(2,int(2+4*t)), col, -1, cv2.LINE_AA)
    pulse = 0.5 + 0.5 * math.sin(time.time()*8)
    rr = int(18 + 4*pulse)
    cv2.circle(canvas, (gx,gy), rr, lerp_color(C_CURSOR_RING, C_CURSOR, pulse), 2, cv2.LINE_AA)
    cv2.circle(canvas, (gx,gy), 6, C_CURSOR, -1, cv2.LINE_AA)
    cv2.circle(canvas, (gx,gy), 2, (255,255,255), -1, cv2.LINE_AA)

def txt(canvas, text, xy, scale=0.6, color=C_TEXT, thick=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def focal_px_for_width(width):
    return width / (2.0 * math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0))

def configure_1080p_camera(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

def draw_bar(canvas, prog, x, y, w, h):
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (45,45,45), -1)
    fill = int(w * prog)
    if fill > 0:
        cv2.rectangle(canvas, (x,y), (x+fill,y+h), lerp_color((0,120,60), C_ACCENT, prog), -1)
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (110,110,110), 1)

def overlay_pip(canvas, cam_frame, pip_w=320, pip_h=240, distance_info=None):
    if cam_frame is None: return
    x0, y0 = SCREEN_W - pip_w - 20, SCREEN_H - pip_h - 20
    if distance_info:
        panel_w = 260
        px0 = max(20, x0 - panel_w - 12)
        cv2.rectangle(canvas, (px0, y0), (px0 + panel_w, y0 + pip_h), (18,18,18), -1)
        cv2.rectangle(canvas, (px0, y0), (px0 + panel_w, y0 + pip_h), (55,55,55), 1)
        txt(canvas, "Distance 1080p", (px0 + 14, y0 + 30), scale=0.62, color=C_ACCENT, thick=1)
        txt(canvas, distance_info["label"], (px0 + 14, y0 + 78), scale=0.80, color=C_TEXT, thick=2)
        txt(canvas, f"IPD px: {distance_info['ipd_px']:.1f}", (px0 + 14, y0 + 122),
            scale=0.48, color=(160,160,160))
        txt(canvas, f"Pos px: {distance_info['pos_px']}", (px0 + 14, y0 + 150),
            scale=0.48, color=(160,160,160))
        txt(canvas, f"Pos norm: {distance_info['pos_norm']}", (px0 + 14, y0 + 178),
            scale=0.48, color=(160,160,160))
        txt(canvas, f"Angle: {distance_info['angle_deg']}", (px0 + 14, y0 + 206),
            scale=0.48, color=(160,160,160))
        txt(canvas, f"Frame: {distance_info['frame']}", (px0 + 14, y0 + 226),
            scale=0.40, color=(120,120,120))
        txt(canvas, "Estimated from eye spacing", (px0 + 14, y0 + pip_h - 18),
            scale=0.40, color=(120,120,120))
    cv2.rectangle(canvas, (x0-3,y0-3), (x0+pip_w+3,y0+pip_h+3), (55,55,55), 2)
    canvas[y0:y0+pip_h, x0:x0+pip_w] = cv2.resize(cam_frame, (pip_w,pip_h))
    txt(canvas, "Camera", (x0+4, y0+pip_h-8), scale=0.40, color=(140,140,140))


# ──────────────────────────────────────────────────────────────
#  7.  Main Application
# ──────────────────────────────────────────────────────────────
class GazeTrackerApp:
    WIN = "GazeTracker"

    def __init__(self, camera_id=0, num_points=16, ema_alpha=0.3,
                pnoise=5e-3, mnoise=8.0, spp=60,
                show_camera_window=True, debug_landmarks=True,
                show_distance=True, tutorial_enabled=True):
        self.cam_id     = camera_id
        self.num_points = num_points
        self.spp        = spp

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.extractor = GazeFeatureExtractor()
        self.smoother  = HybridSmoother(pnoise, mnoise, ema_alpha)
        self.calib     = None

        self.hist   = collections.deque(maxlen=50)
        self._fpsq  = collections.deque(maxlen=30)
        self._pip          = show_camera_window
        self._dbg          = debug_landmarks
        self._show_distance = show_distance
        self._tutorial_enabled = tutorial_enabled
        self._last_distance = None
        self._head_ref = None
        self._head_ref_samples = []
        self._head_scale_ref = None
        self._head_scale_samples = []
        self._head_shift = None
        self._head_shift_threshold = 0.2
        self._head_distance_threshold = 0.08
        self._mouse_ctrl   = True
        self._window_open  = True
        self._blink = False  # blink state indicator
        self._last_gaze = (SCREEN_W // 2, SCREEN_H // 2)  # last known good gaze
        self._tracking_frames = 0
        self._tracking_faces = 0
        self._tracking_predictions = 0
        self._mouse_moves = 0
        self._tracking_error = None
        self.pyautogui_ok = _PYAUTOGUI_OK
        self.pyautogui_screen_size = (_PYAUTOGUI_SCREEN_W, _PYAUTOGUI_SCREEN_H)
        self.canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        self._heatmap_recording = False
        self._heatmap_points = []
        self._heatmap_start_time = None
        self._heatmap_end_time = None

    def _new_calib(self):
        self.calib = CalibrationManager(self.num_points, self.spp)
        self.smoother.reset(); self.hist.clear()
        self._head_ref = None
        self._head_ref_samples = []
        self._head_scale_ref = None
        self._head_scale_samples = []
        self._head_shift = None

    def _fps(self):
        now = time.time(); self._fpsq.append(now)
        return (len(self._fpsq)-1)/(self._fpsq[-1]-self._fpsq[0]+1e-9) if len(self._fpsq)>1 else 0

    def _show_tutorial_text(self, lines, seconds=10, window_ready_callback=None):
        if isinstance(lines, str):
            lines = [lines]
        start = time.time()
        window_ready_notified = False
        while True:
            elapsed = time.time() - start
            if elapsed >= seconds:
                break

            cv = self.canvas
            cv[:] = (18, 18, 18)
            remaining = max(0, int(math.ceil(seconds - elapsed)))
            scale = 1.7
            thick = 3
            line_gap = 76
            sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for line in lines]
            total_h = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1)
            y = (SCREEN_H - total_h) // 2
            for line, size in zip(lines, sizes):
                x = (SCREEN_W - size[0]) // 2
                y += size[1]
                txt(cv, line, (x, y), scale=scale, color=(235, 235, 235), thick=thick)
                y += line_gap

            counter = f"Starting in {remaining}"
            csize = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            txt(cv, counter, ((SCREEN_W - csize[0]) // 2, y + 28),
                scale=0.8, color=(150, 150, 150), thick=2)

            cv2.imshow(self.WIN, cv)
            if not window_ready_notified:
                window_ready_notified = True
                if window_ready_callback:
                    try:
                        window_ready_callback()
                    except Exception:
                        pass
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        return True

    def _estimate_distance(self, lms, w, h):
        if lms is None:
            return self._last_distance
        try:
            left, right = self.extractor.iris_px(lms, w, h)
            left_arr = np.array(left, float)
            right_arr = np.array(right, float)
            glabella = self.extractor.glabella_px(lms, w, h)
            pos_x = glabella[0] - (w / 2.0)
            pos_y = (h / 2.0) - glabella[1]
            pos_x_norm = pos_x / (w / 2.0)
            pos_y_norm = pos_y / (h / 2.0)
            angle_x = math.degrees(math.atan(pos_x / focal_px_for_width(w)))
            angle_y = math.degrees(math.atan(pos_y_norm))
            ipd_px = float(np.linalg.norm(left_arr - right_arr))
            if ipd_px < 1.0:
                return self._last_distance
            focal_px = focal_px_for_width(w)
            distance_cm = (REAL_IPD_CM * focal_px) / ipd_px
            self._last_distance = {
                "cm": distance_cm,
                "label": f"{distance_cm:.1f} cm",
                "ipd_px": ipd_px,
                "focal_px": focal_px,
                "frame": f"{w}x{h}",
                "pos_px": f"{pos_x:+.0f}, {pos_y:+.0f}",
                "pos_norm": f"{pos_x_norm:+.2f}, {pos_y_norm:+.2f}",
                "angle_deg": f"{angle_x:+.1f}, {angle_y:+.1f}",
            }
        except Exception:
            pass
        return self._last_distance

    def _head_position_feature(self, lms, w, h):
        if lms is None:
            return None
        try:
            glabella = self.extractor.glabella_px(lms, w, h)
            scale = self.extractor.interocular_px(lms, w, h)
            if scale < 1.0:
                return None
            return np.array([glabella[0] / scale, glabella[1] / scale], float)
        except Exception:
            return None

    def _head_scale_feature(self, lms, w, h):
        if lms is None:
            return None
        try:
            scale = self.extractor.interocular_px(lms, w, h)
            return float(scale) if scale >= 1.0 else None
        except Exception:
            return None

    def _record_calibration_head_position(self, lms, w, h):
        feat = self._head_position_feature(lms, w, h)
        if feat is not None:
            self._head_ref_samples.append(feat)
        scale = self._head_scale_feature(lms, w, h)
        if scale is not None:
            self._head_scale_samples.append(scale)

    def _mark_initial_head_position(self, lms, w, h):
        if self._head_ref is not None:
            return
        feat = self._head_position_feature(lms, w, h)
        if feat is not None:
            self._head_ref = feat.copy()
            self._head_scale_ref = self._head_scale_feature(lms, w, h)
            print("[Head] Initial calibration head position marked.")

    def _finalize_calibration_head_position(self):
        if self._head_ref_samples:
            self._head_ref = np.mean(np.array(self._head_ref_samples), axis=0)
            print(f"[Head] Calibrated glabella reference from {len(self._head_ref_samples)} samples.")
        if self._head_scale_samples:
            self._head_scale_ref = float(np.mean(np.array(self._head_scale_samples)))

    def _update_head_shift(self, lms, w, h):
        feat = self._head_position_feature(lms, w, h)
        if feat is None or self._head_ref is None:
            self._head_shift = None
            return None
        delta = feat - self._head_ref
        mag = float(np.linalg.norm(delta))
        scale = self._head_scale_feature(lms, w, h)
        distance_delta = 0.0
        if scale is not None and self._head_scale_ref is not None and scale > 0:
            distance_delta = (self._head_scale_ref / scale) - 1.0
        guide_parts = []
        if delta[0] > 0.03:
            guide_parts.append("left")
        elif delta[0] < -0.03:
            guide_parts.append("right")
        if delta[1] > 0.03:
            guide_parts.append("up")
        elif delta[1] < -0.03:
            guide_parts.append("down")
        if distance_delta > self._head_distance_threshold:
            guide_parts.append("closer")
        elif distance_delta < -self._head_distance_threshold:
            guide_parts.append("further")
        if len(guide_parts) > 2:
            guide_text = ", ".join(guide_parts[:-1]) + ", and " + guide_parts[-1]
        else:
            guide_text = " and ".join(guide_parts)
        guide = "Shift " + guide_text if guide_text else "Hold steady"
        self._head_shift = {
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "mag": mag,
            "distance": float(distance_delta),
            "moved": mag >= self._head_shift_threshold or abs(distance_delta) >= self._head_distance_threshold,
            "guide": guide,
            "position": f"x {delta[0]:+.2f}, y {delta[1]:+.2f}",
            "distance_label": f"{distance_delta:+.0%}",
            "target": "x 0.00, y 0.00",
        }
        return self._head_shift

    # ── calibration render ──────────────────────────────────────
    def _render_calib(self, cam, feat, lms=None):
        cv = self.canvas; cv[:] = C_BG
        draw_grid(cv)

        # Skip blink frames during calibration
        is_blinking = False
        head_moved = False
        if lms is not None:
            is_blinking = self.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])
            if not is_blinking:
                self._mark_initial_head_position(lms, cam.shape[1], cam.shape[0])
                self._update_head_shift(lms, cam.shape[1], cam.shape[0])
                head_moved = bool(self._head_shift and self._head_shift["moved"])
        else:
            self._head_shift = None

        for i,(fx,fy) in enumerate(self.calib.pts):
            px,py = int(fx*SCREEN_W), int(fy*SCREEN_H)
            if i < self.calib.idx:
                # completed tick
                cv2.circle(cv,(px,py),12,C_ACCENT,2,cv2.LINE_AA)
                cv2.line(cv,(px-5,py),(px-1,py+5),C_ACCENT,2,cv2.LINE_AA)
                cv2.line(cv,(px-1,py+5),(px+6,py-4),C_ACCENT,2,cv2.LINE_AA)
            elif i > self.calib.idx:
                draw_crosshair(cv,px,py)

        cx,cy = self.calib.target_px
        phase,t = self.calib.dot_state()
        draw_calib_dot(cv, cx, cy, phase, t)

        # progress bar centred at bottom
        prog = self.calib.progress()
        bx = SCREEN_W//2 - 300; by = SCREEN_H - 60
        draw_bar(cv, prog, bx, by, 600, 16)

        label = f"Calibration  ·  Point {self.calib.idx+1} / {self.calib.n}   ({int(prog*100)}%)"
        lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        txt(cv, label, (SCREEN_W//2 - lw//2, by-12), scale=0.65, color=C_TEXT)
        mouse_state = "ON" if self._mouse_ctrl else "OFF"
        hint = f"Keep eyes on dot  ·  R=recalibrate  H=pip  D=debug  X=mouse({mouse_state})  Q=quit"
        hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        txt(cv, hint, (SCREEN_W//2 - hw//2, SCREEN_H-18), scale=0.48, color=(100,100,100))

        if self._head_shift and self._head_shift["moved"]:
            msg = self._head_shift.get("guide", "Shift back to center")
            detail = (
                f"Position {self._head_shift['position']}  -> target {self._head_shift['target']}  |  "
                f"Distance {self._head_shift['distance_label']} -> target 0%"
            )
            mw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0][0]
            dw = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            box_w = max(mw, dw) + 48
            x0 = SCREEN_W // 2 - box_w // 2
            y0 = 88
            cv2.rectangle(cv, (x0, y0), (x0 + box_w, y0 + 72), (24, 24, 24), -1)
            cv2.rectangle(cv, (x0, y0), (x0 + box_w, y0 + 72), C_WARN, 2)
            txt(cv, msg, (SCREEN_W//2 - mw//2, y0 + 30), scale=0.82, color=C_WARN, thick=2)
            txt(cv, detail, (SCREEN_W//2 - dw//2, y0 + 56), scale=0.55, color=C_TEXT)

        if self._pip:
            distance_info = self._estimate_distance(lms, cam.shape[1], cam.shape[0]) if self._show_distance else None
            overlay_pip(cv, cam, distance_info=distance_info)
        # Only add calibration samples when the face is usable and near the starting head position.
        if feat is not None and not is_blinking and not head_moved:
            self._record_calibration_head_position(lms, cam.shape[1], cam.shape[0])
            self.calib.add_sample(feat)

    # ── tracking render ─────────────────────────────────────────
    def _render_track(self, cam, feat, lms, fps):
        cv = self.canvas; cv[:] = C_BG
        draw_grid(cv)

        # Check for blink - if blinking, skip gaze update and use last position
        if lms is not None:
            self._blink = self.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])
            self._update_head_shift(lms, cam.shape[1], cam.shape[0])
        else:
            self._blink = False
            self._head_shift = None

        if feat is not None and not self._blink:
            raw = self.calib.model.predict(feat)
            smo = self.smoother.update(raw)
            gx  = int(np.clip(smo[0]*SCREEN_W,  0, SCREEN_W-1))
            gy  = int(np.clip(smo[1]*SCREEN_H, 0, SCREEN_H-1))
            self._last_gaze = (gx, gy)
            self._tracking_predictions += 1
            self.hist.append((gx, gy))
            if self._heatmap_recording:
                self._heatmap_points.append((gx, gy, time.time()))
            if self._mouse_ctrl and _PYAUTOGUI_OK:
                mx = int(gx * _PYAUTOGUI_SCREEN_W / SCREEN_W)
                my = int(gy * _PYAUTOGUI_SCREEN_H / SCREEN_H)
                pyautogui.moveTo(mx, my)
                self._mouse_moves += 1

        # Always draw cursor at last known position
        gx, gy = self._last_gaze
        if feat is not None or self._blink:
            draw_gaze_cursor(cv, gx, gy, self.hist)
            # coordinate badge
            badge = f"  {gx} x {gy}  "
            if self._blink:
                badge = f"  {gx} x {gy}  [BLINK]"
            bw = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            bx = min(gx+22, SCREEN_W-bw-10); by = max(gy-14, 20)
            cv2.rectangle(cv,(bx-4,by-18),(bx+bw+4,by+5),(22,22,22),-1)
            txt(cv, badge, (bx,by), scale=0.55, color=C_WARN if self._blink else C_ACCENT)
        else:
            msg = "No face detected — move into frame"
            mw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
            txt(cv, msg, (SCREEN_W//2-mw//2, SCREEN_H//2), scale=1.0, color=C_WARN, thick=2)

        # top HUD bar
        cv2.rectangle(cv,(0,0),(SCREEN_W,38),(16,16,16),-1)
        txt(cv, "GazeTracker  1920x1080", (14,26), scale=0.62, color=C_ACCENT, thick=1)
        txt(cv, f"FPS {fps:.1f}", (SCREEN_W-110,26), scale=0.60, color=(150,255,150))
        mouse_state = "ON" if self._mouse_ctrl else "OFF"
        ctrl = f"R=recalibrate   H=pip   D=debug   X=mouse({mouse_state})   Q=quit"
        cw = cv2.getTextSize(ctrl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        txt(cv, ctrl, (SCREEN_W//2-cw//2, 26), scale=0.48, color=(110,110,110))

        if self._head_shift and self._head_shift["moved"]:
            msg = self._head_shift.get("guide", "Shift back to center")
            detail = (
                f"Position {self._head_shift['position']}  -> target {self._head_shift['target']}  |  "
                f"Distance {self._head_shift['distance_label']} -> target 0%"
            )
            mw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)[0][0]
            dw = cv2.getTextSize(detail, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            box_w = max(mw, dw) + 48
            box_h = 72
            margin = 24
            x0 = margin
            y0 = SCREEN_H - box_h - margin
            cv2.rectangle(cv, (x0, y0), (x0 + box_w, y0 + box_h), (24, 24, 24), -1)
            cv2.rectangle(cv, (x0, y0), (x0 + box_w, y0 + box_h), C_WARN, 2)
            txt(cv, msg, (x0 + box_w//2 - mw//2, y0 + 30), scale=0.82, color=C_WARN, thick=2)
            txt(cv, detail, (x0 + box_w//2 - dw//2, y0 + 56), scale=0.55, color=C_TEXT)

        if self._pip:
            distance_info = self._estimate_distance(lms, cam.shape[1], cam.shape[0]) if self._show_distance else None
            overlay_pip(cv, cam, distance_info=distance_info)

    # ── debug overlay on camera pip ─────────────────────────────
    def _debug_cam(self, cam, lms):
        if not self._dbg or lms is None: return
        h,w = cam.shape[:2]
        for pt in self.extractor.iris_px(lms,w,h):
            cv2.circle(cam, pt, 3, (0,220,255), -1)
        g = self.extractor.glabella_px(lms, w, h)
        gx, gy = int(g[0]), int(g[1])
        cv2.circle(cam, (gx, gy), 10, (255, 0, 255), 2)
        cv2.line(cam, (gx - 14, gy), (gx + 14, gy), (255, 0, 255), 2)
        cv2.line(cam, (gx, gy - 14), (gx, gy + 14), (255, 0, 255), 2)
        cv2.putText(cam, "GLABELLA", (gx + 14, gy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
        for idx in GazeFeatureExtractor.L_CORNERS + GazeFeatureExtractor.R_CORNERS:
            cv2.circle(cam,(int(lms[idx].x*w),int(lms[idx].y*h)),3,(255,100,0),-1)

    def _handle_calib_key(self, key):
        if key == 255:
            return None
        ch = chr(key).lower()
        if ch == 'q':
            return "quit"
        if ch == 'r':
            self._new_calib()
            print("[Info] Calibration restarted.")
            return "restart"
        if ch == 'h':
            self._pip = not self._pip
            print(f"[Info] Camera PiP {'on' if self._pip else 'off'}.")
            return "handled"
        if ch == 'd':
            self._dbg = not self._dbg
            print(f"[Info] Debug landmarks {'on' if self._dbg else 'off'}.")
            return "handled"
        if ch == 'x':
            self._mouse_ctrl = not self._mouse_ctrl
            print(f"[Info] Mouse control {'on' if self._mouse_ctrl else 'off'} after calibration.")
            return "handled"
        return None

    def start_heatmap_recording(self):
        self._heatmap_points = []
        self._heatmap_start_time = time.time()
        self._heatmap_end_time = None
        self._heatmap_recording = True
        print("[Info] Heatmap recording started.")

    def stop_heatmap_recording(self):
        self._heatmap_recording = False
        self._heatmap_end_time = time.time()
        print(f"[Info] Heatmap recording stopped. {len(self._heatmap_points)} gaze points captured.")

    def heatmap_stats(self):
        end_time = self._heatmap_end_time or time.time()
        start_time = self._heatmap_start_time or end_time
        jitter = self._heatmap_jitter_metrics()
        return {
            "point_count": len(self._heatmap_points),
            "duration_seconds": int(max(0, round(end_time - start_time))),
            "mean_step_px": jitter["mean_step_px"],
            "mean_spread_px": jitter["mean_spread_px"],
            "max_spread_px": jitter["max_spread_px"],
        }

    def _heatmap_jitter_metrics(self):
        points = list(self._heatmap_points)
        if not points:
            return {
                "point_count": 0,
                "mean_gaze_x": None,
                "mean_gaze_y": None,
                "mean_step_px": 0.0,
                "median_step_px": 0.0,
                "std_step_px": 0.0,
                "min_step_px": 0.0,
                "max_step_px": 0.0,
                "p95_step_px": 0.0,
                "mean_spread_px": 0.0,
                "median_spread_px": 0.0,
                "std_spread_px": 0.0,
                "max_spread_px": 0.0,
                "p95_spread_px": 0.0,
                "mean_sample_interval_ms": 0.0,
            }

        xy = np.array([(point[0], point[1]) for point in points], dtype=float)
        times = np.array([point[2] for point in points], dtype=float)
        mean_xy = xy.mean(axis=0)
        spread = np.linalg.norm(xy - mean_xy, axis=1)

        if len(xy) > 1:
            steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            intervals_ms = np.diff(times) * 1000.0
        else:
            steps = np.array([], dtype=float)
            intervals_ms = np.array([], dtype=float)

        def metric(values, fn, default=0.0):
            return float(fn(values)) if len(values) else default

        return {
            "point_count": len(points),
            "mean_gaze_x": float(mean_xy[0]),
            "mean_gaze_y": float(mean_xy[1]),
            "mean_step_px": metric(steps, np.mean),
            "median_step_px": metric(steps, np.median),
            "std_step_px": metric(steps, np.std),
            "min_step_px": metric(steps, np.min),
            "max_step_px": metric(steps, np.max),
            "p95_step_px": metric(steps, lambda values: np.percentile(values, 95)),
            "mean_spread_px": metric(spread, np.mean),
            "median_spread_px": metric(spread, np.median),
            "std_spread_px": metric(spread, np.std),
            "max_spread_px": metric(spread, np.max),
            "p95_spread_px": metric(spread, lambda values: np.percentile(values, 95)),
            "mean_sample_interval_ms": metric(intervals_ms, np.mean),
        }

    def save_heatmap_png(self, output_path, background_image=None, bins=(40, 20), sigma=1.1):
        """
        Save a keyboard-style gaze heatmap and matching density value files.

        The PNG uses a 40 x 20 coordinate grid like the reference image. The CSV
        and JSON beside it contain normalized density values from 0.0 to 1.0.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        points = list(self._heatmap_points)
        jitter = self._heatmap_jitter_metrics()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:
            raise RuntimeError(f"Heatmap output needs matplotlib and scipy: {exc}") from exc

        cols, rows = bins
        if points:
            xs = np.array([point[0] for point in points], dtype=float)
            ys = np.array([point[1] for point in points], dtype=float)
            heatmap, _, _ = np.histogram2d(
                ys,
                xs,
                bins=[rows, cols],
                range=[[0, SCREEN_H], [0, SCREEN_W]],
            )
            density = gaussian_filter(heatmap, sigma=sigma)
            max_density = float(density.max())
            if max_density > 0:
                density = density / max_density
        else:
            density = np.zeros((rows, cols), dtype=float)

        fig, ax = plt.subplots(figsize=(10.8, 6.2))

        if background_image is not None:
            bg = background_image.resize((SCREEN_W, SCREEN_H))
            ax.imshow(bg, extent=[0, cols, rows, 0], alpha=0.32)
        else:
            ax.imshow(
                np.full((rows, cols), 0.93),
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=[0, cols, rows, 0],
                alpha=1.0,
            )

        heat = ax.imshow(
            density,
            cmap="jet",
            interpolation="bilinear",
            extent=[0, cols, rows, 0],
            vmin=0,
            vmax=1,
            alpha=np.clip(density * 0.95, 0, 0.95),
        )

        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_xticks(np.arange(0, cols + 1, 5))
        ax.set_yticks(np.arange(0, rows + 1, 5))
        ax.grid(color="#d9dce2", linewidth=0.7, alpha=0.6)
        ax.tick_params(labelsize=8, colors="#565b66")
        ax.set_facecolor("#eef0f4")

        if points and jitter["mean_gaze_x"] is not None:
            mean_grid_x = jitter["mean_gaze_x"] * cols / SCREEN_W
            mean_grid_y = jitter["mean_gaze_y"] * rows / SCREEN_H
            spread_w = max(0.7, jitter["p95_spread_px"] * cols / SCREEN_W * 2.0)
            spread_h = max(0.7, jitter["p95_spread_px"] * rows / SCREEN_H * 2.0)
            ax.add_patch(Ellipse(
                (mean_grid_x, mean_grid_y),
                width=spread_w,
                height=spread_h,
                fill=False,
                edgecolor="white",
                linewidth=1.4,
                alpha=0.9,
            ))
            ax.scatter([mean_grid_x], [mean_grid_y], s=28, c="white",
                       edgecolors="#222222", linewidths=0.8, zorder=5)

            jitter_label = (
                f"Jitter mean step: {jitter['mean_step_px']:.1f}px   "
                f"P95 step: {jitter['p95_step_px']:.1f}px   "
                f"Spread mean: {jitter['mean_spread_px']:.1f}px   "
                f"P95 spread: {jitter['p95_spread_px']:.1f}px"
            )
            ax.text(
                0.01,
                -0.11,
                jitter_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#30343b",
                bbox={"facecolor": "white", "edgecolor": "#c8ccd3", "alpha": 0.88, "pad": 4},
            )

        colorbar = fig.colorbar(heat, ax=ax, fraction=0.032, pad=0.025)
        colorbar.set_label("Density", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.ax.text(0.5, -0.08, "normalized", transform=colorbar.ax.transAxes,
                         ha="center", va="top", fontsize=6, color="#555555")

        fig.tight_layout()
        fig.savefig(output_path, dpi=140)
        plt.close(fig)

        csv_path = output_path.with_name(f"{output_path.stem}_density.csv")
        json_path = output_path.with_name(f"{output_path.stem}_values.json")

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["row", "col", "x_grid", "y_grid", "density"])
            for row in range(rows):
                for col in range(cols):
                    writer.writerow([row, col, col + 0.5, row + 0.5, float(density[row, col])])

        metadata = {
            "point_count": len(points),
            "screen_width": SCREEN_W,
            "screen_height": SCREEN_H,
            "grid_columns": cols,
            "grid_rows": rows,
            "density_min": float(density.min()) if density.size else 0.0,
            "density_max": float(density.max()) if density.size else 0.0,
            "density_mean": float(density.mean()) if density.size else 0.0,
            "jitter_metrics": jitter,
            "density_csv": str(csv_path),
            "density_values": density.tolist(),
        }
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        return str(output_path), len(points)

    def _handle_track_key(self, key):
        if key == 255:
            return None
        ch = chr(key).lower()
        if ch == 'q':
            return "quit"
        if ch == 'h':
            self._pip = not self._pip
            print(f"[Info] Camera window {'on' if self._pip else 'off'}.")
            return "handled"
        if ch == 'd':
            self._dbg = not self._dbg
            print(f"[Info] Debug landmarks {'on' if self._dbg else 'off'}.")
            return "handled"
        if ch == 'x':
            self._mouse_ctrl = not self._mouse_ctrl
            print(f"[Info] Mouse control {'on' if self._mouse_ctrl else 'off'}.")
            return "handled"
        return None

    # ── main loop ───────────────────────────────────────────────
    def calibrate(self, window_ready_callback=None):
        """
        Phase 1 — MUST run on the main thread (macOS OpenCV GUI requirement).
        Opens fullscreen calibration window, blocks until calibration completes,
        then destroys the window. Call track() in a background thread after this.
        """
        self._cap = cv2.VideoCapture(self.cam_id)
        configure_1080p_camera(self._cap)
        if not self._cap.isOpened():
            print(f"[Error] Cannot open camera {self.cam_id}")
            self._cap.release()
            cv2.destroyAllWindows()
            return False

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Info] Fullscreen 1920×1080  |  {self.num_points}-point calibration")
        print(f"[Info] Camera capture requested 1920×1080, using {actual_w}×{actual_h}")
        print("[Info] Q=quit  R=recalibrate  H=pip  D=debug  X=mouse ctrl")
        self._new_calib()

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if self._tutorial_enabled:
            if not self._show_tutorial_text([
                "Please look at the green circles",
                "Only blink when you see a check mark",
            ], seconds=10, window_ready_callback=window_ready_callback):
                self._cap.release()
                cv2.destroyAllWindows()
                return False
            window_ready_notified = True
        else:
            window_ready_notified = False

        while not self.calib.done:
            ret, cam = self._cap.read()
            if not ret:
                print("[Error] Camera frame read failed during calibration.")
                self._cap.release()
                cv2.destroyAllWindows()
                return False
            cam = cv2.flip(cam, 1)
            res = self.mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

            feat = lms = None
            if res.multi_face_landmarks:
                lms  = res.multi_face_landmarks[0].landmark
                feat = self.extractor.extract(lms, cam.shape[1], cam.shape[0])

            self._debug_cam(cam, lms)
            self._render_calib(cam, feat, lms)
            cv2.imshow(self.WIN, self.canvas)
            if not window_ready_notified:
                window_ready_notified = True
                if window_ready_callback:
                    try:
                        window_ready_callback()
                    except Exception:
                        pass

            key = cv2.waitKey(1) & 0xFF
            action = self._handle_calib_key(key)
            if action == "quit":
                self._cap.release()
                cv2.destroyAllWindows()
                return False

        # Calibration done — destroy window, continue tracking headlessly
        self._finalize_calibration_head_position()
        cv2.destroyWindow(self.WIN)
        cv2.waitKey(1)
        print("[Info] Calibration done — tracking active.")
        return True

    def track(self, stop_event=None):
        """
        Phase 2 — runs in a background thread (no OpenCV GUI).
        Requires calibrate() to have completed first.
        """
        cap = self._cap
        fps = self._fps
        self._tracking_frames = 0
        self._tracking_faces = 0
        self._tracking_predictions = 0
        self._mouse_moves = 0
        self._tracking_error = None
        window_ready = False

        try:
            while True:
                if stop_event and stop_event.is_set():
                    break

                ret, cam = cap.read()
                if not ret:
                    self._tracking_error = "Camera frame read failed"
                    break
                self._tracking_frames += 1
                cam = cv2.flip(cam, 1)
                res = self.mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

                feat = lms = None
                if res.multi_face_landmarks:
                    self._tracking_faces += 1
                    lms  = res.multi_face_landmarks[0].landmark
                    feat = self.extractor.extract(lms, cam.shape[1], cam.shape[0])

                self._debug_cam(cam, lms)
                self._render_track(cam, feat, lms, fps())
                if self._pip:
                    if not window_ready:
                        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.WIN, 960, 540)
                        window_ready = True
                    cv2.imshow(self.WIN, self.canvas)
                    action = self._handle_track_key(cv2.waitKey(1) & 0xFF)
                    if action == "quit":
                        break
                elif window_ready:
                    cv2.destroyWindow(self.WIN)
                    window_ready = False
        except Exception as e:
            self._tracking_error = str(e)
            print(f"[Error] Tracking crashed: {e}")

        if window_ready:
            cv2.destroyWindow(self.WIN)
        cap.release()
        print("[Info] Tracking stopped.")

    def run(self, calib_done_event=None, stop_event=None):
        """Legacy single-thread entry point (non-macOS or standalone use)."""
        ok = self.calibrate()
        if not ok:
            return
        if calib_done_event:
            calib_done_event.set()
        self.track(stop_event=stop_event)


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fullscreen 1920×1080 MediaPipe Gaze Tracker")
    ap.add_argument("--camera",  type=int,   default=0)
    ap.add_argument("--points",  type=int,   default=16, choices=[5, 9, 16, 25])
    ap.add_argument("--samples", type=int,   default=60)
    ap.add_argument("--ema",     type=float, default=0.30)
    ap.add_argument("--pnoise",  type=float, default=5e-3)
    ap.add_argument("--mnoise",  type=float, default=8.0)
    args = ap.parse_args()

    GazeTrackerApp(
        camera_id  = args.camera,
        num_points = args.points,
        ema_alpha  = args.ema,
        pnoise     = args.pnoise,
        mnoise     = args.mnoise,
        spp        = args.samples,
    ).run()
