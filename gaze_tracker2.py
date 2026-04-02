# hi whats up
"""
Gaze Tracker  –  EyeTrax-style fullscreen edition
==================================================
•  Fullscreen 1920 × 1080 gaze canvas (pure black background)
•  Camera PiP in the bottom-right corner
•  Multiple calibration modes:
   - Fixation-based: 5, 9, 16, or 25 points
   - Hybrid: 5 fixation points + smooth pursuit paths (--hybrid)
•  Polynomial regression gaze mapping with HEAD POSE compensation
•  Kalman Filter  +  EMA hybrid smoother
•  Blink detection (pauses tracking during blinks)
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

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080

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

# Smooth pursuit paths for hybrid calibration
# Each path is a list of (start, end) with intermediate sampling
PURSUIT_PATHS = [
    # From center to corners
    ((0.50, 0.50), (0.10, 0.10)),
    ((0.10, 0.10), (0.90, 0.10)),
    ((0.90, 0.10), (0.90, 0.90)),
    ((0.90, 0.90), (0.10, 0.90)),
    ((0.10, 0.90), (0.50, 0.50)),
    # Horizontal sweep
    ((0.10, 0.50), (0.90, 0.50)),
    # Vertical sweep
    ((0.50, 0.10), (0.50, 0.90)),
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
#  4.  Gaze Feature Extractor (with blink detection + head pose)
# ──────────────────────────────────────────────────────────────
class GazeFeatureExtractor:
    L_CORNERS = [33,  133]
    R_CORNERS = [362, 263]
    L_IRIS    = 468
    R_IRIS    = 473

    # Eye landmarks for EAR (Eye Aspect Ratio) blink detection
    # Left eye: outer corner, upper1, upper2, inner corner, lower2, lower1
    L_EYE = [33, 160, 158, 133, 153, 144]
    # Right eye: outer corner, upper1, upper2, inner corner, lower2, lower1
    R_EYE = [362, 385, 387, 263, 380, 373]

    EAR_THRESHOLD = 0.20  # Below this = blink detected

    # Landmarks for head pose estimation (6-point model)
    # Nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
    POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

    # 3D model points for a generic face (in mm, centered at nose)
    FACE_3D = np.array([
        [0.0, 0.0, 0.0],           # Nose tip
        [0.0, -63.6, -12.5],       # Chin
        [-43.3, 32.7, -26.0],      # Left eye outer
        [43.3, 32.7, -26.0],       # Right eye outer
        [-28.9, -28.9, -24.1],     # Left mouth corner
        [28.9, -28.9, -24.1],      # Right mouth corner
    ], dtype=np.float64)

    def __init__(self):
        self._cam_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _get_cam_matrix(self, w, h):
        """Get camera matrix (estimate focal length from image size)."""
        if self._cam_matrix is None or self._cam_matrix[0, 2] != w / 2:
            focal_length = w  # Approximate focal length
            self._cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)
        return self._cam_matrix

    def get_head_pose(self, lms, w, h):
        """Extract head pose (pitch, yaw, roll) in degrees."""
        try:
            # Get 2D landmark positions
            pts_2d = np.array([
                [lms[i].x * w, lms[i].y * h] for i in self.POSE_LANDMARKS
            ], dtype=np.float64)

            cam_matrix = self._get_cam_matrix(w, h)

            # Solve PnP to get rotation and translation vectors
            success, rot_vec, _ = cv2.solvePnP(
                self.FACE_3D, pts_2d, cam_matrix, self._dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return np.array([0.0, 0.0, 0.0])

            # Convert rotation vector to rotation matrix
            rot_mat, _ = cv2.Rodrigues(rot_vec)

            # Get Euler angles from rotation matrix
            # Using decomposition that gives pitch, yaw, roll
            sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
                yaw = math.atan2(-rot_mat[2, 0], sy)
                roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                yaw = math.atan2(-rot_mat[2, 0], sy)
                roll = 0

            # Convert to degrees and normalize
            return np.array([
                math.degrees(pitch) / 45.0,  # Normalize to ~[-1, 1] range
                math.degrees(yaw) / 45.0,
                math.degrees(roll) / 45.0
            ])
        except Exception:
            return np.array([0.0, 0.0, 0.0])

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
        """Extract gaze features including head pose compensation."""
        try:
            def p(i): return np.array([lms[i].x * w, lms[i].y * h])
            def norm(iris, c0, c1):
                ctr = (p(c0) + p(c1)) / 2.0
                return (iris - ctr) / (np.linalg.norm(p(c1)-p(c0)) / 2.0 + 1e-6)

            nl = norm(p(self.L_IRIS), *self.L_CORNERS)
            nr = norm(p(self.R_IRIS), *self.R_CORNERS)

            # Get head pose (pitch, yaw, roll)
            pose = self.get_head_pose(lms, w, h)

            # Feature vector: iris positions + squared terms + head pose + cross terms
            return np.array([
                # Iris positions (4)
                nl[0], nl[1], nr[0], nr[1],
                # Squared terms (4)
                nl[0]**2, nl[1]**2, nr[0]**2, nr[1]**2,
                # Head pose (3)
                pose[0], pose[1], pose[2],
                # Iris-pose cross terms (6) - these help compensate for head movement
                nl[0] * pose[1], nl[1] * pose[0],  # left iris × yaw/pitch
                nr[0] * pose[1], nr[1] * pose[0],  # right iris × yaw/pitch
                (nl[0] + nr[0]) * pose[1] / 2,     # avg horizontal × yaw
                (nl[1] + nr[1]) * pose[0] / 2,     # avg vertical × pitch
            ], dtype=float)
        except Exception:
            return None

    def iris_px(self, lms, w, h):
        def p(i): return (int(lms[i].x*w), int(lms[i].y*h))
        return p(self.L_IRIS), p(self.R_IRIS)


# ──────────────────────────────────────────────────────────────
#  5.  Polynomial Regression Gaze Model (with head pose support)
# ──────────────────────────────────────────────────────────────
class GazeRegressionModel:
    def __init__(self):
        self.Wx = self.Wy = None
        self.fitted = False

    def _design(self, F):
        """Build design matrix from features.

        Features expected (17 total):
        0-3: iris positions (nl_x, nl_y, nr_x, nr_y)
        4-7: squared terms
        8-10: head pose (pitch, yaw, roll)
        11-16: iris-pose cross terms
        """
        N = F.shape[0]
        n_feat = F.shape[1]

        # Start with bias and all features
        design = [np.ones((N, 1)), F]

        # Add key cross terms for iris positions
        if n_feat >= 4:
            iris_cross = np.column_stack([
                F[:, 0] * F[:, 1],   # left iris x*y
                F[:, 2] * F[:, 3],   # right iris x*y
                F[:, 0] * F[:, 2],   # left x * right x
                F[:, 1] * F[:, 3],   # left y * right y
            ])
            design.append(iris_cross)

        return np.hstack(design)

    def fit(self, F, T):
        A = self._design(F)
        # Use ridge regression for better stability with more features
        ridge = 1e-4
        AtA = A.T @ A + ridge * np.eye(A.shape[1])
        AtT_x = A.T @ T[:, 0]
        AtT_y = A.T @ T[:, 1]
        self.Wx = np.linalg.solve(AtA, AtT_x)
        self.Wy = np.linalg.solve(AtA, AtT_y)
        self.fitted = True
        print(f"[Calibration] Model fitted with {F.shape[1]} features, {A.shape[1]} design columns")

    def predict(self, f):
        if not self.fitted:
            return np.array([0.5, 0.5])
        a = self._design(f.reshape(1, -1))
        return np.clip([float(a @ self.Wx), float(a @ self.Wy)], 0, 1)


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
#  6b. Hybrid Calibration Manager (Fixation + Smooth Pursuit)
# ──────────────────────────────────────────────────────────────
class HybridCalibrationManager:
    """
    Hybrid calibration: fixation points + smooth pursuit paths.
    Phase 1: Fixation at 5 anchor points (corners + center)
    Phase 2: Smooth pursuit along paths connecting screen regions
    """
    HOLD = 15           # frames to stabilize before collecting fixation
    FIX_SAMPLES = 30    # samples per fixation point
    PURSUIT_SPEED = 3.0 # seconds per pursuit path
    PURSUIT_FPS = 30    # approximate frames per second

    def __init__(self):
        # Fixation anchors: center first, then corners
        self.fix_pts = [
            (0.50, 0.50),  # center
            (0.12, 0.12), (0.88, 0.12),  # top corners
            (0.12, 0.88), (0.88, 0.88),  # bottom corners
        ]
        self.pursuit_paths = PURSUIT_PATHS

        # State
        self.phase = 'fixation'  # 'fixation' or 'pursuit'
        self.fix_idx = 0
        self.pursuit_idx = 0
        self.feats = []
        self.tgts = []
        self.done = False
        self.model = GazeRegressionModel()

        # Fixation state
        self._fp = 0
        self._fix_buf = []

        # Pursuit state
        self._pursuit_start_time = None
        self._pursuit_t = 0.0

    @property
    def current_target(self):
        """Returns current (x, y) in normalized coords [0,1]."""
        if self.phase == 'fixation':
            return self.fix_pts[self.fix_idx]
        else:
            # Interpolate along pursuit path
            start, end = self.pursuit_paths[self.pursuit_idx]
            t = self._pursuit_t
            return (
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t
            )

    @property
    def target_px(self):
        fx, fy = self.current_target
        return int(fx * SCREEN_W), int(fy * SCREEN_H)

    def add_sample(self, feat):
        """Add a gaze feature sample at the current target position."""
        if self.done:
            return

        if self.phase == 'fixation':
            self._add_fixation_sample(feat)
        else:
            self._add_pursuit_sample(feat)

    def _add_fixation_sample(self, feat):
        self._fp += 1
        if self._fp <= self.HOLD:
            return  # Stabilizing

        tx, ty = self.fix_pts[self.fix_idx]
        self._fix_buf.append(feat)
        self.feats.append(feat)
        self.tgts.append([tx, ty])

        if len(self._fix_buf) >= self.FIX_SAMPLES:
            # Move to next fixation point
            self._fix_buf = []
            self._fp = 0
            self.fix_idx += 1

            if self.fix_idx >= len(self.fix_pts):
                # Switch to pursuit phase
                self.phase = 'pursuit'
                self._pursuit_start_time = time.time()
                print(f"[Calibration] Fixation complete ({len(self.feats)} samples), starting smooth pursuit...")

    def _add_pursuit_sample(self, feat):
        if self._pursuit_start_time is None:
            self._pursuit_start_time = time.time()

        # Update pursuit progress
        elapsed = time.time() - self._pursuit_start_time
        self._pursuit_t = min(elapsed / self.PURSUIT_SPEED, 1.0)

        # Sample at current interpolated position
        tx, ty = self.current_target
        self.feats.append(feat)
        self.tgts.append([tx, ty])

        if self._pursuit_t >= 1.0:
            # Path complete, move to next
            self.pursuit_idx += 1
            self._pursuit_start_time = time.time()
            self._pursuit_t = 0.0

            if self.pursuit_idx >= len(self.pursuit_paths):
                # All done!
                self.model.fit(np.array(self.feats), np.array(self.tgts))
                print(f"[Calibration] Hybrid complete — {len(self.feats)} total samples "
                      f"({len(self.fix_pts)} fixation + {len(self.pursuit_paths)} pursuit paths)")
                self.done = True

    def progress(self):
        """Returns overall progress 0.0 to 1.0."""
        fix_total = len(self.fix_pts) * self.FIX_SAMPLES
        pursuit_total = len(self.pursuit_paths)  # Each path = 1 unit

        if self.phase == 'fixation':
            fix_done = self.fix_idx * self.FIX_SAMPLES + max(0, self._fp - self.HOLD)
            return (fix_done / fix_total) * 0.5  # Fixation = first 50%
        else:
            pursuit_done = self.pursuit_idx + self._pursuit_t
            return 0.5 + (pursuit_done / pursuit_total) * 0.5  # Pursuit = second 50%

    def dot_state(self):
        """Returns (phase_name, progress) for rendering."""
        if self.phase == 'fixation':
            if self._fp <= self.HOLD:
                return 'hold', self._fp / self.HOLD
            return 'collect', len(self._fix_buf) / self.FIX_SAMPLES
        else:
            return 'pursuit', self._pursuit_t

    def get_status_text(self):
        """Get descriptive status for UI."""
        if self.phase == 'fixation':
            return f"Fixation {self.fix_idx + 1}/{len(self.fix_pts)}"
        else:
            return f"Pursuit {self.pursuit_idx + 1}/{len(self.pursuit_paths)}"


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

def draw_pursuit_dot(canvas, cx, cy, t, R=24):
    """Draw a smooth pursuit target with motion trail."""
    # Outer glow - intensity based on progress t
    pulse = 0.5 + 0.5 * math.sin(time.time() * 10)
    glow_r = int(R + 8 + 4 * pulse)
    glow_intensity = int(40 + 40 * t)  # Gets brighter as path progresses
    cv2.circle(canvas, (cx, cy), glow_r, (glow_intensity, 80 + int(40*t), glow_intensity), 2, cv2.LINE_AA)

    # Main dot (bright, easy to follow)
    cv2.circle(canvas, (cx, cy), R, (0, 255, 100), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), R, (100, 255, 150), 2, cv2.LINE_AA)

    # Inner highlight
    cv2.circle(canvas, (cx, cy), 8, (200, 255, 220), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)

def draw_pursuit_path_preview(canvas, paths, current_idx):
    """Draw faded preview of pursuit paths."""
    for i, (start, end) in enumerate(paths):
        sx, sy = int(start[0] * SCREEN_W), int(start[1] * SCREEN_H)
        ex, ey = int(end[0] * SCREEN_W), int(end[1] * SCREEN_H)
        if i < current_idx:
            # Completed path
            cv2.line(canvas, (sx, sy), (ex, ey), (0, 80, 40), 1, cv2.LINE_AA)
        elif i == current_idx:
            # Current path - brighter
            cv2.line(canvas, (sx, sy), (ex, ey), (0, 150, 80), 2, cv2.LINE_AA)
        else:
            # Future path - dim
            cv2.line(canvas, (sx, sy), (ex, ey), (30, 40, 30), 1, cv2.LINE_AA)

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

def draw_bar(canvas, prog, x, y, w, h):
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (45,45,45), -1)
    fill = int(w * prog)
    if fill > 0:
        cv2.rectangle(canvas, (x,y), (x+fill,y+h), lerp_color((0,120,60), C_ACCENT, prog), -1)
    cv2.rectangle(canvas, (x,y), (x+w,y+h), (110,110,110), 1)

def overlay_pip(canvas, cam_frame, pip_w=320, pip_h=240):
    if cam_frame is None: return
    x0, y0 = SCREEN_W - pip_w - 20, SCREEN_H - pip_h - 20
    cv2.rectangle(canvas, (x0-3,y0-3), (x0+pip_w+3,y0+pip_h+3), (55,55,55), 2)
    canvas[y0:y0+pip_h, x0:x0+pip_w] = cv2.resize(cam_frame, (pip_w,pip_h))
    txt(canvas, "Camera", (x0+4, y0+pip_h-8), scale=0.40, color=(140,140,140))


# ──────────────────────────────────────────────────────────────
#  7.  Main Application
# ──────────────────────────────────────────────────────────────
class GazeTrackerApp:
    WIN = "GazeTracker"

    def __init__(self, camera_id=0, num_points=16, ema_alpha=0.3,
                pnoise=5e-3, mnoise=8.0, spp=60, hybrid=False):
        self.cam_id     = camera_id
        self.num_points = num_points
        self.spp        = spp
        self.hybrid     = hybrid

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.extractor = GazeFeatureExtractor()
        self.smoother  = HybridSmoother(pnoise, mnoise, ema_alpha)
        self.calib     = None

        self.hist   = collections.deque(maxlen=50)
        self._fpsq  = collections.deque(maxlen=30)
        self._pip   = True
        self._dbg   = False
        self._blink = False  # blink state indicator
        self._last_gaze = (SCREEN_W // 2, SCREEN_H // 2)  # last known good gaze
        self.canvas = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)

    def _new_calib(self):
        if self.hybrid:
            self.calib = HybridCalibrationManager()
        else:
            self.calib = CalibrationManager(self.num_points, self.spp)
        self.smoother.reset(); self.hist.clear()

    def _fps(self):
        now = time.time(); self._fpsq.append(now)
        return (len(self._fpsq)-1)/(self._fpsq[-1]-self._fpsq[0]+1e-9) if len(self._fpsq)>1 else 0

    # ── calibration render ──────────────────────────────────────
    def _render_calib(self, cam, feat, lms=None):
        cv = self.canvas; cv[:] = C_BG
        draw_grid(cv)

        # Skip blink frames during calibration
        is_blinking = False
        if lms is not None:
            is_blinking = self.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])

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
        hint = "Keep your eyes on the dot  ·  H=pip  D=debug  Q=quit"
        hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        txt(cv, hint, (SCREEN_W//2 - hw//2, SCREEN_H-18), scale=0.48, color=(100,100,100))

        if self._pip: overlay_pip(cv, cam)
        # Only add calibration samples when not blinking
        if feat is not None and not is_blinking:
            self.calib.add_sample(feat)

    # ── hybrid calibration render ────────────────────────────────
    def _render_hybrid_calib(self, cam, feat, lms=None):
        cv = self.canvas; cv[:] = C_BG
        draw_grid(cv)

        # Skip blink frames
        is_blinking = False
        if lms is not None:
            is_blinking = self.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])

        phase, t = self.calib.dot_state()
        cx, cy = self.calib.target_px

        if self.calib.phase == 'fixation':
            # Draw fixation points
            for i, (fx, fy) in enumerate(self.calib.fix_pts):
                px, py = int(fx * SCREEN_W), int(fy * SCREEN_H)
                if i < self.calib.fix_idx:
                    # Completed
                    cv2.circle(cv, (px, py), 12, C_ACCENT, 2, cv2.LINE_AA)
                    cv2.line(cv, (px-5, py), (px-1, py+5), C_ACCENT, 2, cv2.LINE_AA)
                    cv2.line(cv, (px-1, py+5), (px+6, py-4), C_ACCENT, 2, cv2.LINE_AA)
                elif i > self.calib.fix_idx:
                    # Future
                    draw_crosshair(cv, px, py)
            # Draw current fixation dot
            draw_calib_dot(cv, cx, cy, phase, t)
        else:
            # Pursuit phase - show paths and moving dot
            draw_pursuit_path_preview(cv, self.calib.pursuit_paths, self.calib.pursuit_idx)
            draw_pursuit_dot(cv, cx, cy, t)

        # Progress bar
        prog = self.calib.progress()
        bx = SCREEN_W // 2 - 300
        by = SCREEN_H - 60
        draw_bar(cv, prog, bx, by, 600, 16)

        # Status text
        status = self.calib.get_status_text()
        label = f"Hybrid Calibration  ·  {status}   ({int(prog*100)}%)"
        lw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        txt(cv, label, (SCREEN_W // 2 - lw // 2, by - 12), scale=0.65, color=C_TEXT)

        if self.calib.phase == 'fixation':
            hint = "Keep your eyes on the dot  ·  H=pip  D=debug  Q=quit"
        else:
            hint = "Follow the moving dot smoothly  ·  H=pip  D=debug  Q=quit"
        hw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        txt(cv, hint, (SCREEN_W // 2 - hw // 2, SCREEN_H - 18), scale=0.48, color=(100, 100, 100))

        if self._pip:
            overlay_pip(cv, cam)

        # Collect samples when not blinking
        if feat is not None and not is_blinking:
            self.calib.add_sample(feat)

    # ── tracking render ─────────────────────────────────────────
    def _render_track(self, cam, feat, lms, fps):
        cv = self.canvas; cv[:] = C_BG
        draw_grid(cv)

        # Check for blink - if blinking, skip gaze update and use last position
        if lms is not None:
            self._blink = self.extractor.is_blinking(lms, cam.shape[1], cam.shape[0])

        if feat is not None and not self._blink:
            raw = self.calib.model.predict(feat)
            smo = self.smoother.update(raw)
            gx  = int(np.clip(smo[0]*SCREEN_W,  0, SCREEN_W-1))
            gy  = int(np.clip(smo[1]*SCREEN_H, 0, SCREEN_H-1))
            self._last_gaze = (gx, gy)
            self.hist.append((gx, gy))

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
        ctrl = "R=recalibrate   H=pip   D=debug   Q=quit"
        cw = cv2.getTextSize(ctrl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        txt(cv, ctrl, (SCREEN_W//2-cw//2, 26), scale=0.48, color=(110,110,110))

        if self._pip: overlay_pip(cv, cam)

    # ── debug overlay on camera pip ─────────────────────────────
    def _debug_cam(self, cam, lms):
        if not self._dbg or lms is None: return
        h,w = cam.shape[:2]
        for pt in self.extractor.iris_px(lms,w,h):
            cv2.circle(cam, pt, 4, (0,220,255), -1)
        for idx in GazeFeatureExtractor.L_CORNERS + GazeFeatureExtractor.R_CORNERS:
            cv2.circle(cam,(int(lms[idx].x*w),int(lms[idx].y*h)),3,(255,100,0),-1)

    # ── main loop ───────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            print(f"[Error] Cannot open camera {self.cam_id}"); return

        calib_mode = "hybrid (fixation + pursuit)" if self.hybrid else f"{self.num_points}-point"
        print(f"[Info] Fullscreen 1920×1080  |  {calib_mode} calibration")
        print("[Info] Q=quit  R=recalibrate  H=pip  D=debug")
        self._new_calib()

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, cam = cap.read()
            if not ret: break
            cam = cv2.flip(cam, 1)
            res = self.mesh.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))

            feat = lms = None
            if res.multi_face_landmarks:
                lms  = res.multi_face_landmarks[0].landmark
                feat = self.extractor.extract(lms, cam.shape[1], cam.shape[0])

            self._debug_cam(cam, lms)
            fps = self._fps()

            if not self.calib.done:
                if self.hybrid:
                    self._render_hybrid_calib(cam, feat, lms)
                else:
                    self._render_calib(cam, feat, lms)
                if self.calib.done:
                    print("[Info] Calibration done — tracking active.")
            else:
                self._render_track(cam, feat, lms, fps)

            cv2.imshow(self.WIN, self.canvas)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('r'): print("[Info] Recalibrating…"); self._new_calib()
            elif key == ord('h'): self._pip = not self._pip
            elif key == ord('d'): self._dbg = not self._dbg

        cap.release(); cv2.destroyAllWindows()
        print("[Info] Stopped.")


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
    ap.add_argument("--hybrid",  action="store_true",
                    help="Use hybrid calibration (fixation + smooth pursuit)")
    args = ap.parse_args()

    GazeTrackerApp(
        camera_id  = args.camera,
        num_points = args.points,
        ema_alpha  = args.ema,
        pnoise     = args.pnoise,
        mnoise     = args.mnoise,
        spp        = args.samples,
        hybrid     = args.hybrid,
    ).run()