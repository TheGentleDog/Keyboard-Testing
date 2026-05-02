# =============================================================================
# run.py — Single entry point for Gaze-Based Filipino Keyboard
#
# Flow:
#   1. Launcher UI — user picks calibration settings
#   2. Gaze tracker starts (OpenCV, fullscreen calibration)
#   3. Once calibration is done, mouse control is enabled automatically
#   4. Tkinter keyboard launches on the main thread
#   5. Gaze tracker keeps running in background, moving the mouse cursor
#   6. Closing the keyboard also stops the gaze tracker
# =============================================================================

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk

# ── Make keyboard modules importable ─────────────────────────────────────────
KEYBOARD_DIR = os.path.join(os.path.dirname(__file__), "Bench", "Cutted_File", "files")
sys.path.insert(0, KEYBOARD_DIR)


# =============================================================================
#  Launcher UI
# =============================================================================

class LauncherUI(tk.Tk):
    """
    Dark-themed startup window.
    User picks calibration & tracker settings, then clicks Start.
    Returns settings via self.result (None if cancelled).
    """

    DARK = {
        "bg":         "#1e1f22",
        "panel":      "#2b2d31",
        "card":       "#313338",
        "card_alt":   "#26282d",
        "accent":     "#5865f2",
        "accent_hov": "#4752c4",
        "accent_soft":"#2f3f87",
        "text":       "#dcddde",
        "subtext":    "#96989d",
        "muted":      "#7d8187",
        "danger":     "#ed4245",
        "border":     "#3f4147",
    }

    def __init__(self):
        super().__init__()
        self.result = None
        self._preview_stop = None
        self._preview_thread = None
        d = self.DARK

        self.title("Gaze Keyboard — Launcher")
        self.resizable(False, False)
        self.configure(bg=d["bg"])

        # ── Center window ─────────────────────────────────────────
        W, H = 560, 770
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

        self._build(d)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))

    def _section(self, parent, title, subtitle=None):
        """Returns a card frame with a section label."""
        d = self.DARK
        outer = tk.Frame(parent, bg=d["card"], bd=0, highlightbackground=d["border"],
                         highlightthickness=1)
        outer.pack(fill="x", padx=18, pady=(0, 12))
        tk.Label(outer, text=title, bg=d["card"], fg=d["subtext"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(10, 2))
        if subtitle:
            tk.Label(outer, text=subtitle, bg=d["card"], fg=d["muted"],
                     font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=(0, 6))
        inner = tk.Frame(outer, bg=d["card"])
        inner.pack(fill="x", padx=14, pady=(0, 12))
        return inner

    def _row(self, parent, label, widget_fn):
        """Two-column row: label left, widget right."""
        d = self.DARK
        row = tk.Frame(parent, bg=d["card"])
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        w = widget_fn(row)
        w.pack(side="left", fill="x", expand=True)
        return w

    def _build(self, d):
        # ── Header ───────────────────────────────────────────────
        hdr = tk.Frame(self, bg=d["panel"])
        hdr.pack(fill="x")
        pill = tk.Label(hdr, text="SESSION SETUP",
                        bg=d["accent_soft"], fg="#ffffff",
                        font=("Segoe UI", 8, "bold"),
                        padx=10, pady=4)
        pill.pack(anchor="w", padx=18, pady=(18, 10))
        tk.Label(hdr, text="Gaze-Based Filipino Keyboard",
                 bg=d["panel"], fg=d["text"],
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=18)
        tk.Label(hdr, text="Tune calibration, smoothing, dwell mode, and layout before launch.",
                 bg=d["panel"], fg=d["subtext"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=(4, 16))

        body_outer = tk.Frame(self, bg=d["bg"])
        body_outer.pack(fill="both", expand=True, pady=10)

        canvas = tk.Canvas(body_outer, bg=d["bg"], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(body_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=d["bg"])
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def _sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _resize_body(_event):
            canvas.itemconfigure(body_window, width=_event.width)

        body.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _resize_body)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        # ────────────────────────────────────────────────────────
        #  CALIBRATION
        # ────────────────────────────────────────────────────────
        calib = self._section(body, "CALIBRATION",
                              "Grid density and sample count affect precision and startup time.")

        # Calibration points
        self._points_var = tk.IntVar(value=9)
        pts_frame = tk.Frame(calib, bg=d["card"])
        pts_frame.pack(fill="x", pady=3)
        tk.Label(pts_frame, text="Grid points", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        for pts, label in [(5, "5  (fast)"), (9, "9  (default)"),
                           (16, "16  (precise)"), (25, "25  (max)")]:
            rb = tk.Radiobutton(pts_frame, text=label, variable=self._points_var,
                                value=pts, bg=d["card"], fg=d["text"],
                                selectcolor=d["accent"], activebackground=d["card"],
                                activeforeground=d["text"],
                                font=("Segoe UI", 10))
            rb.pack(side="left", padx=6)

        # Samples per point
        self._samples_var = tk.IntVar(value=90)
        samples_frame = tk.Frame(calib, bg=d["card"])
        samples_frame.pack(fill="x", pady=3)
        tk.Label(samples_frame, text="Samples / point", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        self._samples_lbl = tk.Label(samples_frame, text="90", bg=d["card"],
                                     fg=d["accent"], font=("Segoe UI", 11, "bold"), width=4)
        self._samples_lbl.pack(side="right", padx=(0, 8))
        sl = tk.Scale(samples_frame, from_=20, to=120, orient="horizontal",
                      variable=self._samples_var, showvalue=False,
                      bg=d["card"], fg=d["text"], troughcolor=d["border"],
                      highlightthickness=0, command=lambda v: self._samples_lbl.config(text=v))
        sl.pack(side="left", fill="x", expand=True)

        # ────────────────────────────────────────────────────────
        #  DWELL MODE
        # ────────────────────────────────────────────────────────
        dwell = self._section(body, "DWELL MODE",
                              "Choose whether selection happens by trial winner or immediate hold.")
        self._dwell_mode_var = tk.StringVar(value="sync")
        dwell_frame = tk.Frame(dwell, bg=d["card"])
        dwell_frame.pack(fill="x", pady=3)
        tk.Label(dwell_frame, text="Selection mode", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        for mode, label in [("sync", "Synchronous"), ("async", "Asynchronous")]:
            rb = tk.Radiobutton(dwell_frame, text=label, variable=self._dwell_mode_var,
                                value=mode, bg=d["card"], fg=d["text"],
                                selectcolor=d["accent"], activebackground=d["card"],
                                activeforeground=d["text"],
                                font=("Segoe UI", 10))
            rb.pack(side="left", padx=8)

        # ────────────────────────────────────────────────────────
        #  UI LAYOUT
        # ────────────────────────────────────────────────────────
        layout = self._section(body, "UI LAYOUT",
                               "Pick the keyboard layout shown after calibration.")
        self._ui_layout_var = tk.StringVar(value="qwerty")
        layout_frame = tk.Frame(layout, bg=d["card"])
        layout_frame.pack(fill="x", pady=3)
        tk.Label(layout_frame, text="Keyboard UI", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        for value, label in [("qwerty", "QWERTY"), ("ui2", "UI2")]:
            rb = tk.Radiobutton(layout_frame, text=label, variable=self._ui_layout_var,
                                value=value, bg=d["card"], fg=d["text"],
                                selectcolor=d["accent"], activebackground=d["card"],
                                activeforeground=d["text"],
                                font=("Segoe UI", 10))
            rb.pack(side="left", padx=8)

        # ────────────────────────────────────────────────────────
        #  SMOOTHER
        # ────────────────────────────────────────────────────────
        smooth = self._section(body, "SMOOTHER",
                               "EMA reduces jitter before gaze positions are applied.")

        # EMA toggle
        self._ema_on = tk.BooleanVar(value=True)
        ema_frame = tk.Frame(smooth, bg=d["card"])
        ema_frame.pack(fill="x", pady=3)
        tk.Label(ema_frame, text="EMA smoother", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        tk.Checkbutton(ema_frame, text="Enabled", variable=self._ema_on,
                       bg=d["card"], fg=d["text"], selectcolor=d["accent"],
                       activebackground=d["card"], activeforeground=d["text"],
                       font=("Segoe UI", 10),
                       command=self._toggle_ema).pack(side="left")

        # EMA alpha
        self._ema_var = tk.DoubleVar(value=0.30)
        self._ema_frame = tk.Frame(smooth, bg=d["card"])
        self._ema_frame.pack(fill="x", pady=3)
        tk.Label(self._ema_frame, text="EMA alpha  (0–1)", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        self._ema_lbl = tk.Label(self._ema_frame, text="0.30", bg=d["card"],
                                 fg=d["accent"], font=("Segoe UI", 11, "bold"), width=5)
        self._ema_lbl.pack(side="right", padx=(0, 8))
        tk.Scale(self._ema_frame, from_=0.01, to=1.0, resolution=0.01,
                 orient="horizontal", variable=self._ema_var, showvalue=False,
                 bg=d["card"], fg=d["text"], troughcolor=d["border"],
                 highlightthickness=0,
                 command=lambda v: self._ema_lbl.config(text=f"{float(v):.2f}")
                 ).pack(side="left", fill="x", expand=True)

        # ────────────────────────────────────────────────────────
        #  KALMAN FILTER
        # ────────────────────────────────────────────────────────
        kalman = self._section(body, "KALMAN FILTER",
                               "Lower noise values react faster; higher values smooth more.")

        # Process noise
        self._pnoise_var = tk.DoubleVar(value=0.0100)
        pn_frame = tk.Frame(kalman, bg=d["card"])
        pn_frame.pack(fill="x", pady=3)
        tk.Label(pn_frame, text="Process noise", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        self._pnoise_lbl = tk.Label(pn_frame, text="0.0100", bg=d["card"],
                                    fg=d["accent"], font=("Segoe UI", 11, "bold"), width=6)
        self._pnoise_lbl.pack(side="right", padx=(0, 8))
        tk.Scale(pn_frame, from_=1e-4, to=0.1, resolution=1e-4,
                 orient="horizontal", variable=self._pnoise_var, showvalue=False,
                 bg=d["card"], fg=d["text"], troughcolor=d["border"],
                 highlightthickness=0,
                 command=lambda v: self._pnoise_lbl.config(text=f"{float(v):.4f}")
                 ).pack(side="left", fill="x", expand=True)

        # Measurement noise
        self._mnoise_var = tk.DoubleVar(value=5.5)
        mn_frame = tk.Frame(kalman, bg=d["card"])
        mn_frame.pack(fill="x", pady=3)
        tk.Label(mn_frame, text="Measurement noise", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        self._mnoise_lbl = tk.Label(mn_frame, text="5.5", bg=d["card"],
                                    fg=d["accent"], font=("Segoe UI", 11, "bold"), width=6)
        self._mnoise_lbl.pack(side="right", padx=(0, 8))
        tk.Scale(mn_frame, from_=1.0, to=50.0, resolution=0.5,
                 orient="horizontal", variable=self._mnoise_var, showvalue=False,
                 bg=d["card"], fg=d["text"], troughcolor=d["border"],
                 highlightthickness=0,
                 command=lambda v: self._mnoise_lbl.config(text=f"{float(v):.1f}")
                 ).pack(side="left", fill="x", expand=True)

        # ────────────────────────────────────────────────────────
        #  CAMERA
        # ────────────────────────────────────────────────────────
        cam = self._section(body, "CAMERA",
                            "Use the camera index that matches the device you want for tracking.")
        self._camera_var = tk.IntVar(value=0)
        cam_frame = tk.Frame(cam, bg=d["card"])
        cam_frame.pack(fill="x", pady=3)
        tk.Label(cam_frame, text="Camera index", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        for idx, lbl in [(0, "0  (built-in)"), (1, "1  (external)"), (2, "2")]:
            tk.Radiobutton(cam_frame, text=lbl, variable=self._camera_var,
                           value=idx, bg=d["card"], fg=d["text"],
                           selectcolor=d["accent"], activebackground=d["card"],
                           activeforeground=d["text"],
                           font=("Segoe UI", 10)).pack(side="left", padx=6)

        camera_window = self._section(body, "CAMERA WINDOW",
                                      "Preview and debug tools shown during eye tracking.")
        self._camera_window_var = tk.BooleanVar(value=True)
        self._camera_debug_var = tk.BooleanVar(value=True)
        self._distance_var = tk.BooleanVar(value=True)

        preview_frame = tk.Frame(camera_window, bg=d["card"])
        preview_frame.pack(fill="x", pady=3)
        tk.Label(preview_frame, text="Camera preview", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        tk.Checkbutton(preview_frame, text="Enabled", variable=self._camera_window_var,
                       bg=d["card"], fg=d["text"], selectcolor=d["accent"],
                       activebackground=d["card"], activeforeground=d["text"],
                       font=("Segoe UI", 10)).pack(side="left")

        posture_frame = tk.Frame(camera_window, bg=d["card"])
        posture_frame.pack(fill="x", pady=(8, 3))
        tk.Label(posture_frame, text="Setup preview", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        self._preview_btn = tk.Label(posture_frame, text="Open Preview", bg=d["accent"],
                                     fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                     relief="flat", bd=0, padx=14, pady=7,
                                     cursor="hand2")
        self._preview_btn.pack(side="left")
        self._preview_btn.bind("<Button-1>", self._toggle_setup_preview)
        self._preview_btn.bind("<Enter>", lambda _: self._preview_btn.config(bg=d["accent_hov"]))
        self._preview_btn.bind("<Leave>", lambda _: self._sync_preview_button())
        self._preview_status = tk.Label(posture_frame, text="Closed", bg=d["card"],
                                        fg=d["muted"], font=("Segoe UI", 9))
        self._preview_status.pack(side="left", padx=10)

        debug_frame = tk.Frame(camera_window, bg=d["card"])
        debug_frame.pack(fill="x", pady=3)
        tk.Label(debug_frame, text="Debug landmarks", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        tk.Checkbutton(debug_frame, text="Enabled", variable=self._camera_debug_var,
                       bg=d["card"], fg=d["text"], selectcolor=d["accent"],
                       activebackground=d["card"], activeforeground=d["text"],
                       font=("Segoe UI", 10)).pack(side="left")

        distance_frame = tk.Frame(camera_window, bg=d["card"])
        distance_frame.pack(fill="x", pady=3)
        tk.Label(distance_frame, text="Distance panel", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        tk.Checkbutton(distance_frame, text="Enabled", variable=self._distance_var,
                       bg=d["card"], fg=d["text"], selectcolor=d["accent"],
                       activebackground=d["card"], activeforeground=d["text"],
                       font=("Segoe UI", 10)).pack(side="left")

        # ────────────────────────────────────────────────────────
        #  Buttons
        # ────────────────────────────────────────────────────────
        footer = tk.Frame(self, bg=d["panel"])
        footer.pack(fill="x", side="bottom")
        tk.Label(footer, text="You can recalibrate later with R during the session.",
                 bg=d["panel"], fg=d["muted"], font=("Segoe UI", 9)).pack(anchor="w", padx=18, pady=(10, 6))

        btn_row = tk.Frame(footer, bg=d["panel"])
        btn_row.pack(fill="x", padx=18, pady=(0, 16))

        cancel_btn = tk.Label(btn_row, text="Cancel", bg=d["card_alt"], fg=d["text"],
                              font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
                              padx=22, pady=11, cursor="hand2")
        cancel_btn.pack(side="left")
        cancel_btn.bind("<Button-1>", self._on_cancel)
        cancel_btn.bind("<Enter>", lambda _: cancel_btn.config(bg=d["border"]))
        cancel_btn.bind("<Leave>", lambda _: cancel_btn.config(bg=d["card_alt"]))

        start_btn = tk.Label(btn_row, text="Start with Tutorial", bg=d["accent"], fg="#ffffff",
                             font=("Segoe UI", 12, "bold"), relief="flat", bd=0,
                             padx=20, pady=11, cursor="hand2")
        start_btn.pack(side="right")
        start_btn.bind("<Button-1>", lambda e: self._on_start(e, tutorial=True))
        start_btn.bind("<Enter>", lambda _: start_btn.config(bg=d["accent_hov"]))
        start_btn.bind("<Leave>", lambda _: start_btn.config(bg=d["accent"]))

        skip_btn = tk.Label(btn_row, text="Start without Tutorial", bg=d["card_alt"], fg=d["text"],
                            font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
                            padx=16, pady=11, cursor="hand2")
        skip_btn.pack(side="right", padx=(0, 10))
        skip_btn.bind("<Button-1>", lambda e: self._on_start(e, tutorial=False))
        skip_btn.bind("<Enter>", lambda _: skip_btn.config(bg=d["border"]))
        skip_btn.bind("<Leave>", lambda _: skip_btn.config(bg=d["card_alt"]))

    def _toggle_ema(self):
        state = "normal" if self._ema_on.get() else "disabled"
        for child in self._ema_frame.winfo_children():
            try:
                child.config(state=state)
            except Exception:
                pass

    def _sync_preview_button(self):
        d = self.DARK
        running = self._preview_thread is not None and self._preview_thread.is_alive()
        if hasattr(self, "_preview_btn"):
            self._preview_btn.config(
                text="Close Preview" if running else "Open Preview",
                bg=d["danger"] if running else d["accent"],
            )

    def _set_preview_status(self, text, color=None):
        if not hasattr(self, "_preview_status"):
            return
        self._preview_status.config(text=text, fg=color or self.DARK["muted"])
        self._sync_preview_button()

    def _toggle_setup_preview(self, _event=None):
        running = self._preview_thread is not None and self._preview_thread.is_alive()
        if running:
            self._stop_setup_preview()
        else:
            self._start_setup_preview()
        return "break"

    def _start_setup_preview(self):
        self._stop_setup_preview(join=False)
        self._preview_stop = threading.Event()
        self._preview_thread = threading.Thread(
            target=self._run_setup_preview,
            args=(
                self._camera_var.get(),
                self._camera_debug_var.get(),
                self._distance_var.get(),
            ),
            daemon=True,
        )
        self._preview_thread.start()
        self._set_preview_status("Opening...", self.DARK["accent"])

    def _stop_setup_preview(self, join=True):
        if self._preview_stop is not None:
            self._preview_stop.set()
        if join and self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=1.0)
        self._sync_preview_button()

    def _run_setup_preview(self, camera_id, debug_on, distance_on):
        win = "Setup Camera Preview"
        cap = None
        mesh = None
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
            import math

            camera_w, camera_h = 1920, 1080
            camera_hfov_deg = 60.0
            real_ipd_cm = 6.3

            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                self.after(0, lambda: self._set_preview_status("Camera not available", self.DARK["danger"]))
                return
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w == camera_w and actual_h == camera_h:
                status = "Preview open 1920x1080"
            else:
                status = f"Preview {actual_w}x{actual_h}"

            mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.after(0, lambda: self._set_preview_status(status, self.DARK["accent"]))

            while self._preview_stop is not None and not self._preview_stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.after(0, lambda: self._set_preview_status("Frame read failed", self.DARK["danger"]))
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                lms = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None

                if lms is not None:
                    left_eye = np.array([lms[468].x * w, lms[468].y * h])
                    right_eye = np.array([lms[473].x * w, lms[473].y * h])
                    eye_center = (left_eye + right_eye) / 2.0
                    pos_x = eye_center[0] - (w / 2.0)
                    pos_y = (h / 2.0) - eye_center[1]
                    pos_x_norm = pos_x / (w / 2.0)
                    pos_y_norm = pos_y / (h / 2.0)
                    focal_x_px = w / (2.0 * math.tan(math.radians(camera_hfov_deg) / 2.0))
                    angle_x = math.degrees(math.atan(pos_x / focal_x_px))
                    angle_y = math.degrees(math.atan(pos_y_norm))

                    cx, cy = int(w / 2), int(h / 2)
                    ex, ey = int(eye_center[0]), int(eye_center[1])
                    cv2.line(frame, (cx - 24, cy), (cx + 24, cy), (80, 80, 80), 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy - 24), (cx, cy + 24), (80, 80, 80), 1, cv2.LINE_AA)
                    cv2.circle(frame, (ex, ey), 7, (0, 220, 120), 2, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy), (ex, ey), (0, 160, 220), 1, cv2.LINE_AA)

                    if debug_on:
                        for idx in [468, 473]:
                            cv2.circle(frame, (int(lms[idx].x * w), int(lms[idx].y * h)), 4, (0, 220, 255), -1)
                        for idx in [33, 133, 362, 263]:
                            cv2.circle(frame, (int(lms[idx].x * w), int(lms[idx].y * h)), 3, (255, 100, 0), -1)

                    if distance_on:
                        ipd_px = float(np.linalg.norm(left_eye - right_eye))
                        if ipd_px > 1.0:
                            distance_cm = (real_ipd_cm * focal_x_px) / ipd_px
                            cv2.rectangle(frame, (10, 10), (330, 222), (18, 18, 18), -1)
                            cv2.putText(frame, f"1080p distance: {distance_cm:.1f} cm", (22, 42),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2, cv2.LINE_AA)
                            cv2.putText(frame, f"IPD px: {ipd_px:.1f}", (22, 72),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Position px: {pos_x:+.0f}, {pos_y:+.0f}", (22, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Position norm: {pos_x_norm:+.2f}, {pos_y_norm:+.2f}", (22, 126),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Angle deg: {angle_x:+.1f}, {angle_y:+.1f}", (22, 152),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Frame: {w}x{h}", (22, 178),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Camera {camera_id}", (22, 204),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (10, 10), (275, 58), (18, 18, 18), -1)
                    cv2.putText(frame, "No face detected", (22, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: self._set_preview_status(msg[:28], self.DARK["danger"]))
        finally:
            if mesh is not None:
                mesh.close()
            if cap is not None:
                cap.release()
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass
            if self._preview_stop is not None:
                self._preview_stop.set()
            self.after(0, lambda: self._set_preview_status("Closed"))

    def _on_cancel(self, _event=None):
        self._stop_setup_preview()
        self.destroy()

    def _on_start(self, _event=None, tutorial=True):
        self._stop_setup_preview()
        self.result = {
            "camera":  self._camera_var.get(),
            "points":  self._points_var.get(),
            "samples": self._samples_var.get(),
            "ema":     self._ema_var.get() if self._ema_on.get() else 1.0,
            "pnoise":  self._pnoise_var.get(),
            "mnoise":  self._mnoise_var.get(),
            "dwell_mode": self._dwell_mode_var.get(),
            "ui_layout": self._ui_layout_var.get(),
            "camera_window": self._camera_window_var.get(),
            "camera_debug": self._camera_debug_var.get(),
            "distance_panel": self._distance_var.get(),
            "tutorial": tutorial,
        }
        self.destroy()


# =============================================================================
#  Helpers
# =============================================================================

# =============================================================================
#  Main
# =============================================================================

def main():
    # ── Show launcher UI ──────────────────────────────────────────────────────
    launcher = LauncherUI()
    launcher.mainloop()

    if launcher.result is None:
        print("[Info] Launcher cancelled.")
        sys.exit(0)

    cfg = launcher.result
    print("=" * 60)
    print("  GAZE-BASED DIGITAL KEYBOARD")
    print(f"  Points: {cfg['points']}  |  Samples: {cfg['samples']}  |  "
          f"EMA: {cfg['ema']:.2f}  |  Camera: {cfg['camera']}  |  "
          f"Dwell: {cfg['dwell_mode']}  |  UI: {cfg['ui_layout']}")
    print("=" * 60)

    # ── Deferred imports (avoid slowing down launcher) ────────────────────────
    from gaze_tracker2 import GazeTrackerApp
    import config
    from model import ngram_model
    from generate_flores_rules import generate_if_missing as _ensure_flores
    from config import FILIPINO_DATASET_FILE, ENGLISH_DATASET_FILE, NGRAM_CACHE_FILE

    config.DWELL_MODE = cfg["dwell_mode"]

    def _ensure_datasets():
        missing = []
        if not os.path.exists(FILIPINO_DATASET_FILE):
            missing.append(("Filipino", FILIPINO_DATASET_FILE, "generate_dataset"))
        if not os.path.exists(ENGLISH_DATASET_FILE):
            missing.append(("English", ENGLISH_DATASET_FILE, "generate_dataset_english"))
        if not missing:
            return
        try:
            import transformers  # noqa
        except ImportError:
            print("'transformers' not installed. pip install transformers torch")
            sys.exit(1)
        for label, path, module_name in missing:
            print(f"Generating {label} dataset...")
            module = __import__(module_name)
            module.generate(output_file=path)
        if os.path.exists(NGRAM_CACHE_FILE):
            os.remove(NGRAM_CACHE_FILE)

    def _rebuild_datasets():
        try:
            import transformers  # noqa
        except ImportError:
            print("'transformers' not installed. pip install transformers torch")
            sys.exit(1)
        for label, path, module_name in [
            ("Filipino", FILIPINO_DATASET_FILE, "generate_dataset"),
            ("English", ENGLISH_DATASET_FILE, "generate_dataset_english"),
        ]:
            print(f"Regenerating {label} dataset...")
            module = __import__(module_name)
            module.generate(output_file=path)
        if os.path.exists(NGRAM_CACHE_FILE):
            os.remove(NGRAM_CACHE_FILE)

    # ── Pre-load datasets & model ─────────────────────────────────────────────
    _ensure_datasets()
    _ensure_flores()

    if not os.path.exists(NGRAM_CACHE_FILE):
        print("Model cache missing — forcing dataset regeneration first.")
        _rebuild_datasets()

    if not ngram_model.load_cache():
        print("Building n-gram model from datasets...")
        if not ngram_model.train_from_builtin():
            print("Failed to build n-gram model: datasets were not available.")
            sys.exit(1)
        ngram_model.save_cache()
    ngram_model.load_user_learning()

    stop_gaze = threading.Event()
    gaze_thread = None

    # ── Build gaze tracker ────────────────────────────────────────────────────
    tracker = GazeTrackerApp(
        camera_id  = cfg["camera"],
        num_points = cfg["points"],
        spp        = cfg["samples"],
        ema_alpha  = cfg["ema"],
        pnoise     = cfg["pnoise"],
        mnoise     = cfg["mnoise"],
        show_camera_window = cfg["camera_window"],
        debug_landmarks    = cfg["camera_debug"],
        show_distance      = cfg["distance_panel"],
        tutorial_enabled   = cfg["tutorial"],
    )

    # ── Phase 1: Calibration on main thread (required on macOS) ──────────────
    print("  Starting calibration...")
    ok = tracker.calibrate()
    if not ok:
        print("[Info] Calibration cancelled.")
        sys.exit(0)
    tracker._mouse_ctrl = True
    print("\n✓ Calibration complete — launching keyboard...\n")

    # ── Phase 2: Tracking in background thread (no OpenCV GUI) ───────────────
    def start_tracking():
        nonlocal gaze_thread
        tracker._tracking_error = None
        gaze_thread = threading.Thread(
            target=tracker.track,
            kwargs={"stop_event": stop_gaze},
            daemon=True,
        )
        gaze_thread.start()
        print("[Info] Tracking thread started.")

    def stop_tracking():
        stop_gaze.set()
        if gaze_thread and gaze_thread.is_alive():
            gaze_thread.join(timeout=2.0)

    start_tracking()

    # ── Launch Tkinter keyboard on main thread ────────────────────────────────
    from ui import FilipinoKeyboard

    app = FilipinoKeyboard(ui_layout=cfg["ui_layout"], gaze_tracking_active=True)

    def on_close():
        stop_tracking()
        app.destroy()

    def quit_session(_event=None):
        """Q: close keyboard and stop gaze tracking from anywhere in Tk."""
        on_close()
        return "break"

    def toggle_mouse_control(_event=None):
        """X: pause/resume gaze-driven mouse movement without closing the app."""
        tracker._mouse_ctrl = not tracker._mouse_ctrl
        state = "ON" if tracker._mouse_ctrl else "OFF"
        app.status_bar.config(text=f"Gaze mouse control {state}")
        return "break"

    def recalibrate(_event=None):
        """
        R: recalibrate while the keyboard is open.
        Calibration must run on the main thread on macOS, so this callback stops
        the tracking thread, opens calibration, then restarts tracking.
        """
        nonlocal stop_gaze
        app.status_bar.config(text="Recalibrating gaze...")
        app.update_idletasks()

        stop_tracking()
        stop_gaze = threading.Event()

        ok = tracker.calibrate()
        if ok:
            tracker._mouse_ctrl = True
            start_tracking()
            app.status_bar.config(text="Recalibration complete | gaze tracking active")
        else:
            app.status_bar.config(text="Recalibration cancelled | closing session")
            on_close()
        return "break"

    def monitor_tracking():
        current_status = app.status_bar.cget("text")
        can_update_status = current_status.startswith((
            "Gaze tracking",
            "Gaze active",
            "Gaze-based keyboard ready",
            "Recalibration",
        ))
        if tracker._tracking_error:
            app.status_bar.config(text=f"Gaze tracking stopped: {tracker._tracking_error}")
        elif gaze_thread and not gaze_thread.is_alive():
            app.status_bar.config(text="Gaze tracking stopped")
        else:
            if can_update_status:
                state = "ON" if tracker._mouse_ctrl else "OFF"
                app.status_bar.config(
                    text=(
                        f"Gaze active | mouse {state} | frames {tracker._tracking_frames} | "
                        f"faces {tracker._tracking_faces} | points {tracker._tracking_predictions} | "
                        f"moves {tracker._mouse_moves} | pyauto {'OK' if tracker.pyautogui_ok else 'NO'}"
                    )
                )
            app.after(3000, monitor_tracking)

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.bind_all("<KeyPress-q>", quit_session)
    app.bind_all("<KeyPress-Q>", quit_session)
    app.bind_all("<KeyPress-x>", toggle_mouse_control)
    app.bind_all("<KeyPress-X>", toggle_mouse_control)
    app.bind_all("<KeyPress-r>", recalibrate)
    app.bind_all("<KeyPress-R>", recalibrate)
    app.after(1000, monitor_tracking)
    app.mainloop()

    # Cleanup
    stop_tracking()
    print("[Info] Application closed.")


if __name__ == "__main__":
    main()
