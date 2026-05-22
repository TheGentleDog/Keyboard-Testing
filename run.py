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
import math
import ctypes
import time
import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageTk, ImageSequence
except ImportError:
    Image = ImageDraw = ImageFilter = ImageTk = ImageSequence = None

# ── Make keyboard modules importable ─────────────────────────────────────────
KEYBOARD_DIR = os.path.join(os.path.dirname(__file__), "Bench", "Cutted_File", "files")
sys.path.insert(0, KEYBOARD_DIR)
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
TANAW_LOGO_PATH = os.path.join(ASSETS_DIR, "tanaw_logo.gif")


# =============================================================================
#  Launcher UI
# =============================================================================

def _colorref(hex_color):
    """Convert #RRGGBB to Windows COLORREF 0x00BBGGRR."""
    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return b << 16 | g << 8 | r


def _set_dark_title_bar(window, bg="#252628", fg="#ffffff"):
    """Ask Windows 10/11 to draw a dark native title bar."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        value = ctypes.c_int(1)
        # 20 is supported by recent Windows 10/11; 19 is the older fallback.
        for attribute in (20, 19):
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                attribute,
                ctypes.byref(value),
                ctypes.sizeof(value),
            )

        caption = ctypes.c_int(_colorref(bg))
        text = ctypes.c_int(_colorref(fg))
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 36, ctypes.byref(text), ctypes.sizeof(text))
    except Exception:
        pass


def _show_borderless_window_in_taskbar(window):
    """Give a borderless Tk window a normal Windows taskbar button."""
    if sys.platform != "win32":
        return
    try:
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id()) or window.winfo_id()
        exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        exstyle &= ~0x00000080  # WS_EX_TOOLWINDOW
        exstyle |= 0x00040000   # WS_EX_APPWINDOW
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, exstyle)
        window.withdraw()
        window.after(10, window.deiconify)
    except Exception:
        pass


class WelcomeUI(tk.Tk):
    """
    First screen shown to the user.
    Clicking Start continues to the existing setup launcher.
    """

    BG = "#1b1b1b"
    STAGE_BG = "#0B0D0F"
    GRADIENT = "#98A5B3"
    PANEL = "#242728"
    TEXT = "#f3f3f3"
    SUBTEXT = "#b9bcbc"
    BUTTON = "#111416"
    BUTTON_HOVER = "#1B2020"
    BUTTON_BORDER = "#E8ECEC"

    def __init__(self):
        super().__init__()
        self.result = None

        self.title("Gaze Keyboard")
        self.overrideredirect(True)
        self.resizable(False, False)
        self.configure(bg=self.BG)
        self._gradient_phase = 0
        self._gradient_bounds = None
        self._gradient_image_id = None
        self._gradient_photo = None
        self._button_photo = None
        self._button_hover_photo = None
        self._logo_frames = []
        self._logo_frame_index = 0
        self._logo_item = None
        self._logo_job = None
        self._animate_job = None
        self._fade_job = None
        self._typewriter_job = None
        self._subtitle_text = "A Gaze-based Digital Keyboard Interface"
        self._subtitle_index = 0
        self._caret_visible = True
        self._start_bounds = None

        W, H = 900, 560
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")
        self.after(0, lambda: _set_dark_title_bar(self, self.STAGE_BG, "#ffffff"))

        self.canvas = tk.Canvas(self, width=W, height=H, bg=self.BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self._build(W, H)

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Return>", self._on_start)
        self.bind("<Escape>", self._on_cancel)
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))
        self.after(250, lambda: _show_borderless_window_in_taskbar(self))

    def _build(self, width, height):
        c = self.canvas

        # Fill the whole client area so there are no edge gaps.
        c.create_rectangle(0, 0, width, height, fill=self.STAGE_BG, outline="")

        margin = 0
        stage_left = margin
        stage_top = margin
        stage_right = width
        stage_bottom = height
        c.create_rectangle(stage_left, stage_top, stage_right, stage_bottom,
                           fill=self.STAGE_BG, outline="", width=0)
        self._gradient_bounds = (stage_left, stage_top, stage_right, stage_bottom)
        if Image is not None:
            self._gradient_photo = self._render_angular_gradient(width, height, 0)
            self._gradient_image_id = c.create_image(0, 0, anchor="nw", image=self._gradient_photo)

        self._build_window_controls(width)
        self.canvas.bind("<ButtonPress-1>", self._start_window_drag)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._drag_window)
        self.canvas.bind("<ButtonRelease-1>", self._stop_window_drag)

        nav_y = 68
        self._logo_frames = self._load_logo_frames(TANAW_LOGO_PATH, 34, 24)
        if self._logo_frames:
            self._logo_item = c.create_image(24, 28, image=self._logo_frames[0][0])
            self._animate_logo()
        c.create_text(112, nav_y, text="About us", fill="#d6d7d8",
                      font=("Krona One", 10), anchor="w")
        c.create_text(width // 2, nav_y, text="Home", fill="#d6d7d8",
                      font=("Krona One", 10))
        c.create_text(width - 112, nav_y, text="Help", fill="#d6d7d8",
                      font=("Krona One", 10), anchor="e")
        c.create_text(width - 32, height - 30, text="SeenByEveryone", fill="#d6d7d8",
                      font=("Krona One", 10), anchor="e")
        center_x = width // 2
        content_y = int(height * 0.43)

        self.welcome_item = c.create_text(center_x, content_y, text="",
                                          fill=self.TEXT, font=("Krona One", 27))
        self.subtitle_item = c.create_text(center_x, content_y + 34, text="",
                                           fill="#c2c4c5", font=("Actor", 16))

        btn_w, btn_h = 122, 34
        x1 = center_x - btn_w // 2
        y1 = content_y + 74
        x2 = x1 + btn_w
        y2 = y1 + btn_h
        self._start_bounds = (x1, y1, x2, y2)
        if Image is not None:
            self._button_photo = self._render_pill_button(btn_w, btn_h, self.BUTTON, opacity=0.0)
            self._button_hover_photo = self._render_pill_button(btn_w, btn_h, self.BUTTON_HOVER)
            self.start_btn_image = c.create_image(x1, y1, anchor="nw", image=self._button_photo)
            self.start_btn_parts = [self.start_btn_image]
        else:
            self.start_btn_parts = [
                c.create_rectangle(x1, y1, x2, y2, fill=self.BUTTON,
                                   outline=self.BUTTON_BORDER, width=1),
            ]
        self.start_btn_text = c.create_text(center_x, y1 + btn_h / 2, text="START",
                                            fill="", font=("Segoe UI", 10, "bold"))
        for item in [*self.start_btn_parts, self.start_btn_text]:
            c.tag_bind(item, "<Button-1>", self._on_start)
            c.tag_bind(item, "<Enter>", self._on_button_enter)
            c.tag_bind(item, "<Leave>", self._on_button_leave)
        if Image is not None:
            self._animate_gradient()
        self._fade_job = self.after(1000, lambda: self._fade_intro(0))
        self._typewriter_job = self.after(2000, self._type_subtitle)

    def _load_logo_frames(self, path, max_width, max_height):
        if Image is None or ImageTk is None or ImageSequence is None or not os.path.exists(path):
            return []

        frames = []
        try:
            with Image.open(path) as gif:
                for frame in ImageSequence.Iterator(gif):
                    img = frame.convert("RGBA")
                    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                    duration = frame.info.get("duration", gif.info.get("duration", 80))
                    frames.append((ImageTk.PhotoImage(img), max(35, int(duration or 80))))
        except Exception:
            return []
        return frames

    def _animate_logo(self):
        if not self._logo_frames or self._logo_item is None:
            return

        frame, delay = self._logo_frames[self._logo_frame_index]
        self.canvas.itemconfigure(self._logo_item, image=frame)
        self._logo_frame_index = (self._logo_frame_index + 1) % len(self._logo_frames)
        self._logo_job = self.after(delay, self._animate_logo)

    def _render_pill_button(self, width, height, fill, opacity=1.0):
        scale = 4
        img = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        border = self._hex_to_rgb(self.BUTTON_BORDER)
        fill_rgb = self._hex_to_rgb(fill)
        alpha = max(0, min(255, round(255 * opacity)))
        rect = (2 * scale, 2 * scale, (width - 2) * scale, (height - 2) * scale)
        draw.rounded_rectangle(
            rect,
            radius=(height // 2 - 1) * scale,
            fill=(*fill_rgb, alpha),
            outline=(*border, alpha),
            width=scale,
        )
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _fade_intro(self, step):
        steps = 18
        amount = min(1.0, step / steps)
        eased = 1.0 - ((1.0 - amount) ** 3)
        if step == 0:
            self.canvas.itemconfigure(self.welcome_item, text="Welcome to TANAW")
            self.canvas.itemconfigure(self.start_btn_text, fill="#ffffff")
        self.canvas.itemconfigure(
            self.welcome_item,
            fill=self._blend(self._hex_to_rgb(self.STAGE_BG), self._hex_to_rgb(self.TEXT), eased),
        )
        self.canvas.itemconfigure(
            self.start_btn_text,
            fill=self._blend(self._hex_to_rgb(self.STAGE_BG), (255, 255, 255), eased),
        )
        if Image is not None and hasattr(self, "start_btn_image"):
            self._button_photo = self._render_pill_button(122, 34, self.BUTTON, opacity=eased)
            self.canvas.itemconfigure(self.start_btn_image, image=self._button_photo)

        if step < steps:
            self._fade_job = self.after(28, lambda: self._fade_intro(step + 1))
        else:
            self._fade_job = None

    def _build_window_controls(self, width):
        c = self.canvas
        min_x = width - 46
        close_x = width - 24
        y = 28
        self.minimize_btn = c.create_text(min_x, y - 1, text="-",
                                          fill="#d7d9db", font=("Segoe UI", 15, "bold"))
        self.close_btn = c.create_text(close_x, y, text="x",
                                       fill="#d7d9db", font=("Segoe UI", 13, "bold"))

        c.tag_bind(self.minimize_btn, "<Button-1>", self._on_minimize)
        c.tag_bind(self.close_btn, "<Button-1>", self._on_cancel)
        c.tag_bind(self.minimize_btn, "<Enter>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#ffffff"))
        c.tag_bind(self.minimize_btn, "<Leave>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#d7d9db"))
        c.tag_bind(self.close_btn, "<Enter>", lambda _e: c.itemconfigure(self.close_btn, fill="#ff6b6b"))
        c.tag_bind(self.close_btn, "<Leave>", lambda _e: c.itemconfigure(self.close_btn, fill="#d7d9db"))

    def _build_title_bar(self, width):
        c = self.canvas
        bar_h = 42
        title_hitbox = c.create_rectangle(0, 0, width - 120, bar_h, fill=self.STAGE_BG, outline="")
        c.addtag_withtag("titlebar", title_hitbox)
        c.create_rectangle(width - 120, 0, width, bar_h, fill=self.STAGE_BG, outline="")
        c.create_line(0, bar_h, width, bar_h, fill="#171B1F")
        c.create_text(22, bar_h // 2, text="Gaze Keyboard", fill="#d7d9db",
                      font=("Segoe UI", 10, "bold"), anchor="w")

        min_x = width - 92
        close_x = width - 46
        self.minimize_btn = c.create_text(min_x, bar_h // 2 - 1, text="—",
                                          fill="#d7d9db", font=("Segoe UI", 16))
        self.close_btn = c.create_text(close_x, bar_h // 2, text="×",
                                       fill="#d7d9db", font=("Segoe UI", 15))

        c.tag_bind(self.minimize_btn, "<Button-1>", self._on_minimize)
        c.tag_bind(self.close_btn, "<Button-1>", self._on_cancel)
        c.tag_bind(self.minimize_btn, "<Enter>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#ffffff"))
        c.tag_bind(self.minimize_btn, "<Leave>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#d7d9db"))
        c.tag_bind(self.close_btn, "<Enter>", lambda _e: c.itemconfigure(self.close_btn, fill="#ff6b6b"))
        c.tag_bind(self.close_btn, "<Leave>", lambda _e: c.itemconfigure(self.close_btn, fill="#d7d9db"))

        c.tag_bind("titlebar", "<ButtonPress-1>", self._start_window_drag)
        c.tag_bind("titlebar", "<B1-Motion>", self._drag_window)

    def _start_window_drag(self, event):
        if self._is_over_start(event):
            self._dragging_window = False
            return
        self._dragging_window = event.y < 105 and event.x < self.winfo_width() - 85
        if not self._dragging_window:
            return
        self._drag_offset_x = event.x
        self._drag_offset_y = event.y

    def _drag_window(self, event):
        if not getattr(self, "_dragging_window", False):
            return
        x = self.winfo_pointerx() - self._drag_offset_x
        y = self.winfo_pointery() - self._drag_offset_y
        self.geometry(f"+{x}+{y}")

    def _stop_window_drag(self, _event=None):
        self._dragging_window = False

    def _is_over_start(self, event):
        if self._start_bounds is None:
            return False
        x1, y1, x2, y2 = self._start_bounds
        return x1 <= event.x <= x2 and y1 <= event.y <= y2

    def _on_canvas_click(self, event):
        if self._is_over_start(event):
            self._on_start(event)
            return "break"

    def _on_minimize(self, _event=None):
        self.overrideredirect(False)
        self.iconify()
        self.after(50, lambda: self.overrideredirect(True))

    def _type_subtitle(self):
        if self._subtitle_index <= len(self._subtitle_text):
            text = self._subtitle_text[:self._subtitle_index]
            caret = "|" if self._caret_visible else ""
            self.canvas.itemconfigure(self.subtitle_item, text=f"{text}{caret}")
            self._subtitle_index += 1
            delay = 95 if self._subtitle_index in (2, 9, 21, 30) else 42
            self._typewriter_job = self.after(delay, self._type_subtitle)
        else:
            self._blink_subtitle_caret()

    def _blink_subtitle_caret(self):
        self._caret_visible = not self._caret_visible
        caret = "|" if self._caret_visible else ""
        self.canvas.itemconfigure(self.subtitle_item, text=f"{self._subtitle_text}{caret}")
        self._typewriter_job = self.after(480, self._blink_subtitle_caret)

    def _render_angular_gradient(self, width, height, phase, offset_x=0, offset_y=0,
                                 full_width=None, full_height=None):
        full_width = full_width or width
        full_height = full_height or height
        small_w = 260
        small_h = max(1, round(full_height * (small_w / full_width)))
        base = self._hex_to_rgb(self.STAGE_BG)
        accent = self._hex_to_rgb(self.GRADIENT)
        img = Image.new("RGB", (small_w, small_h), self.STAGE_BG)
        overlay = Image.new("RGBA", (small_w, small_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        cx = small_w / 2
        cy = small_h / 2
        radius = math.hypot(small_w, small_h)
        rotation = phase * 0.016
        bands = [
            (0.0, 96, 0.27),
            (math.pi * 0.72, 72, 0.17),
            (math.pi * 1.34, 86, 0.14),
        ]

        for offset, width_deg, opacity in bands:
            center = rotation + offset
            half = math.radians(width_deg) / 2
            points = [
                (cx, cy),
                (cx + math.cos(center - half) * radius, cy + math.sin(center - half) * radius),
                (cx + math.cos(center + half) * radius, cy + math.sin(center + half) * radius),
            ]
            alpha = round(255 * opacity)
            draw.polygon(points, fill=(*accent, alpha))

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=28))
        img = Image.alpha_composite(img.convert("RGBA"), overlay)

        shade = Image.new("RGBA", (small_w, small_h), (0, 0, 0, 0))
        shade_draw = ImageDraw.Draw(shade, "RGBA")
        for y in range(small_h):
            distance = abs((y / small_h) - 0.52)
            alpha = round(52 * max(0.0, 1.0 - distance * 2.2))
            shade_draw.line((0, y, small_w, y), fill=(*base, alpha))
        img = Image.alpha_composite(img, shade)

        if offset_x or offset_y or full_width != width or full_height != height:
            sx = small_w / full_width
            sy = small_h / full_height
            left = int(round(offset_x * sx))
            top = int(round(offset_y * sy))
            right = int(round((offset_x + width) * sx))
            bottom = int(round((offset_y + height) * sy))
            img = img.crop((left, top, max(left + 1, right), max(top + 1, bottom)))

        img = img.resize((width, height), Image.Resampling.BICUBIC)
        return ImageTk.PhotoImage(img)

    def _animate_gradient(self):
        if not self._gradient_bounds or self._gradient_image_id is None:
            return

        self._gradient_phase += 1
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        self._gradient_photo = self._render_angular_gradient(width, height, self._gradient_phase)
        self.canvas.itemconfigure(self._gradient_image_id, image=self._gradient_photo)
        self._animate_job = self.after(85, self._animate_gradient)

    @staticmethod
    def _hex_to_rgb(color):
        color = color.lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _blend(start, end, amount):
        rgb = tuple(round(start[i] + (end[i] - start[i]) * amount) for i in range(3))
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _on_button_enter(self, _event=None):
        self.canvas.configure(cursor="hand2")
        if Image is not None and hasattr(self, "start_btn_image"):
            self.canvas.itemconfigure(self.start_btn_image, image=self._button_hover_photo)
        else:
            for item in self.start_btn_parts:
                self.canvas.itemconfigure(item, fill=self.BUTTON_HOVER)

    def _on_button_leave(self, _event=None):
        self.canvas.configure(cursor="")
        if Image is not None and hasattr(self, "start_btn_image"):
            self.canvas.itemconfigure(self.start_btn_image, image=self._button_photo)
        else:
            for item in self.start_btn_parts:
                self.canvas.itemconfigure(item, fill=self.BUTTON)

    def _on_cancel(self, _event=None):
        self._finish(None)

    def _on_start(self, _event=None):
        self._finish("start")

    def _finish(self, result):
        self._cancel_welcome_jobs()
        self.result = result
        self.withdraw()
        self.quit()

    def _cancel_welcome_jobs(self):
        for job in (self._animate_job, self._logo_job, self._fade_job, self._typewriter_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._animate_job = None
        self._logo_job = None
        self._fade_job = None
        self._typewriter_job = None


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
    STAGE_BG = WelcomeUI.STAGE_BG
    GRADIENT = WelcomeUI.GRADIENT

    def __init__(self):
        super().__init__()
        self.result = None
        self._preview_stop = None
        self._preview_thread = None
        self._gradient_phase = 0
        self._gradient_photo = None
        self._gradient_image_id = None
        self._content_gradient_photo = None
        self._content_gradient_id = None
        self._content_canvas = None
        self._settings_scroll_y = 0
        self._settings_content_height = 1
        self._animate_job = None
        self._dragging_window = False
        d = self.DARK

        self.title("Gaze Keyboard — Launcher")
        self.overrideredirect(True)
        self.resizable(False, False)
        self.configure(bg=self.STAGE_BG)

        # ── Center window ─────────────────────────────────────────
        W, H = 843, 555
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{(sw-W)//2}+{(sh-H)//2}")

        self._build(d)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.lift()
        self.focus_force()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))
        self.after(250, lambda: _show_borderless_window_in_taskbar(self))

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

    def _build_legacy(self, d):
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
        #  CALIBRATIONq
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
        self._dwell_mode_var = tk.StringVar(value="async")
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
        self._ui_layout_var = tk.StringVar(value="ui2")
        layout_frame = tk.Frame(layout, bg=d["card"])
        layout_frame.pack(fill="x", pady=3)
        tk.Label(layout_frame, text="Keyboard UI", bg=d["card"], fg=d["text"],
                 font=("Segoe UI", 11), anchor="w", width=22).pack(side="left")
        for value, label in [("ui2", "Default"), ("qwerty", "QWERTY")]:
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

    def _build(self, d):
        self._camera_var = tk.IntVar(value=0)
        self._camera_window_var = tk.BooleanVar(value=True)
        self._camera_debug_var = tk.BooleanVar(value=True)
        self._distance_var = tk.BooleanVar(value=True)
        self._ui_layout_var = tk.StringVar(value="ui2")
        self._language_english_var = tk.BooleanVar(value=True)
        self._language_tagalog_var = tk.BooleanVar(value=True)
        self._dwell_mode_var = tk.StringVar(value="async")
        self._points_var = tk.IntVar(value=9)
        self._samples_var = tk.IntVar(value=90)
        self._pnoise_var = tk.DoubleVar(value=0.0100)
        self._mnoise_var = tk.DoubleVar(value=5.5)
        self._ema_on = tk.BooleanVar(value=True)
        self._ema_var = tk.DoubleVar(value=0.30)

        self._launcher_canvas = tk.Canvas(self, bg=self.STAGE_BG, highlightthickness=0, bd=0)
        self._launcher_canvas.pack(fill="both", expand=True)
        if Image is not None:
            self._gradient_photo = self._render_launcher_gradient(843, 555, 0)
            self._gradient_image_id = self._launcher_canvas.create_image(
                0, 0, anchor="nw", image=self._gradient_photo
            )
            self._animate_launcher_gradient()
        self._build_launcher_controls()
        self._launcher_canvas.bind("<ButtonPress-1>", self._start_window_drag)
        self._launcher_canvas.bind("<B1-Motion>", self._drag_window)
        self._launcher_canvas.bind("<ButtonRelease-1>", self._stop_window_drag)

        nav_bg = "#050607"
        launcher_w, launcher_h = 843, 555
        nav_x, nav_y, nav_w, nav_h = 14, 40, 225, 501
        content_x, content_y, content_w, content_h = 253, 40, 576, 501
        content_wrap = tk.Frame(self._launcher_canvas, bg="#070809", width=576, height=501)
        self._launcher_canvas.create_window(
            content_x, content_y, anchor="nw", width=content_w, height=content_h, window=content_wrap
        )
        content_wrap.pack_propagate(False)
        self._content_gradient_view = (content_x, content_y, launcher_w, launcher_h)

        if Image is not None:
            self._nav_panel_photo = self._render_alpha_panel(nav_w, nav_h, nav_bg, 0.3)
            self._launcher_canvas.create_image(nav_x, nav_y, anchor="nw", image=self._nav_panel_photo)
        else:
            self._launcher_canvas.create_rectangle(
                nav_x, nav_y, nav_x + nav_w, nav_y + nav_h,
                outline="", fill=nav_bg, stipple="gray25",
            )

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Launcher.Vertical.TScrollbar",
            troughcolor="#070809",
            background="#050607",
            bordercolor="#070809",
            arrowcolor="#050607",
            darkcolor="#050607",
            lightcolor="#050607",
            relief="flat",
            width=6,
        )
        style.map(
            "Launcher.Vertical.TScrollbar",
            background=[("active", "#111315"), ("pressed", "#1c1f22")],
        )
        canvas = tk.Canvas(content_wrap, bg="#070809", highlightthickness=0, bd=0)
        self._content_canvas = canvas
        if Image is not None:
            self._content_gradient_photo = self._render_launcher_gradient(
                content_w, content_h, self._gradient_phase,
                offset_x=content_x, offset_y=content_y,
                full_width=launcher_w, full_height=launcher_h,
            )
            self._content_gradient_id = canvas.create_image(
                0, 0, anchor="nw", image=self._content_gradient_photo
            )
        canvas.pack(fill="both", expand=True)

        sections = {}
        section_windows = {}
        section_order = []
        nav_items = {}
        nav_images = {}
        self._settings_canvas = canvas
        self._settings_sections = sections
        panel_bg = "#070809"
        panel_text = "#f3f3f3"
        panel_muted = "#c6c9cc"

        def sync_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_body(event):
            if self._content_gradient_id is not None:
                canvas.tag_lower(self._content_gradient_id)
            self._layout_canvas_sections(canvas, sections, section_windows, section_order)

        def set_active(key):
            self._active_setup_section = key
            for item_key, item in nav_items.items():
                if Image is not None:
                    image = nav_images[item_key]["active" if item_key == key else "idle"]
                    self._launcher_canvas.itemconfigure(item["bg"], image=image)
                    self._launcher_canvas.itemconfigure(
                        item["text"], fill="#ffffff" if item_key == key else d["text"]
                    )
                else:
                    self._launcher_canvas.itemconfigure(
                        item["bg"], fill="#15181c" if item_key == key else nav_bg
                    )
                    self._launcher_canvas.itemconfigure(
                        item["text"], fill="#ffffff" if item_key == key else d["text"]
                    )

        def sync_active_from_scroll():
            if not sections:
                return
            marker = self._settings_scroll_y + max(56, int(canvas.winfo_height() * 0.18))
            active_key = self._active_setup_section
            for item_key, section_top in sorted(sections.items(), key=lambda item: item[1]):
                if section_top <= marker:
                    active_key = item_key
                else:
                    break
            if active_key != self._active_setup_section:
                set_active(active_key)

        def mousewheel(event):
            self._scroll_canvas_settings(canvas, sections, int(-1 * (event.delta / 120)) * 42)
            sync_active_from_scroll()
            return "break"

        scrollbar_drag = {"offset": 0}

        def scrollbar_metrics():
            view_h = max(1, canvas.winfo_height())
            content_h = max(1, self._settings_content_height)
            top = 18
            bottom = view_h - 18
            track_h = max(1, bottom - top)
            if content_h <= view_h:
                return top, track_h, 0, 1, top
            thumb_h = max(42, track_h * (view_h / content_h))
            max_scroll = max(1, content_h - view_h)
            travel = max(1, track_h - thumb_h)
            thumb_y = top + travel * (self._settings_scroll_y / max_scroll)
            return top, thumb_h, max_scroll, travel, thumb_y

        def drag_scrollbar(event):
            top, thumb_h, max_scroll, travel, _thumb_y = scrollbar_metrics()
            if max_scroll <= 0:
                self._settings_scroll_y = 0
                self._draw_canvas_settings(canvas, sections, keep_scroll=True)
                sync_active_from_scroll()
                return "break"
            pct = (event.y - top - scrollbar_drag["offset"]) / travel
            self._settings_scroll_y = int(round(max(0, min(max_scroll, pct * max_scroll))))
            self._draw_canvas_settings(canvas, sections, keep_scroll=True)
            sync_active_from_scroll()
            return "break"

        def start_scrollbar_drag(event):
            _top, thumb_h, _max_scroll, _travel, thumb_y = scrollbar_metrics()
            if thumb_y <= event.y <= thumb_y + thumb_h:
                scrollbar_drag["offset"] = event.y - thumb_y
            else:
                scrollbar_drag["offset"] = thumb_h / 2
            canvas.configure(cursor="sb_v_double_arrow")
            canvas.bind("<B1-Motion>", drag_scrollbar)
            canvas.bind("<ButtonRelease-1>", stop_scrollbar_drag)
            return drag_scrollbar(event)

        def stop_scrollbar_drag(_event=None):
            canvas.configure(cursor="")
            canvas.unbind("<B1-Motion>")
            canvas.unbind("<ButtonRelease-1>")
            return "break"

        def jump_to(key):
            self.update_idletasks()
            section = sections.get(key)
            if isinstance(section, (int, float)):
                self._settings_scroll_y = int(round(max(0, section - 12)))
                self._draw_canvas_settings(canvas, sections, keep_scroll=True)
                set_active(key)
                return
            section_window = section_windows.get(key)
            bbox = canvas.bbox("all")
            if section is None or section_window is None or not bbox:
                return
            scroll_height = max(1, bbox[3] - bbox[1])
            visible = max(1, canvas.winfo_height())
            max_fraction = max(0.0, (scroll_height - visible) / scroll_height)
            target_y = max(0, canvas.coords(section_window)[1] - 12)
            canvas.yview_moveto(min(max_fraction, target_y / scroll_height))
            set_active(key)

        def make_nav(key, text):
            index = len(nav_items)
            item_y = nav_y + 33 + index * 46
            item_x = nav_x + 10
            if Image is not None:
                nav_images[key] = {
                    "idle": self._render_nav_pill(206, 42, nav_bg, 12),
                    "hover": self._render_nav_pill(206, 42, "#111315", 12),
                    "active": self._render_nav_pill(206, 42, "#15181c", 12),
                }
                bg_item = self._launcher_canvas.create_image(
                    item_x, item_y, anchor="nw", image=nav_images[key]["idle"],
                    tags=(f"nav_{key}", "nav_item"),
                )
            else:
                bg_item = self._launcher_canvas.create_rectangle(
                    item_x, item_y, item_x + 206, item_y + 42,
                    outline="", fill=nav_bg, tags=(f"nav_{key}", "nav_item"),
                )
            text_item = self._launcher_canvas.create_text(
                item_x + 103, item_y + 21, text=text, fill=d["text"],
                font=("Segoe UI", 12), anchor="center", tags=(f"nav_{key}", "nav_item"),
            )
            self._launcher_canvas.tag_bind(f"nav_{key}", "<Button-1>", lambda _e, k=key: jump_to(k))
            if Image is not None:
                self._launcher_canvas.tag_bind(
                    f"nav_{key}", "<Enter>",
                    lambda _e, b=bg_item, k=key: (
                        self._launcher_canvas.configure(cursor="hand2"),
                        self._launcher_canvas.itemconfigure(b, image=nav_images[k]["hover"]),
                    ),
                )
            else:
                self._launcher_canvas.tag_bind(
                    f"nav_{key}", "<Enter>",
                    lambda _e, b=bg_item: (
                        self._launcher_canvas.configure(cursor="hand2"),
                        self._launcher_canvas.itemconfigure(b, fill="#111315"),
                    ),
                )
            self._launcher_canvas.tag_bind(
                f"nav_{key}", "<Leave>",
                lambda _e: (self._launcher_canvas.configure(cursor=""), set_active(self._active_setup_section)),
            )
            nav_items[key] = {"bg": bg_item, "text": text_item}

        def section(key, title):
            frame = tk.Frame(canvas, bg=panel_bg, bd=0)
            sections[key] = frame
            section_order.append(key)
            section_windows[key] = canvas.create_window(
                270, 0, anchor="nw", width=480, window=frame
            )
            tk.Label(frame, text=title, bg=panel_bg, fg=panel_text,
                     font=("Segoe UI", 10, "bold")).pack(anchor="center", pady=(13, 10))
            inner = tk.Frame(frame, bg=panel_bg)
            inner.pack(fill="x", padx=30, pady=(0, 24))
            return inner

        def choice(parent, text, variable, value):
            rb = tk.Radiobutton(parent, text=text, variable=variable, value=value,
                                bg=panel_bg, fg=panel_text, selectcolor="#25ba4a",
                                activebackground=panel_bg, activeforeground=panel_text,
                                font=("Segoe UI", 12))
            rb.pack(anchor="w", pady=5)
            return rb

        def check(parent, text, variable, command=None):
            cb = tk.Checkbutton(parent, text=text, variable=variable, command=command,
                                bg=panel_bg, fg=panel_text, selectcolor="#25ba4a",
                                activebackground=panel_bg, activeforeground=panel_text,
                                font=("Segoe UI", 12))
            cb.pack(anchor="w", pady=5)
            return cb

        def slider(parent, label, variable, from_, to, resolution, fmt):
            row = tk.Frame(parent, bg=panel_bg)
            row.pack(fill="x", pady=(18, 4))
            tk.Label(row, text=label, bg=panel_bg, fg=panel_text,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w")
            control = tk.Frame(row, bg=panel_bg)
            control.pack(fill="x", pady=(7, 0))
            value_lbl = tk.Label(control, text=fmt(variable.get()), bg=panel_bg,
                                 fg=panel_text, font=("Segoe UI", 10, "bold"), width=7)
            value_lbl.pack(side="right", padx=(8, 0))
            tk.Scale(control, from_=from_, to=to, resolution=resolution,
                     orient="horizontal", variable=variable, showvalue=False,
                     bg=panel_bg, fg=panel_text, troughcolor="#4f5052",
                     activebackground="#25ba4a", highlightthickness=0,
                     command=lambda v: value_lbl.config(text=fmt(float(v)))
                     ).pack(side="left", fill="x", expand=True)
            return row

        for key, title in [
            ("camera", "Camera Window"),
            ("layout", "Keyboard layout"),
            ("language", "Language"),
            ("dwell", "Dwell Mode"),
            ("calibration", "Calibration"),
            ("kalman", "Kalman Filter"),
            ("smoother", "Smoother"),
        ]:
            make_nav(key, title)

        def footer_button(key, label, y, fill, hover, callback):
            x1 = nav_x + 18
            w = nav_w - 36
            h = 34
            tag = f"footer_{key}"
            if Image is not None:
                normal_img = self._render_setup_action_button(w, h, fill)
                hover_img = self._render_setup_action_button(w, h, hover)
                nav_images[tag] = {"idle": normal_img, "hover": hover_img}
                rect = self._launcher_canvas.create_image(
                    x1, y, anchor="nw", image=normal_img, tags=(tag, "footer_button"),
                )
            else:
                rect = self._launcher_canvas.create_rectangle(
                    x1, y, x1 + w, y + h, outline="", fill=fill, tags=(tag, "footer_button"),
                )
            self._launcher_canvas.create_text(
                x1 + w / 2, y + h / 2, text=label, fill="#ffffff",
                font=("Segoe UI", 8, "bold"), anchor="center", tags=(tag, "footer_button"),
            )
            self._launcher_canvas.tag_bind(tag, "<Button-1>", callback)
            if Image is not None:
                self._launcher_canvas.tag_bind(
                    tag, "<Enter>",
                    lambda _e, r=rect, t=tag: (
                        self._launcher_canvas.configure(cursor="hand2"),
                        self._launcher_canvas.itemconfigure(r, image=nav_images[t]["hover"]),
                    ),
                )
                self._launcher_canvas.tag_bind(
                    tag, "<Leave>",
                    lambda _e, r=rect, t=tag: (
                        self._launcher_canvas.configure(cursor=""),
                        self._launcher_canvas.itemconfigure(r, image=nav_images[t]["idle"]),
                    ),
                )
            else:
                self._launcher_canvas.tag_bind(
                    tag, "<Enter>",
                    lambda _e, r=rect: (
                        self._launcher_canvas.configure(cursor="hand2"),
                        self._launcher_canvas.itemconfigure(r, fill=hover),
                    ),
                )
                self._launcher_canvas.tag_bind(
                    tag, "<Leave>",
                    lambda _e, r=rect: (
                        self._launcher_canvas.configure(cursor=""),
                        self._launcher_canvas.itemconfigure(r, fill=fill),
                    ),
                )

        footer_y = nav_y + nav_h - 92
        footer_button("tutorial", "Start with tutorial", footer_y, WelcomeUI.BUTTON, WelcomeUI.BUTTON_HOVER,
                      lambda e: self._on_start(e, tutorial=True))
        footer_button("skip", "Start without tutorial", footer_y + 42, WelcomeUI.BUTTON, WelcomeUI.BUTTON_HOVER,
                      lambda e: self._on_start(e, tutorial=False))

        self._draw_canvas_settings(canvas, sections)
        canvas.bind("<Configure>", lambda _e: self._draw_canvas_settings(canvas, sections, keep_scroll=True))
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        canvas.tag_bind("settings_scrollbar", "<Button-1>", start_scrollbar_drag)
        canvas.tag_bind("settings_scrollbar", "<Enter>", lambda _e: canvas.configure(cursor="sb_v_double_arrow"))
        canvas.tag_bind("settings_scrollbar", "<Leave>", lambda _e: canvas.configure(cursor=""))
        self._active_setup_section = "camera"
        self.after(50, lambda: jump_to("camera"))
        return

        camera = section("camera", "Camera Window")
        check(camera, "Show camera preview during tracking", self._camera_window_var)
        check(camera, "Show debug eye landmarks", self._camera_debug_var)
        check(camera, "Show distance panel", self._distance_var)
        preview_row = tk.Frame(camera, bg=panel_bg)
        preview_row.pack(fill="x", pady=(12, 0))
        self._preview_btn = tk.Label(preview_row, text="Open Preview", bg=d["accent"],
                                     fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                     padx=14, pady=7, cursor="hand2")
        self._preview_btn.pack(side="left")
        self._preview_btn.bind("<Button-1>", self._toggle_setup_preview)
        self._preview_btn.bind("<Enter>", lambda _e: self._preview_btn.config(bg=d["accent_hov"]))
        self._preview_btn.bind("<Leave>", lambda _e: self._sync_preview_button())
        self._preview_status = tk.Label(preview_row, text="Closed", bg=panel_bg,
                                        fg=panel_muted, font=("Segoe UI", 9))
        self._preview_status.pack(side="left", padx=10)

        layout = section("layout", "Keyboard Layout")
        choice(layout, "Default", self._ui_layout_var, "ui2")
        choice(layout, "QWERTY", self._ui_layout_var, "qwerty")

        language = section("language", "Language")
        check(language, "English", self._language_english_var)
        check(language, "Tagalog", self._language_tagalog_var)

        dwell = section("dwell", "Dwell Mode")
        choice(dwell, "Synchronous", self._dwell_mode_var, "sync")
        choice(dwell, "Asynchronous", self._dwell_mode_var, "async")

        calibration = section("calibration", "Calibration")
        tk.Label(calibration, text="Grid Points", bg=panel_bg, fg=panel_text,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        for pts, label in [(5, "5 (Fast)"), (9, "9 (Default)"),
                           (16, "16 (Precise)"), (25, "25 (Deadeye)")]:
            choice(calibration, label, self._points_var, pts)
        slider(calibration, "Samples/Points", self._samples_var, 20, 120, 1,
               lambda v: f"{int(float(v))}")

        kalman = section("kalman", "Kalman Filter")
        slider(kalman, "Process noise", self._pnoise_var, 1e-4, 0.1, 1e-4,
               lambda v: f"{float(v):.4f}")
        slider(kalman, "Measurement noise", self._mnoise_var, 1.0, 50.0, 0.5,
               lambda v: f"{float(v):.1f}")

        smoother = section("smoother", "Smoother")
        check(smoother, "Enable EMA smoother", self._ema_on, self._toggle_ema)
        self._ema_frame = tk.Frame(smoother, bg=panel_bg)
        self._ema_frame.pack(fill="x")
        slider(self._ema_frame, "EMA alpha", self._ema_var, 0.01, 1.0, 0.01,
               lambda v: f"{float(v):.2f}")

        canvas.bind("<Configure>", resize_body)
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        self._active_setup_section = "camera"
        self.after(50, lambda: (self._layout_canvas_sections(canvas, sections, section_windows, section_order), jump_to("camera")))

    def _set_launcher_defaults(self, _event=None):
        self._camera_var.set(0)
        self._camera_window_var.set(True)
        self._camera_debug_var.set(True)
        self._distance_var.set(True)
        self._ui_layout_var.set("ui2")
        self._language_english_var.set(True)
        self._language_tagalog_var.set(True)
        self._dwell_mode_var.set("async")
        self._points_var.set(9)
        self._samples_var.set(90)
        self._pnoise_var.set(0.0100)
        self._mnoise_var.set(5.5)
        self._ema_on.set(True)
        self._ema_var.set(0.30)
        self._toggle_ema()
        if hasattr(self, "_settings_canvas") and hasattr(self, "_settings_sections"):
            self._draw_canvas_settings(self._settings_canvas, self._settings_sections, keep_scroll=True)
        return "break"

    def _draw_canvas_settings(self, canvas, sections, keep_scroll=False):
        canvas.delete("settings_ui")
        sections.clear()
        self._settings_header_panels = []
        if self._content_gradient_id is not None:
            canvas.tag_lower(self._content_gradient_id)
        if not keep_scroll:
            self._settings_scroll_y = 0
        max_scroll = max(0, self._settings_content_height - max(1, canvas.winfo_height()))
        self._settings_scroll_y = int(round(max(0, min(self._settings_scroll_y, max_scroll))))

        x = 46
        width = max(360, canvas.winfo_width() - x - 70)
        y = 36 - self._settings_scroll_y
        text = "#f3f3f3"
        muted = "#c6c9cc"
        accent = self.DARK["accent"]
        green = "#25ba4a"

        def bind_click(tag, callback):
            canvas.tag_bind(tag, "<Button-1>", callback)
            canvas.tag_bind(tag, "<Enter>", lambda _e: canvas.configure(cursor="hand2"))
            canvas.tag_bind(tag, "<Leave>", lambda _e: canvas.configure(cursor=""))

        def title(label):
            canvas.create_text(
                x + width / 2, y_positions[0],
                text=label,
                fill=text,
                font=("Segoe UI", 10, "bold"),
                tags=("settings_ui",),
            )
            y_positions[0] += 38

        def checkbox(label, var, key):
            tag = f"settings_{key}"
            cy = y_positions[0]
            box = (x + 4, cy - 9, x + 22, cy + 9)
            outline = "#ffffff" if var.get() else muted
            canvas.create_rectangle(*box, outline=outline, fill="#101214", width=2,
                                    tags=("settings_ui", tag))
            if var.get():
                canvas.create_line(
                    x + 8, cy + 1,
                    x + 12, cy + 5,
                    x + 19, cy - 6,
                    fill="#ffffff",
                    width=2,
                    capstyle="round",
                    joinstyle="round",
                    tags=("settings_ui", tag),
                )
            canvas.create_text(x + 34, cy, text=label, fill=text, anchor="w",
                               font=("Segoe UI", 12), tags=("settings_ui", tag))
            bind_click(tag, lambda _e, v=var: (v.set(not v.get()), self._draw_canvas_settings(canvas, sections, True)))
            y_positions[0] += 42

        def radio(label, var, value, key):
            tag = f"settings_{key}_{value}"
            cy = y_positions[0]
            selected = var.get() == value
            canvas.create_oval(x + 5, cy - 7, x + 19, cy + 7,
                               outline="#ffffff" if selected else muted,
                               fill="#101214", width=2,
                               tags=("settings_ui", tag))
            if selected:
                canvas.create_oval(x + 9, cy - 3, x + 15, cy + 3, outline="", fill="#ffffff",
                                   tags=("settings_ui", tag))
            canvas.create_text(x + 28, cy, text=label, fill=text, anchor="w",
                               font=("Segoe UI", 12), tags=("settings_ui", tag))
            bind_click(tag, lambda _e, v=var, val=value: (v.set(val), self._draw_canvas_settings(canvas, sections, True)))
            y_positions[0] += 42

        def slider(label, var, min_value, max_value, fmt, key):
            y_positions[0] += 10
            canvas.create_text(x + 6, y_positions[0], text=label, fill=text, anchor="w",
                               font=("Segoe UI", 10, "bold"), tags=("settings_ui",))
            y_positions[0] += 30
            sx1 = x + 6
            sx2 = x + width - 90
            sy = y_positions[0]
            value = float(var.get())
            pct = (value - min_value) / (max_value - min_value)
            pct = max(0.0, min(1.0, pct))
            thumb_x = sx1 + (sx2 - sx1) * pct
            tag = f"settings_slider_{key}"
            canvas.create_line(sx1, sy, sx2, sy, fill="#6b7075", width=4,
                               capstyle="round", tags=("settings_ui", tag))
            canvas.create_line(sx1, sy, thumb_x, sy, fill="#ffffff", width=4,
                               capstyle="round", tags=("settings_ui", tag))
            canvas.create_oval(thumb_x - 8, sy - 8, thumb_x + 8, sy + 8,
                               outline="#ffffff", fill="#ffffff",
                               tags=("settings_ui", tag))
            canvas.create_text(sx2 + 36, sy, text=fmt(value), fill=text, anchor="w",
                               font=("Segoe UI", 10, "bold"), tags=("settings_ui", tag))

            def set_from_event(event):
                pct_inner = max(0.0, min(1.0, (event.x - sx1) / (sx2 - sx1)))
                var.set(min_value + (max_value - min_value) * pct_inner)
                self._draw_canvas_settings(canvas, sections, True)

            def start_drag(event):
                set_from_event(event)
                canvas.bind("<B1-Motion>", set_from_event)
                canvas.bind("<ButtonRelease-1>", stop_drag)
                return "break"

            def stop_drag(_event=None):
                canvas.unbind("<B1-Motion>")
                canvas.unbind("<ButtonRelease-1>")
                return "break"

            canvas.tag_bind(tag, "<Button-1>", start_drag)
            canvas.tag_bind(tag, "<B1-Motion>", set_from_event)
            canvas.tag_bind(tag, "<Enter>", lambda _e: canvas.configure(cursor="hand2"))
            canvas.tag_bind(tag, "<Leave>", lambda _e: canvas.configure(cursor=""))
            y_positions[0] += 52

        def preview_button():
            tag = "settings_preview"
            bx, by = x + 6, y_positions[0]
            bw, bh = 120, 36
            running = self._preview_thread is not None and self._preview_thread.is_alive()
            label = "Close Preview" if running else "Open Preview"
            fill = self.DARK["danger"] if running else WelcomeUI.BUTTON
            if Image is not None:
                preview_photo = (
                    self._render_nav_pill(bw, bh, fill, 10)
                    if running else self._render_setup_action_button(bw, bh, fill)
                )
                self._settings_header_panels.append(preview_photo)
                canvas.create_image(bx, by, anchor="nw", image=preview_photo,
                                    tags=("settings_ui", tag))
            else:
                outline = "" if running else WelcomeUI.BUTTON_BORDER
                canvas.create_rectangle(bx, by, bx + bw, by + bh, outline=outline, fill=fill,
                                        tags=("settings_ui", tag))
            canvas.create_text(bx + bw / 2, by + bh / 2, text=label, fill="#ffffff",
                               font=("Segoe UI", 10, "bold"), tags=("settings_ui", tag))
            self._preview_status_canvas = canvas.create_text(
                bx + 138, by + bh / 2,
                text=getattr(self, "_preview_status_text", "Closed"),
                fill=muted,
                anchor="w",
                font=("Segoe UI", 9),
                tags=("settings_ui",),
            )
            bind_click(tag, self._toggle_setup_preview)
            y_positions[0] += 66

        def default_button():
            tag = "settings_default"
            bw, bh = 170, 36
            bx = x + width - bw - 6
            by = y_positions[0]
            if Image is not None:
                default_photo = self._render_setup_action_button(bw, bh, WelcomeUI.BUTTON)
                self._settings_header_panels.append(default_photo)
                canvas.create_image(bx, by, anchor="nw", image=default_photo,
                                    tags=("settings_ui", tag))
            else:
                canvas.create_rectangle(bx, by, bx + bw, by + bh, outline=WelcomeUI.BUTTON_BORDER,
                                        fill=WelcomeUI.BUTTON,
                                        tags=("settings_ui", tag))
            canvas.create_text(bx + bw / 2, by + bh / 2, text="Set default", fill="#ffffff",
                               font=("Segoe UI", 10, "bold"), tags=("settings_ui", tag))
            bind_click(tag, self._set_launcher_defaults)
            y_positions[0] += 58

        def section(key, label, draw_fn):
            sections[key] = y_positions[0] + self._settings_scroll_y
            panel_x = x - 16
            panel_y = y_positions[0] - 18
            panel_w = width + 32
            title(label)
            draw_fn()
            panel_h = y_positions[0] - panel_y + 20
            if Image is not None:
                panel_photo = self._render_alpha_panel(panel_w, panel_h, "#000000", 0.3)
                self._settings_header_panels.append(panel_photo)
                panel_id = canvas.create_image(
                    panel_x, panel_y, anchor="nw", image=panel_photo, tags=("settings_ui",)
                )
            else:
                panel_id = canvas.create_rectangle(
                    panel_x, panel_y, panel_x + panel_w, panel_y + panel_h,
                    outline="", fill="#000000", stipple="gray25", tags=("settings_ui",)
                )
            canvas.tag_lower(panel_id, "settings_ui")
            y_positions[0] += 44

        y_positions = [y]
        section("camera", "Camera Window", lambda: (
            checkbox("Show camera preview during tracking", self._camera_window_var, "camera_window"),
            checkbox("Show debug eye landmarks", self._camera_debug_var, "camera_debug"),
            checkbox("Show distance panel", self._distance_var, "distance"),
            preview_button(),
        ))
        section("layout", "Keyboard Layout", lambda: (
            radio("Default", self._ui_layout_var, "ui2", "layout"),
            radio("QWERTY", self._ui_layout_var, "qwerty", "layout"),
        ))
        section("language", "Language", lambda: (
            checkbox("English", self._language_english_var, "english"),
            checkbox("Tagalog", self._language_tagalog_var, "tagalog"),
        ))
        section("dwell", "Dwell Mode", lambda: (
            radio("Synchronous", self._dwell_mode_var, "sync", "dwell"),
            radio("Asynchronous", self._dwell_mode_var, "async", "dwell"),
        ))
        section("calibration", "Calibration", lambda: (
            canvas.create_text(x + 6, y_positions[0], text="Grid Points", fill=text, anchor="w",
                               font=("Segoe UI", 10, "bold"), tags=("settings_ui",)),
            y_positions.__setitem__(0, y_positions[0] + 34),
            radio("5 (Fast)", self._points_var, 5, "points"),
            radio("9 (Default)", self._points_var, 9, "points"),
            radio("16 (Precise)", self._points_var, 16, "points"),
            radio("25 (Deadeye)", self._points_var, 25, "points"),
            slider("Samples/Points", self._samples_var, 20, 120, lambda v: f"{int(round(v))}", "samples"),
        ))
        section("kalman", "Kalman Filter", lambda: (
            slider("Process noise", self._pnoise_var, 0.0001, 0.1, lambda v: f"{v:.4f}", "pnoise"),
            slider("Measurement noise", self._mnoise_var, 1.0, 50.0, lambda v: f"{v:.1f}", "mnoise"),
        ))
        section("smoother", "Smoother", lambda: (
            checkbox("Enable EMA smoother", self._ema_on, "ema_on"),
            slider("EMA alpha", self._ema_var, 0.01, 1.0, lambda v: f"{v:.2f}", "ema"),
        ))

        default_button()

        self._settings_content_height = max(1, y_positions[0] + self._settings_scroll_y)
        actual_max_scroll = max(0, self._settings_content_height - max(1, canvas.winfo_height()))
        if self._settings_scroll_y > actual_max_scroll:
            self._settings_scroll_y = int(round(actual_max_scroll))
            self._draw_canvas_settings(canvas, sections, keep_scroll=True)
            return
        self._draw_settings_scrollbar(canvas)

    def _scroll_canvas_settings(self, canvas, sections, delta):
        if not delta:
            return
        max_scroll = max(0, self._settings_content_height - max(1, canvas.winfo_height()))
        self._settings_scroll_y = int(round(max(0, min(max_scroll, self._settings_scroll_y + delta))))
        self._draw_canvas_settings(canvas, sections, keep_scroll=True)

    def _draw_settings_scrollbar(self, canvas):
        canvas.delete("settings_scrollbar")
        view_h = max(1, canvas.winfo_height())
        content_h = max(1, self._settings_content_height)
        if content_h <= view_h:
            return
        x = canvas.winfo_width() - 16
        top = 18
        bottom = view_h - 18
        track_h = max(1, bottom - top)
        thumb_h = max(42, track_h * (view_h / content_h))
        max_scroll = max(1, content_h - view_h)
        thumb_y = top + (track_h - thumb_h) * (self._settings_scroll_y / max_scroll)
        canvas.create_line(x, top, x, bottom, fill="#070809", width=18,
                           capstyle="round", tags=("settings_scrollbar",))
        canvas.create_line(x, top, x, bottom, fill="#050607", width=4,
                           capstyle="round", tags=("settings_scrollbar",))
        canvas.create_line(x, thumb_y, x, thumb_y + thumb_h, fill="#2c3034", width=4,
                           capstyle="round", tags=("settings_scrollbar",))

    def _layout_canvas_sections(self, canvas, sections, section_windows, section_order):
        canvas.update_idletasks()
        x = 270
        width = max(320, canvas.winfo_width() - x - 70)
        y = 38
        for key in section_order:
            frame = sections[key]
            frame.configure(width=width)
            frame.update_idletasks()
            canvas.itemconfigure(section_windows[key], width=width)
            canvas.coords(section_windows[key], x, y)
            y += frame.winfo_reqheight() + 30
        canvas.configure(scrollregion=(0, 0, max(1, canvas.winfo_width()), y))

    def _render_nav_pill(self, width, height, fill, radius):
        scale = 3
        img = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        fill_rgb = self._hex_to_rgb(fill)
        rect = (0, 0, width * scale - 1, height * scale - 1)
        if radius:
            draw.rounded_rectangle(rect, radius=radius * scale, fill=(*fill_rgb, 255))
        else:
            draw.rectangle(rect, fill=(*fill_rgb, 255))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _render_setup_action_button(self, width, height, fill):
        scale = 4
        img = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, "RGBA")
        border = self._hex_to_rgb(WelcomeUI.BUTTON_BORDER)
        fill_rgb = self._hex_to_rgb(fill)
        rect = (2 * scale, 2 * scale, (width - 2) * scale, (height - 2) * scale)
        draw.rounded_rectangle(
            rect,
            radius=(height // 2 - 1) * scale,
            fill=(*fill_rgb, 255),
            outline=(*border, 255),
            width=scale,
        )
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _render_alpha_panel(self, width, height, fill, opacity):
        width = max(1, int(round(width)))
        height = max(1, int(round(height)))
        alpha = max(0, min(255, round(255 * opacity)))
        fill_rgb = self._hex_to_rgb(fill)
        img = Image.new("RGBA", (width, height), (*fill_rgb, alpha))
        return ImageTk.PhotoImage(img)

    def _build_launcher_controls(self):
        c = self._launcher_canvas
        self.minimize_btn = c.create_text(780, 23, text="-",
                                          fill="#d7d9db", font=("Segoe UI", 15, "bold"))
        self.close_btn = c.create_text(812, 23, text="x",
                                       fill="#d7d9db", font=("Segoe UI", 13, "bold"))
        c.tag_bind(self.minimize_btn, "<Button-1>", self._on_minimize)
        c.tag_bind(self.close_btn, "<Button-1>", self._on_cancel)
        c.tag_bind(self.minimize_btn, "<Enter>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#ffffff"))
        c.tag_bind(self.minimize_btn, "<Leave>", lambda _e: c.itemconfigure(self.minimize_btn, fill="#d7d9db"))
        c.tag_bind(self.close_btn, "<Enter>", lambda _e: c.itemconfigure(self.close_btn, fill="#ff6b6b"))
        c.tag_bind(self.close_btn, "<Leave>", lambda _e: c.itemconfigure(self.close_btn, fill="#d7d9db"))

    def _start_window_drag(self, event):
        self._dragging_window = event.y < 44 and event.x < self.winfo_width() - 92
        if not self._dragging_window:
            return
        self._drag_offset_x = event.x
        self._drag_offset_y = event.y

    def _drag_window(self, _event):
        if not self._dragging_window:
            return
        x = self.winfo_pointerx() - self._drag_offset_x
        y = self.winfo_pointery() - self._drag_offset_y
        self.geometry(f"+{x}+{y}")

    def _stop_window_drag(self, _event=None):
        self._dragging_window = False

    def _on_minimize(self, _event=None):
        self.overrideredirect(False)
        self.iconify()
        self.after(50, lambda: self.overrideredirect(True))

    def _render_launcher_gradient(self, width, height, phase, offset_x=0, offset_y=0,
                                  full_width=None, full_height=None):
        return WelcomeUI._render_angular_gradient(
            self, width, height, phase,
            offset_x=offset_x, offset_y=offset_y,
            full_width=full_width, full_height=full_height,
        )

    def _animate_launcher_gradient(self):
        if self._gradient_image_id is None:
            return
        self._gradient_phase += 1
        width = self.winfo_width() if self.winfo_width() > 1 else 843
        height = self.winfo_height() if self.winfo_height() > 1 else 555
        self._gradient_photo = self._render_launcher_gradient(width, height, self._gradient_phase)
        self._launcher_canvas.itemconfigure(self._gradient_image_id, image=self._gradient_photo)
        if self._content_canvas is not None and self._content_gradient_id is not None:
            content_width = self._content_canvas.winfo_width()
            content_height = self._content_canvas.winfo_height()
            if content_width > 1 and content_height > 1:
                offset_x, offset_y, full_width, full_height = getattr(
                    self, "_content_gradient_view",
                    (0, 0, width, height),
                )
                self._content_gradient_photo = self._render_launcher_gradient(
                    content_width,
                    content_height,
                    self._gradient_phase,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    full_width=full_width,
                    full_height=full_height,
                )
                self._content_canvas.itemconfigure(
                    self._content_gradient_id,
                    image=self._content_gradient_photo,
                )
        self._animate_job = self.after(85, self._animate_launcher_gradient)

    @staticmethod
    def _hex_to_rgb(color):
        return WelcomeUI._hex_to_rgb(color)

    def _toggle_ema(self):
        if not hasattr(self, "_ema_frame"):
            return
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
        elif hasattr(self, "_settings_canvas") and hasattr(self, "_settings_sections"):
            self._draw_canvas_settings(self._settings_canvas, self._settings_sections, keep_scroll=True)

    def _set_preview_status(self, text, color=None):
        self._preview_status_text = text
        if not hasattr(self, "_preview_status"):
            if hasattr(self, "_settings_canvas") and hasattr(self, "_settings_sections"):
                self._draw_canvas_settings(self._settings_canvas, self._settings_sections, keep_scroll=True)
                self._sync_preview_button()
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
            head_ref = None
            head_shift_threshold = 0.2

            while self._preview_stop is not None and not self._preview_stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.after(0, lambda: self._set_preview_status("Frame read failed", self.DARK["danger"]))
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                lms = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None
                current_head_feature = None

                if lms is not None:
                    left_eye = np.array([lms[468].x * w, lms[468].y * h])
                    right_eye = np.array([lms[473].x * w, lms[473].y * h])
                    glabella = np.array([lms[9].x * w, lms[9].y * h])
                    left_anchor = (
                        np.array([lms[33].x * w, lms[33].y * h]) +
                        np.array([lms[133].x * w, lms[133].y * h])
                    ) / 2.0
                    right_anchor = (
                        np.array([lms[362].x * w, lms[362].y * h]) +
                        np.array([lms[263].x * w, lms[263].y * h])
                    ) / 2.0
                    face_scale = float(np.linalg.norm(right_anchor - left_anchor))
                    if face_scale > 1.0:
                        current_head_feature = glabella / face_scale
                    pos_x = glabella[0] - (w / 2.0)
                    pos_y = (h / 2.0) - glabella[1]
                    pos_x_norm = pos_x / (w / 2.0)
                    pos_y_norm = pos_y / (h / 2.0)
                    focal_x_px = w / (2.0 * math.tan(math.radians(camera_hfov_deg) / 2.0))
                    angle_x = math.degrees(math.atan(pos_x / focal_x_px))
                    angle_y = math.degrees(math.atan(pos_y_norm))

                    cx, cy = int(w / 2), int(h / 2)
                    gx, gy = int(glabella[0]), int(glabella[1])
                    cv2.line(frame, (cx - 24, cy), (cx + 24, cy), (80, 80, 80), 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy - 24), (cx, cy + 24), (80, 80, 80), 1, cv2.LINE_AA)
                    cv2.circle(frame, (gx, gy), 11, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (gx - 15, gy), (gx + 15, gy), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.line(frame, (gx, gy - 15), (gx, gy + 15), (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, "GLABELLA", (gx + 16, gy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(frame, (cx, cy), (gx, gy), (0, 160, 220), 1, cv2.LINE_AA)

                    cv2.rectangle(frame, (10, h - 78), (470, h - 14), (18, 18, 18), -1)
                    cv2.putText(frame, "Press L to set/restart head baseline", (22, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                    if head_ref is not None and current_head_feature is not None:
                        delta = current_head_feature - head_ref
                        shift = float(np.linalg.norm(delta))
                        moved = shift >= head_shift_threshold
                        color = (0, 100, 255) if moved else (0, 220, 120)
                        label = "HEAD MOVED" if moved else "Head stable"
                        cv2.putText(frame, f"{label}: {shift:.2f}  dx {delta[0]:+.2f}  dy {delta[1]:+.2f}",
                                    (22, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        if moved:
                            cv2.rectangle(frame, (w // 2 - 300, 26), (w // 2 + 300, 86), (18, 18, 18), -1)
                            cv2.rectangle(frame, (w // 2 - 300, 26), (w // 2 + 300, 86), color, 2)
                            cv2.putText(frame, "Head moved from baseline", (w // 2 - 220, 64),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "No baseline set", (22, h - 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

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
                if key in (ord("l"), ord("L")):
                    if current_head_feature is not None:
                        head_ref = current_head_feature.copy()
                        self.after(0, lambda: self._set_preview_status("Head baseline set", self.DARK["accent"]))
                    else:
                        self.after(0, lambda: self._set_preview_status("No face for baseline", self.DARK["danger"]))
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
        if self._animate_job is not None:
            try:
                self.after_cancel(self._animate_job)
            except Exception:
                pass
            self._animate_job = None
        self.destroy()

    def destroy(self):
        if getattr(self, "_destroying", False):
            return
        self._destroying = True
        if self._animate_job is not None:
            try:
                self.after_cancel(self._animate_job)
            except Exception:
                pass
            self._animate_job = None
        try:
            if self._content_canvas is not None:
                self._content_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            if hasattr(self, "_launcher_canvas") and self._launcher_canvas is not None:
                self._launcher_canvas.delete("all")
        except Exception:
            pass
        self._settings_header_panels = []
        self._gradient_photo = None
        self._content_gradient_photo = None
        self._nav_panel_photo = None
        super().destroy()

    def _on_start(self, _event=None, tutorial=True):
        self._stop_setup_preview()
        if self._animate_job is not None:
            try:
                self.after_cancel(self._animate_job)
            except Exception:
                pass
            self._animate_job = None
        ui_layout = self._ui_layout_var.get()
        if tutorial and ui_layout != "ui2":
            print("[Info] Tutorial uses Default layout; switching from QWERTY for this session.")
            ui_layout = "ui2"
        language_preset = []
        if self._language_english_var.get():
            language_preset.append("english")
        if self._language_tagalog_var.get():
            language_preset.append("tagalog")
        if not language_preset:
            language_preset = ["english", "tagalog"]
        self.result = {
            "camera":  self._camera_var.get(),
            "points":  self._points_var.get(),
            "samples": self._samples_var.get(),
            "ema":     self._ema_var.get() if self._ema_on.get() else 1.0,
            "pnoise":  self._pnoise_var.get(),
            "mnoise":  self._mnoise_var.get(),
            "dwell_mode": self._dwell_mode_var.get(),
            "ui_layout": ui_layout,
            "language_preset": language_preset,
            "camera_window": self._camera_window_var.get(),
            "camera_debug": self._camera_debug_var.get(),
            "distance_panel": self._distance_var.get(),
            "tutorial": tutorial,
        }
        self.quit()


# =============================================================================
#  Helpers
# =============================================================================

# =============================================================================
#  Main
# =============================================================================

def main():
    welcome = WelcomeUI()
    welcome.mainloop()
    welcome_result = welcome.result
    try:
        welcome.destroy()
    except Exception:
        pass

    if welcome_result != "start":
        print("[Info] Welcome cancelled.")
        sys.exit(0)

    # ── Show launcher UI ──────────────────────────────────────────────────────
    launcher = LauncherUI()
    launcher.mainloop()

    if launcher.result is None:
        try:
            launcher.destroy()
        except Exception:
            pass
        print("[Info] Launcher cancelled.")
        sys.exit(0)

    cfg = launcher.result
    try:
        launcher.destroy()
    except Exception:
        pass

    startup_loading = None
    startup_loading_label = None
    startup_loading_detail = None

    def show_startup_loading(message="Preparing calibration...", detail="Loading system components."):
        nonlocal startup_loading, startup_loading_label, startup_loading_detail
        hide_startup_loading()
        loading = tk.Tk()
        loading.withdraw()
        loading.overrideredirect(True)
        loading.attributes("-topmost", True)
        loading.configure(bg="#111214", cursor="watch")
        loading.title("Starting")

        width, height = 500, 180
        x = max(0, (loading.winfo_screenwidth() - width) // 2)
        y = max(0, (loading.winfo_screenheight() - height) // 2)
        loading.geometry(f"{width}x{height}+{x}+{y}")

        frame = tk.Frame(loading, bg="#111214", highlightthickness=1, highlightbackground="#44484f")
        frame.pack(fill="both", expand=True)
        startup_loading_label = tk.Label(
            frame,
            text=message,
            bg="#111214",
            fg="#ffffff",
            font=("Segoe UI", 18, "bold"),
        )
        startup_loading_label.pack(pady=(42, 8))
        startup_loading_detail = tk.Label(
            frame,
            text=detail,
            bg="#111214",
            fg="#b9bec7",
            font=("Segoe UI", 10),
        )
        startup_loading_detail.pack()

        startup_loading = loading
        loading.deiconify()
        loading.lift()
        loading.update()

    def update_startup_loading(message=None, detail=None):
        if startup_loading is None:
            return
        if message is not None and startup_loading_label is not None:
            startup_loading_label.config(text=message)
        if detail is not None and startup_loading_detail is not None:
            startup_loading_detail.config(text=detail)
        try:
            startup_loading.update()
        except Exception:
            pass

    def hide_startup_loading():
        nonlocal startup_loading, startup_loading_label, startup_loading_detail
        if startup_loading is None:
            return
        try:
            startup_loading.destroy()
        except Exception:
            pass
        startup_loading = None
        startup_loading_label = None
        startup_loading_detail = None

    show_startup_loading()

    print("=" * 60)
    print("  GAZE-BASED DIGITAL KEYBOARD")
    print(f"  Points: {cfg['points']}  |  Samples: {cfg['samples']}  |  "
          f"EMA: {cfg['ema']:.2f}  |  Camera: {cfg['camera']}  |  "
          f"Dwell: {cfg['dwell_mode']}  |  UI: {cfg['ui_layout']}  |  "
          f"Language: {', '.join(cfg['language_preset'])}")
    print("=" * 60)

    # ── Deferred imports (avoid slowing down launcher) ────────────────────────
    update_startup_loading("Preparing calibration...", "Loading gaze tracker and keyboard modules.")
    from gaze_tracker2 import GazeTrackerApp
    import config
    from model import ngram_model
    from generate_flores_rules import generate_if_missing as _ensure_flores
    from config import FILIPINO_DATASET_FILE, ENGLISH_DATASET_FILE, NGRAM_CACHE_FILE

    config.DWELL_MODE = cfg["dwell_mode"]
    if cfg["language_preset"] == ["english"]:
        config.PREDICTION_LANGUAGE = "english"
    elif cfg["language_preset"] == ["tagalog"]:
        config.PREDICTION_LANGUAGE = "filipino"
    else:
        config.PREDICTION_LANGUAGE = "both"

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
    update_startup_loading("Preparing calibration...", "Checking datasets.")
    _ensure_datasets()
    update_startup_loading("Preparing calibration...", "Checking language rules.")
    _ensure_flores()

    if not os.path.exists(NGRAM_CACHE_FILE):
        update_startup_loading("Preparing calibration...", "Building missing model cache.")
        print("Model cache missing — forcing dataset regeneration first.")
        _rebuild_datasets()

    update_startup_loading("Preparing calibration...", "Loading prediction model.")
    if not ngram_model.load_cache():
        update_startup_loading("Preparing calibration...", "Training prediction model.")
        print("Building n-gram model from datasets...")
        if not ngram_model.train_from_builtin():
            hide_startup_loading()
            print("Failed to build n-gram model: datasets were not available.")
            sys.exit(1)
        ngram_model.save_cache()
    ngram_model.load_user_learning()

    stop_gaze = threading.Event()
    gaze_thread = None
    recalibrating = False

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
    update_startup_loading("Opening calibration...", "Please wait while the camera starts.")
    ok = tracker.calibrate(window_ready_callback=hide_startup_loading)
    if not ok:
        hide_startup_loading()
        print("[Info] Calibration cancelled.")
        sys.exit(0)
    hide_startup_loading()
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

    def stop_tracking(timeout=6.0, pump_ui=None):
        stop_gaze.set()
        if not gaze_thread or not gaze_thread.is_alive():
            return True
        deadline = time.monotonic() + timeout
        while gaze_thread.is_alive() and time.monotonic() < deadline:
            gaze_thread.join(timeout=0.05)
            if pump_ui is not None:
                try:
                    pump_ui()
                except Exception:
                    pass
        return not gaze_thread.is_alive()

    start_tracking()

    # ── Launch Tkinter keyboard on main thread ────────────────────────────────
    from ui import FilipinoKeyboard

    app = FilipinoKeyboard(
        ui_layout=cfg["ui_layout"],
        gaze_tracking_active=True,
        ui_tutorial=cfg["tutorial"],
    )
    recalibration_overlay = None

    def show_recalibration_overlay(message="Preparing recalibration..."):
        nonlocal recalibration_overlay
        hide_recalibration_overlay()
        overlay = tk.Toplevel()
        overlay.withdraw()
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#111214", cursor="watch")
        overlay.title("Recalibrating")
        try:
            overlay.grab_set()
        except Exception:
            pass

        width, height = 460, 170
        x = max(0, (overlay.winfo_screenwidth() - width) // 2)
        y = max(0, (overlay.winfo_screenheight() - height) // 2)
        overlay.geometry(f"{width}x{height}+{x}+{y}")

        frame = tk.Frame(overlay, bg="#111214", highlightthickness=1, highlightbackground="#44484f")
        frame.pack(fill="both", expand=True)
        tk.Label(
            frame,
            text=message,
            bg="#111214",
            fg="#ffffff",
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=(40, 8))
        tk.Label(
            frame,
            text="Please wait while gaze tracking pauses.",
            bg="#111214",
            fg="#b9bec7",
            font=("Segoe UI", 10),
        ).pack()

        recalibration_overlay = overlay
        overlay.deiconify()
        overlay.lift()
        app.update()

    def hide_recalibration_overlay():
        nonlocal recalibration_overlay
        if recalibration_overlay is None:
            return
        try:
            recalibration_overlay.grab_release()
        except Exception:
            pass
        try:
            recalibration_overlay.destroy()
        except Exception:
            pass
        recalibration_overlay = None

    def on_close():
        stop_tracking()
        hide_recalibration_overlay()
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
        nonlocal stop_gaze, recalibrating
        if recalibrating:
            return "break"
        recalibrating = True

        tutorial_step_before_recalibration = getattr(app, "_tutorial_step", None)
        tutorial_was_active = bool(
            getattr(app, "_ui_tutorial_enabled", False)
            and tutorial_step_before_recalibration not in (None, "done")
        )
        app.prepare_for_recalibration(preserve_tutorial=True)
        previous_dwell = True if tutorial_was_active else getattr(app, "dwell_enabled", True)
        tracker._head_shift = None
        app.status_bar.config(text="Recalibrating gaze...")
        app.withdraw()
        app.update()
        show_recalibration_overlay()
        app.update()

        original_calib_tutorial = tracker._tutorial_enabled
        try:
            if not stop_tracking(pump_ui=app.update):
                hide_recalibration_overlay()
                app.deiconify()
                app.dwell_enabled = previous_dwell
                if getattr(app, "_use_pointer_overlay", False) and getattr(app, "_pointer_overlay", None) is None:
                    app._init_pointer_overlay()
                app._show_main_pointer()
                if tutorial_was_active:
                    app.restart_ui2_tutorial_after_recalibration()
                else:
                    app.update_display()
                app.status_bar.config(text="Recalibration blocked | camera is still stopping")
                return "break"

            stop_gaze = threading.Event()
            tracker._tutorial_enabled = False
            tracker._pip = False
            app.update()
            ok = tracker.calibrate(window_ready_callback=hide_recalibration_overlay)
            if ok:
                tracker._head_shift = None
                tracker._mouse_ctrl = True
                tracker._pip = False
                app.deiconify()
                app.lift()
                app.dwell_enabled = previous_dwell
                if getattr(app, "_use_pointer_overlay", False) and getattr(app, "_pointer_overlay", None) is None:
                    app._init_pointer_overlay()
                app._show_main_pointer()
                if tutorial_was_active:
                    app.restart_ui2_tutorial_after_recalibration()
                else:
                    app.update_display()
                start_tracking()
                app.status_bar.config(text="Recalibration complete | gaze tracking active")
            else:
                hide_recalibration_overlay()
                app.status_bar.config(text="Recalibration cancelled | closing session")
                on_close()
        finally:
            hide_recalibration_overlay()
            tracker._tutorial_enabled = original_calib_tutorial
            recalibrating = False
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

    def monitor_head_position():
        if recalibrating:
            app.after(250, monitor_head_position)
            return
        shift = getattr(tracker, "_head_shift", None)
        if shift and shift.get("moved"):
            app.show_head_position_warning(shift)
        else:
            app.hide_head_position_warning()
        app.after(250, monitor_head_position)

    app.protocol("WM_DELETE_WINDOW", on_close)
    app.bind_all("<KeyPress-q>", quit_session)
    app.bind_all("<KeyPress-Q>", quit_session)
    app.bind_all("<KeyPress-x>", toggle_mouse_control)
    app.bind_all("<KeyPress-X>", toggle_mouse_control)
    app.bind_all("<KeyPress-r>", recalibrate)
    app.bind_all("<KeyPress-R>", recalibrate)
    app.after(1000, monitor_tracking)
    app.after(250, monitor_head_position)
    app.mainloop()

    # Cleanup
    stop_tracking()
    print("[Info] Application closed.")


if __name__ == "__main__":
    main()
