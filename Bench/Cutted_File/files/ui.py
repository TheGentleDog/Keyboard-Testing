# =============================================================================
# ui.py — FilipinoKeyboard UI (display, keyboard layout, predictions, settings)
# =============================================================================

import json
import os
import tkinter as tk
from tkinter import ttk

import config
from dwell import DwellMixin
from model import ngram_model, get_context_words
from tts import speak
import panic_sound

try:
    from cnn_phrase_model import cnn_phrase_model
except Exception:
    cnn_phrase_model = None

PREDEFINED_FILE      = "predefined_sentences.json"
PREDEFINED_THRESHOLD = 3   # times spoken before auto-saving
MAX_PREDEFINED_SENTENCES = 9


class FilipinoKeyboard(tk.Tk, DwellMixin):
    GAZE_FRAME_GAP_PX = 60

    THEMES = {
        "light": {
            "bg":               "#f0f0f0",
            "output_bg":        "#ffffff",
            "input_bg":         "#f9f9f9",
            "text_fg":          "black",
            "suggestion_fg":    "gray",
            "button_bg":        "#e3e3e3",
            "button_fg":        "black",
            "button_active_bg": "#d4d4d4",
            "funckey_bg":       "#dedede",
            "funckey_fg":       "black",
            "funckey_active_bg":"#d2d2d2",
            "button_border":    "#a8a8a8",
            "dwell_bar":        "#00cc44",
            "dwell_bg":         "#c8f0d8",
        },
        "dark": {
            "bg":               "#1e1f22",
            "output_bg":        "#535353",
            "input_bg":         "#535353",
            "text_fg":          "#dcddde",
            "suggestion_fg":    "#8e9297",
            # letter keys + prediction bar
            "button_bg":        "#282828",
            "button_fg":        "#ffffff",
            "button_active_bg": "#5865f2",
            # function row (arrows, space, predefined, tts)
            "funckey_bg":       "#171719",
            "funckey_fg":       "#ffffff",
            "funckey_active_bg":"#5865f2",
            "tts_bg":           "#071f4a",
            "tts_active_bg":    "#0b2d68",
            # panic button
            "panic_bg":         "#660002",
            "dwell_bar":        "#55ff88",
            "dwell_bg":         "#1a3a28",
        },
    }

    POINTER_SIZE = 72
    POINTER_OUTLINE = "#8b0018"
    POINTER_FILL = "#8f8f8f"

    # ── Override dwell flash to restore correct per-button colour ─────────────
    def _dwell_flash(self, btn):
        theme = self.themes[self.current_theme]
        kb = self.keyboard_buttons if hasattr(self, 'keyboard_buttons') else []
        special = []
        if hasattr(self, '_backspace_btn'): special.append(self._backspace_btn)
        if hasattr(self, '_clearall_btn'):  special.append(self._clearall_btn)
        func_keys = kb[:5] + special + \
                    (self.predefined_func_buttons if hasattr(self, 'predefined_func_buttons') else [])
        if hasattr(self, 'panic_btn') and btn is self.panic_btn:
            restore_bg = theme.get("panic_bg", "#8b0000")
            restore_fg = "white"
        elif btn in func_keys:
            restore_bg = theme.get("funckey_bg", theme["button_bg"])
            restore_fg = theme.get("funckey_fg", theme["button_fg"])
        else:
            restore_bg = theme["button_bg"]
            restore_fg = theme["button_fg"]
        try:
            btn.config(bg="#00cc44", fg="#ffffff")
            def restore():
                try:
                    btn.config(bg=restore_bg, fg=restore_fg)
                except Exception:
                    pass
            btn.after(200, restore)
        except Exception:
            pass

    # ── macOS-compatible button override ──────────────────────────────────────
    def _theme_button_chrome(self):
        theme = self.themes[self.current_theme]
        border = theme.get("button_border")
        if border:
            return {
                "relief": "flat",
                "bd": 0,
                "highlightthickness": 1,
                "highlightbackground": border,
                "highlightcolor": border,
            }
        return {
            "relief": "raised",
            "bd": 1,
            "highlightthickness": 0,
        }

    def _apply_button_chrome(self, widget):
        try:
            widget.config(**self._theme_button_chrome())
        except Exception:
            pass

    def _make_dwell_btn(self, parent, command, **kwargs):
        """
        Override DwellMixin._make_dwell_btn to use tk.Label instead of
        tk.Button so that bg colours render correctly on macOS (Tkinter
        buttons ignore bg on macOS due to native rendering).
        """
        # Strip args that are Button-only
        kwargs.pop('command', None)
        relief = kwargs.pop('relief', 'flat')
        bd     = kwargs.pop('bd', 1)
        kwargs['cursor'] = getattr(self, "pointer_cursor", "arrow")
        chrome = self._theme_button_chrome().copy()
        if self.themes[self.current_theme].get("button_border"):
            relief = chrome.pop("relief")
            bd = chrome.pop("bd")
            kwargs.update(chrome)

        lbl = tk.Label(parent, relief=relief, bd=bd, **kwargs)

        def guarded_command():
            if not self._tutorial_allows_widget(lbl):
                self._show_tutorial_locked_status()
                return
            command()

        lbl.bind('<Button-1>', lambda _e, c=guarded_command: c())
        self._dwell_register(lbl, guarded_command)
        return lbl

    def _show_tutorial_locked_status(self):
        if not self._ui_tutorial_enabled or not hasattr(self, "status_bar"):
            return
        self.status_bar.config(text="Tutorial: follow the highlighted target")

    def _tutorial_allows_widget(self, widget):
        if getattr(self, "_panic_active", False):
            return False
        if not getattr(self, "_ui_tutorial_enabled", False):
            return True
        step = getattr(self, "_tutorial_step", None)
        if step in (None, "done"):
            return True
        if getattr(self, "_tutorial_input_paused", False):
            return widget is getattr(self, "_finish_tutorial_btn", None)
        allowed = self._tutorial_allowed_widgets()
        if allowed is None:
            return True
        return any(widget is item for item in allowed if item is not None)

    def _tutorial_allowed_widgets(self):
        step = getattr(self, "_tutorial_step", None)
        if step == "welcome":
            return []
        if step == "type_hello":
            return [self._expected_hello_target()[1]]
        if step == "select_there":
            return [self._tutorial_select_there_target()]
        if step in ("tts_hello", "tts_hi"):
            return [self.keyboard_buttons[4] if len(self.keyboard_buttons) >= 5 else None]
        if step == "edit_hello":
            return [self._tutorial_edit_hello_target()]
        if step == "save_sentence":
            return [self.keyboard_buttons[4] if len(self.keyboard_buttons) >= 5 else None]
        if step == "clear_before_predefined":
            if self._in_predefined_mode:
                return [self.keyboard_buttons[3] if len(self.keyboard_buttons) >= 4 else None]
            if self.output_words or self.current_input:
                return [getattr(self, "_clearall_btn", None)]
            return []
        if step == "predefined":
            if self._tutorial_predefined_selected:
                return []
            if self._in_predefined_mode:
                return [self.predefined_func_buttons[0] if self.predefined_func_buttons else None]
            return [self.keyboard_buttons[3] if len(self.keyboard_buttons) >= 4 else None]
        if step == "finish":
            return [getattr(self, "_finish_tutorial_btn", None)]
        if step == "familiarize":
            return []
        return None

    def _tutorial_select_there_target(self):
        words = [w.lower() for w in self.output_words]
        if words == ["hello"]:
            if self.current_input:
                return getattr(self, "_backspace_btn", None)
            return self._prediction_button("there")
        if words and words[0] != "hello":
            return getattr(self, "_clearall_btn", None)
        if len(words) >= 2:
            bad_index = len(words) - 1
            if self.output_cursor == -1:
                return self.keyboard_buttons[0] if len(self.keyboard_buttons) >= 1 else None
            if self.output_cursor < bad_index:
                return self.keyboard_buttons[1] if len(self.keyboard_buttons) >= 2 else None
            if self.output_cursor > bad_index:
                return self.keyboard_buttons[0] if len(self.keyboard_buttons) >= 1 else None
            return getattr(self, "_backspace_btn", None)
        return self._prediction_button("there")

    def _tutorial_edit_hello_target(self):
        words = [w.lower() for w in self.output_words]
        if len(words) < 2 or words[1] != "there" or words[0] not in ("hello", "hi"):
            return getattr(self, "_clearall_btn", None)
        if words[:2] == ["hi", "there"] and not self.current_input:
            return None
        if self.current_input and self.output_cursor != 0:
            return getattr(self, "_backspace_btn", None)
        if self.current_input:
            return self._expected_hi_target()[1]
        if self.output_cursor == -1:
            return self.keyboard_buttons[0] if len(self.keyboard_buttons) >= 1 else None
        if self.output_cursor == 1:
            return self.keyboard_buttons[0] if len(self.keyboard_buttons) >= 1 else None
        if self.output_cursor == 0:
            return self._expected_hi_target()[1]
        return self.keyboard_buttons[1] if len(self.keyboard_buttons) >= 2 else None

    def __init__(self, ui_layout="qwerty", gaze_tracking_active=False, ui_tutorial=False):
        super().__init__()
        self.title("Filipino Keyboard - Gaze-Based")
        self.attributes('-fullscreen', True)
        self._gaze_tracking_active     = gaze_tracking_active
        self._use_pointer_overlay      = gaze_tracking_active
        self._system_cursor_hidden     = False
        self.pointer_cursor            = "none" if self._use_pointer_overlay else "arrow"
        self.configure(cursor=self.pointer_cursor)
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))
        self.bind_all('<KeyPress-s>', self._open_settings_shortcut)
        self.bind_all('<KeyPress-S>', self._open_settings_shortcut)
        self.bind_all('<KeyPress-q>', self._quit_keyboard)
        self.bind_all('<KeyPress-Q>', self._quit_keyboard)
        self.bind_all('<KeyPress-x>', self._keyboard_only_gaze_shortcut)
        self.bind_all('<KeyPress-X>', self._keyboard_only_gaze_shortcut)
        self.bind_all('<KeyPress-r>', self._keyboard_only_gaze_shortcut)
        self.bind_all('<KeyPress-R>', self._keyboard_only_gaze_shortcut)
        self.bind_all('<KeyPress-1>', self._stop_panic_shortcut)

        self.ui_layout               = ui_layout if ui_layout in ("qwerty", "ui2") else "qwerty"
        self._ui2_groups             = ("abcd", "efgh", "ijkl", "mnop", "qrstu", "vwxyz")
        self.current_theme           = "dark"
        self.themes                  = self.THEMES
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.current_input           = ""
        self.output_words            = []
        self.output_cursor           = -1
        self._in_predefined_mode     = False
        self.predefined_func_buttons = []
        self._panic_active           = False
        self._settings_open          = False
        self._settings_window        = None
        self._panic_overlay          = None
        self._panic_overlay_label    = None
        self._panic_blink_job        = None
        self._panic_blink_visible    = True
        self._pointer_overlay        = None
        self._pointer_canvas         = None
        self._pointer_job            = None
        self._head_warning_overlay   = None
        self._head_warning_title     = None
        self._head_warning_detail    = None
        self._current_word_overlay   = None
        self._current_word_label     = None
        self._ui_tutorial_enabled    = ui_tutorial and self.ui_layout == "ui2"
        self._tutorial_overlay       = None
        self._tutorial_prev_dwell    = None
        self._tutorial_step          = None
        self._tutorial_job           = None
        self._tutorial_overlay_canvas = None
        self._guide_overlay          = None
        self._guide_canvas           = None
        self._guide_job              = None
        self._finish_tutorial_btn    = None
        self._tutorial_predefined_selected = False
        self._tutorial_input_paused  = False
        self._ui2_group_buttons      = {}
        self._ui2_letter_buttons     = {}
        self._ui2_letters_open       = False

        self._dwell_init()
        self._load_sentence_counts()
        self._create_widgets()
        if self._use_pointer_overlay:
            self._init_pointer_overlay()
        self._show_main_pointer()
        self.after(50, self._take_focus)
        if self._ui_tutorial_enabled:
            self.after(700, self._start_ui2_tutorial)

    def _quit_keyboard(self, _event=None):
        self.destroy()
        return "break"

    def destroy(self):
        self._hide_panic_stop_overlay()
        self.prepare_for_recalibration(preserve_tutorial=False)
        self._show_system_cursor()
        self._destroy_pointer_overlay()
        self._hide_current_word_overlay()
        super().destroy()

    def prepare_for_recalibration(self, preserve_tutorial=True):
        """Remove transient topmost UI so calibration can own the screen."""
        if not preserve_tutorial:
            self._ui_tutorial_enabled = False
            self._tutorial_input_paused = False
            self._tutorial_step = "done"
        if self._tutorial_job is not None:
            try:
                self.after_cancel(self._tutorial_job)
            except Exception:
                pass
            self._tutorial_job = None
        self._destroy_guide_overlay()
        self._hide_ui_tutorial_text()
        self._hide_finish_tutorial_button()
        self._hide_panic_stop_overlay()
        self.hide_head_position_warning()
        self._hide_current_word_overlay()
        self._show_system_cursor()
        self._destroy_pointer_overlay()
        self._dwell_reset_all()

    def restart_ui2_tutorial_after_recalibration(self):
        if self.ui_layout != "ui2":
            return
        self._ui_tutorial_enabled = True
        self._tutorial_input_paused = False
        self._tutorial_step = None
        self._tutorial_predefined_selected = False
        self._tutorial_prev_dwell = None
        self.dwell_enabled = True
        self.current_input = ""
        self.output_words = []
        self.output_cursor = -1
        self.current_completion = ""
        self.alternative_suggestions = []
        self.update_display()
        self.after(500, self._start_ui2_tutorial)

    def _keyboard_only_gaze_shortcut(self, event=None):
        if hasattr(self, "status_bar"):
            key = event.keysym.upper() if event else ""
            self.status_bar.config(text=f"{key} is available in gaze mode only")
        return "break"

    def _open_settings_shortcut(self, _event=None):
        if self._settings_open:
            self._close_settings_window()
        else:
            self.show_settings()
        return "break"

    def _apply_none_cursor(self, widget):
        try:
            widget.configure(cursor="none")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._apply_none_cursor(child)

    def _apply_arrow_cursor(self, widget):
        try:
            widget.configure(cursor="arrow")
        except Exception:
            pass
        for child in widget.winfo_children():
            self._apply_arrow_cursor(child)

    def _hide_system_cursor(self):
        if os.name != "nt" or not self._gaze_tracking_active or self._system_cursor_hidden:
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            for _ in range(20):
                if user32.ShowCursor(False) < 0:
                    break
            self._system_cursor_hidden = True
        except Exception:
            pass

    def _show_system_cursor(self):
        if os.name != "nt" or not self._system_cursor_hidden:
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            for _ in range(20):
                if user32.ShowCursor(True) >= 0:
                    break
        except Exception:
            pass
        self._system_cursor_hidden = False

    def _init_pointer_overlay(self):
        try:
            overlay = tk.Toplevel(self)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.attributes("-alpha", 0.3)
            overlay.configure(bg="#010203", cursor="none")
            try:
                overlay.wm_attributes("-transparentcolor", "#010203")
            except Exception:
                pass
            try:
                overlay.wm_attributes("-disabled", True)
            except Exception:
                pass

            size = self.POINTER_SIZE
            canvas = tk.Canvas(
                overlay,
                width=size,
                height=size,
                bg="#010203",
                highlightthickness=0,
                bd=0,
                cursor="none",
            )
            canvas.pack()
            border = 5
            canvas.create_oval(
                border,
                border,
                size - border,
                size - border,
                outline=self.POINTER_OUTLINE,
                width=4,
                fill=self.POINTER_FILL,
            )
            self._pointer_overlay = overlay
            self._pointer_canvas = canvas
            self._track_pointer_overlay()
        except Exception:
            self._pointer_overlay = None
            self._pointer_canvas = None

    def _track_pointer_overlay(self):
        if not self._pointer_overlay or self._settings_open:
            return
        try:
            half = self.POINTER_SIZE // 2
            x = self.winfo_pointerx() - half
            y = self.winfo_pointery() - half
            self._pointer_overlay.geometry(f"{self.POINTER_SIZE}x{self.POINTER_SIZE}+{x}+{y}")
            self._pointer_overlay.deiconify()
        except Exception:
            pass
        self._pointer_job = self.after(16, self._track_pointer_overlay)

    def _destroy_pointer_overlay(self):
        if self._pointer_job is not None:
            try:
                self.after_cancel(self._pointer_job)
            except Exception:
                pass
            self._pointer_job = None
        if self._pointer_overlay:
            try:
                self._pointer_overlay.destroy()
            except Exception:
                pass
        self._pointer_overlay = None
        self._pointer_canvas = None

    def show_head_position_warning(self, shift):
        try:
            mag = float(shift.get("mag", 0.0))
            dx = float(shift.get("dx", 0.0))
            dy = float(shift.get("dy", 0.0))
            distance = float(shift.get("distance", 0.0))
        except Exception:
            mag = dx = dy = 0.0
            distance = 0.0
        guide = shift.get("guide") if isinstance(shift, dict) else None
        if not guide:
            guide_parts = []
            if dx > 0.03:
                guide_parts.append("left")
            elif dx < -0.03:
                guide_parts.append("right")
            if dy > 0.03:
                guide_parts.append("up")
            elif dy < -0.03:
                guide_parts.append("down")
            if distance > 0.08:
                guide_parts.append("closer")
            elif distance < -0.08:
                guide_parts.append("further")
            if len(guide_parts) > 2:
                guide_text = ", ".join(guide_parts[:-1]) + ", and " + guide_parts[-1]
            else:
                guide_text = " and ".join(guide_parts)
            guide = "Shift " + guide_text if guide_text else "Hold steady"

        if self._head_warning_overlay is None:
            overlay = tk.Toplevel(self)
            overlay.withdraw()
            overlay.overrideredirect(True)
            overlay.attributes("-topmost", True)
            overlay.configure(bg="#1b1b1b", cursor=self.pointer_cursor)
            try:
                overlay.wm_attributes("-disabled", True)
            except Exception:
                pass

            frame = tk.Frame(overlay, bg="#1b1b1b", highlightthickness=2, highlightbackground="#ff6b35")
            frame.pack(fill="both", expand=True)
            title = tk.Label(
                frame,
                text=guide,
                bg="#1b1b1b",
                fg="#ffb199",
                font=("Segoe UI", 18, "bold"),
            )
            title.pack(fill="x", padx=24, pady=(14, 2))
            detail = tk.Label(
                frame,
                bg="#1b1b1b",
                fg="#f1f1f1",
                font=("Segoe UI", 11),
            )
            detail.pack(fill="x", padx=24, pady=(0, 14))

            self._head_warning_overlay = overlay
            self._head_warning_title = title
            self._head_warning_detail = detail

        self._head_warning_title.config(text=guide)
        self._head_warning_detail.config(
            text=(
                f"Position x {dx:+.2f}, y {dy:+.2f} -> target x 0.00, y 0.00\n"
                f"Distance {distance:+.0%} -> target 0%   Shift {mag:.2f}"
            )
        )
        try:
            width = 680
            height = 116
            x = max(0, (self.winfo_screenwidth() - width) // 2)
            y = 76
            self._head_warning_overlay.geometry(f"{width}x{height}+{x}+{y}")
            self._head_warning_overlay.deiconify()
            self._head_warning_overlay.lift()
        except Exception:
            pass

    def hide_head_position_warning(self):
        if self._head_warning_overlay is None:
            return
        try:
            self._head_warning_overlay.destroy()
        except Exception:
            pass
        self._head_warning_overlay = None
        self._head_warning_title = None
        self._head_warning_detail = None

    def _hide_current_word_overlay(self):
        if self._current_word_overlay is None:
            return
        try:
            self._current_word_overlay.destroy()
        except Exception:
            pass
        self._current_word_overlay = None
        self._current_word_label = None

    def _update_current_word_overlay(self):
        if not hasattr(self, "letters_frame") or self._in_predefined_mode:
            self._hide_current_word_overlay()
            return
        word = self.current_input.strip()
        if not word:
            self._hide_current_word_overlay()
            return

        try:
            if self._current_word_overlay is None:
                overlay = tk.Toplevel(self)
                overlay.withdraw()
                overlay.overrideredirect(True)
                overlay.attributes("-topmost", True)
                overlay.attributes("-alpha", 0.42)
                overlay.configure(bg="#010203", cursor=self.pointer_cursor)
                try:
                    overlay.wm_attributes("-transparentcolor", "#010203")
                except Exception:
                    pass
                try:
                    overlay.wm_attributes("-disabled", True)
                except Exception:
                    pass

                label = tk.Label(
                    overlay,
                    text=word,
                    bg="#010203",
                    fg="#ffffff",
                    font=("Segoe UI", 72, "bold"),
                    anchor="center",
                )
                label.pack(fill="both", expand=True)
                self._current_word_overlay = overlay
                self._current_word_label = label

            self._current_word_label.config(text=word)
            self.letters_frame.update_idletasks()
            x = self.letters_frame.winfo_rootx()
            y = self.letters_frame.winfo_rooty()
            width = self.letters_frame.winfo_width()
            height = self.letters_frame.winfo_height()
            self._current_word_overlay.geometry(f"{width}x{height}+{x}+{y}")
            self._current_word_overlay.deiconify()
            self._current_word_overlay.lift()
        except Exception:
            self._hide_current_word_overlay()

    def _show_main_pointer(self):
        if self._use_pointer_overlay:
            self.configure(cursor="none")
            self._apply_none_cursor(self)
            self._hide_system_cursor()
        else:
            self.configure(cursor="arrow")
            self._apply_arrow_cursor(self)
            self._show_system_cursor()
        if self._pointer_overlay and self._pointer_job is None:
            self._track_pointer_overlay()

    def _show_system_pointer(self):
        self._settings_open = True
        self._show_system_cursor()
        if self._pointer_job is not None:
            try:
                self.after_cancel(self._pointer_job)
            except Exception:
                pass
            self._pointer_job = None
        if self._pointer_overlay:
            try:
                self._pointer_overlay.withdraw()
            except Exception:
                pass
        self.configure(cursor="arrow")
        self._apply_arrow_cursor(self)

    def _restore_main_cursor(self):
        self._settings_open = False
        self._show_main_pointer()

    def _pause_gaze_input_for_settings(self):
        self.dwell_enabled = False
        self.dwell_hovered = None
        self._dwell_reset_all()
        self._zoom_hide()

    def _resume_gaze_input_after_settings(self):
        self.dwell_enabled = config.DWELL_ENABLED
        self.dwell_hovered = None
        self._dwell_reset_all()

    def _close_settings_window(self, _event=None):
        settings_win = self._settings_window
        self._settings_window = None
        if settings_win is not None:
            try:
                settings_win.grab_release()
            except Exception:
                pass
            try:
                settings_win.destroy()
            except Exception:
                pass
        self._resume_gaze_input_after_settings()
        self._restore_main_cursor()
        self._take_focus()
        return "break"

    def _take_focus(self):
        try:
            self.lift()
            self.focus_force()
            self.attributes("-topmost", True)
            self.after(150, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _start_ui2_tutorial(self):
        self._tutorial_messages = [
            ("welcome", "Welcome to the keyboard", None),
            ("type_hello", "Let's type 'hello there'", "keyboard"),
        ]
        self._run_ui2_tutorial_message(0)

    def _run_ui2_tutorial_message(self, index):
        if index >= len(self._tutorial_messages):
            self._start_interactive_hello_guide()
            return
        step, message, target = self._tutorial_messages[index]
        self._tutorial_step = step
        self._show_ui_tutorial_text(
            message,
            5000,
            lambda: self._run_ui2_tutorial_message(index + 1),
            target=target,
        )

    def _start_interactive_hello_guide(self):
        self._tutorial_step = "type_hello"
        self._ensure_guide_overlay()
        self._update_hello_guide()

    def _cancel_guide_job(self):
        if self._guide_job is not None:
            try:
                self.after_cancel(self._guide_job)
            except Exception:
                pass
            self._guide_job = None

    def _position_overlay(self, overlay):
        if overlay is None:
            return
        try:
            self.update_idletasks()
            overlay.geometry(
                f"{self.winfo_width()}x{self.winfo_height()}+"
                f"{self.winfo_rootx()}+{self.winfo_rooty()}"
            )
            overlay.lift()
        except Exception:
            pass

    def _expected_hello_target(self):
        expected = "hello"
        words = [w.lower() for w in self.output_words]
        if words and words != ["hello"]:
            return "Gaze at Clear all, then type Hello again", getattr(self, "_clearall_btn", None)
        typed = self.current_input.lower()
        if not expected.startswith(typed):
            return "Gaze at backspace to fix the word", getattr(self, "_backspace_btn", None)
        if typed == expected:
            return "Gaze at space to enter 'Hello'", self.keyboard_buttons[2] if len(self.keyboard_buttons) >= 3 else None

        next_ch = expected[len(typed)]
        for group in self._ui2_groups:
            if next_ch in group:
                if next_ch in self._ui2_letter_buttons:
                    return f"Choose {next_ch.upper()}", self._ui2_letter_buttons[next_ch]
                return f"Gaze at {group.upper()}", self._ui2_group_buttons.get(group)
        return "", None

    def _expected_hi_target(self):
        expected = "hi"
        typed = self.current_input.lower()
        if not expected.startswith(typed):
            return "Gaze at backspace to fix the word", getattr(self, "_backspace_btn", None)
        if typed == expected:
            return "Gaze at space to replace 'Hello'", self.keyboard_buttons[2] if len(self.keyboard_buttons) >= 3 else None

        next_ch = expected[len(typed)]
        for group in self._ui2_groups:
            if next_ch in group:
                if next_ch in self._ui2_letter_buttons:
                    return f"Choose {next_ch.upper()}", self._ui2_letter_buttons[next_ch]
                return f"Gaze at {group.upper()}", self._ui2_group_buttons.get(group)
        return "", None

    def _update_hello_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "type_hello":
            return
        if self._guide_back_to_keyboard_if_needed(self._update_hello_guide):
            return
        if self.output_words and self.output_words[0].lower() == "hello":
            self._tutorial_step = "select_there"
            self._show_ui_tutorial_text(
                "Gaze at 'there' if it appears in the predictions",
                5000,
                self._update_prediction_guide,
                target="predictions",
            )
            return
        text, widget = self._expected_hello_target()
        self._draw_guide(text, widget)
        self._guide_job = self.after(150, self._update_hello_guide)

    def _update_prediction_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "select_there":
            return
        if self._guide_back_to_keyboard_if_needed(self._update_prediction_guide):
            return
        words = [w.lower() for w in self.output_words]
        if words == ["hello", "there"]:
            self._tutorial_step = "tts_hello"
            self._show_ui_tutorial_text(
                "Gaze at text to speech for 'Hello there'",
                5000,
                self._update_tts_guide,
                target="tts",
            )
            return
        if words and words[0] != "hello":
            self._draw_guide("Gaze at Clear all, then type Hello again", getattr(self, "_clearall_btn", None))
            self._guide_job = self.after(150, self._update_prediction_guide)
            return
        if len(words) >= 2:
            bad_index = len(words) - 1
            if self.output_cursor == -1:
                self._draw_guide("Gaze at left arrow to choose the extra word", self.keyboard_buttons[0])
            elif self.output_cursor < bad_index:
                self._draw_guide("Gaze at right arrow to choose the extra word", self.keyboard_buttons[1])
            elif self.output_cursor > bad_index:
                self._draw_guide("Gaze at left arrow to choose the extra word", self.keyboard_buttons[0])
            else:
                self._draw_guide("Gaze at backspace to remove only this word", getattr(self, "_backspace_btn", None))
            self._guide_job = self.after(150, self._update_prediction_guide)
            return
        if self.current_input:
            self._draw_guide("Gaze at backspace, then gaze at 'there'", getattr(self, "_backspace_btn", None))
            self._guide_job = self.after(150, self._update_prediction_guide)
            return
        self._draw_guide("Gaze at 'there'", self._prediction_button("there"))
        self._guide_job = self.after(150, self._update_prediction_guide)

    def _update_edit_hello_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "edit_hello":
            return
        if self._guide_back_to_keyboard_if_needed(self._update_edit_hello_guide):
            return

        words = [w.lower() for w in self.output_words]
        if words[:2] == ["hi", "there"] and not self.current_input:
            self.output_cursor = -1
            self.update_display()
            self._tutorial_step = "tts_hi"
            self._show_ui_tutorial_text(
                "Gaze at text to speech for 'Hi there'",
                5000,
                self._update_tts_guide,
                target="tts",
            )
            return

        if len(words) < 2 or words[1] != "there":
            self._draw_guide("Gaze at Clear all, then type 'Hello there' again", getattr(self, "_clearall_btn", None))
            self._guide_job = self.after(150, self._update_edit_hello_guide)
            return

        if words[0] not in ("hello", "hi"):
            self._draw_guide("Gaze at Clear all, then type 'Hello there' again", getattr(self, "_clearall_btn", None))
            self._guide_job = self.after(150, self._update_edit_hello_guide)
            return

        if self.current_input and self.output_cursor != 0:
            self._draw_guide("Gaze at backspace, then choose 'Hello'", getattr(self, "_backspace_btn", None))
            self._guide_job = self.after(150, self._update_edit_hello_guide)
            return

        if self.current_input:
            text, widget = self._expected_hi_target()
            self._draw_guide(text, widget)
            self._guide_job = self.after(150, self._update_edit_hello_guide)
            return

        if self.output_cursor == -1:
            self._draw_guide("Gaze at left arrow to choose 'there'", self.keyboard_buttons[0])
        elif self.output_cursor == 1:
            self._draw_guide("Gaze at left arrow again to choose 'Hello'", self.keyboard_buttons[0])
        elif self.output_cursor == 0:
            text, widget = self._expected_hi_target()
            self._draw_guide(text or "Type 'Hi'", widget or self.letters_frame)
        else:
            self._draw_guide("Gaze at right arrow to choose 'Hello'", self.keyboard_buttons[1])
        self._guide_job = self.after(150, self._update_edit_hello_guide)

    def _update_tts_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step not in ("tts_hello", "tts_hi"):
            return
        btn = self.keyboard_buttons[4] if len(self.keyboard_buttons) >= 5 else None
        self._draw_guide("Gaze at text to speech", btn)
        self._guide_job = self.after(150, self._update_tts_guide)

    def _start_save_sentence_intro(self):
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        sentence = " ".join(self.output_words).strip()

        already_saved = bool(sentence and self.sentence_counts.get(sentence, 0) >= PREDEFINED_THRESHOLD)

        def after_save_intro():
            if already_saved:
                self._tutorial_step = "clear_before_predefined"
                self._show_ui_tutorial_text(
                    "This sentence is already saved.",
                    4000,
                    self._update_clear_before_predefined_guide,
                    target=None,
                )
                return
            self._show_ui_tutorial_text(
                "Gaze at text to speech 3 times to save it",
                5000,
                self._update_save_sentence_guide,
                target=None,
            )

        self._show_ui_tutorial_text(
            "Good job!",
            2500,
            lambda: self._show_ui_tutorial_text(
                "Now let's save 'Hi there' as a predefined sentence",
                4500,
                after_save_intro,
                target=None,
            ),
            target=None,
        )

    def _update_save_sentence_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "save_sentence":
            return
        if self._guide_back_to_keyboard_if_needed(self._update_save_sentence_guide):
            return

        sentence = " ".join(self.output_words).strip()
        if not sentence:
            self._show_ui_tutorial_text(
                "Type a sentence first, then gaze at text to speech to save it.",
                5000,
                self._finish_tutorial_from_predefined,
                target=None,
            )
            return

        count = self.sentence_counts.get(sentence, 0)
        if count >= PREDEFINED_THRESHOLD:
            self._tutorial_step = "clear_before_predefined"
            self._show_ui_tutorial_text(
                "Saved. You can find it in predefined sentences.",
                4500,
                self._update_clear_before_predefined_guide,
                target=None,
            )
            return

        remaining = PREDEFINED_THRESHOLD - count
        plural = "" if remaining == 1 else "s"
        btn = self.keyboard_buttons[4] if len(self.keyboard_buttons) >= 5 else None
        self._draw_guide(f"Gaze at text to speech {remaining} more time{plural}", btn)
        self._guide_job = self.after(150, self._update_save_sentence_guide)

    def _guide_back_to_keyboard_if_needed(self, next_callback):
        if not self._in_predefined_mode:
            return False
        btn = self.keyboard_buttons[3] if len(self.keyboard_buttons) >= 4 else None
        self._draw_guide("Gaze here to return to the keyboard", btn)
        self._guide_job = self.after(150, next_callback)
        return True

    def _update_clear_before_predefined_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "clear_before_predefined":
            return
        if self._in_predefined_mode:
            btn = self.keyboard_buttons[3] if len(self.keyboard_buttons) >= 4 else None
            self._draw_guide("Gaze here to return to the keyboard", btn)
            self._guide_job = self.after(150, self._update_clear_before_predefined_guide)
            return
        if self.output_words or self.current_input:
            self._draw_guide("Gaze at Clear all first", getattr(self, "_clearall_btn", None))
            self._guide_job = self.after(150, self._update_clear_before_predefined_guide)
            return
        self._tutorial_step = "predefined"
        self._start_predefined_intro()

    def _start_edit_hello_intro(self):
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        self._show_ui_tutorial_text(
            "Great job!",
            2500,
            lambda: self._show_ui_tutorial_text(
                "Now let's try editing 'Hello' to 'Hi'",
                4000,
                self._update_edit_hello_guide,
                target=None,
            ),
            target=None,
        )

    def _start_predefined_intro(self):
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        self._show_ui_tutorial_text(
            "Great job!",
            2500,
            lambda: self._show_ui_tutorial_text(
                "Now let's look at predefined sentences",
                4000,
                self._update_predefined_guide,
                target=None,
            ),
            target=None,
        )

    def _update_predefined_guide(self):
        self._cancel_guide_job()
        if self._tutorial_step != "predefined":
            return

        if self._tutorial_predefined_selected:
            self._show_ui_tutorial_text(
                "Nice. That loaded a predefined sentence.",
                4000,
                self._finish_tutorial_from_predefined,
                target=None,
            )
            return

        if self._in_predefined_mode:
            if not self.predefined_func_buttons:
                self._show_ui_tutorial_text(
                    "No predefined sentences yet. Gaze at text to speech 3 times to save one.",
                    6000,
                    self._finish_tutorial_from_predefined,
                    target=None,
                )
                return
            self._draw_guide("Gaze at a saved sentence", self.predefined_func_buttons[0])
            self._guide_job = self.after(150, self._update_predefined_guide)
            return

        btn = self.keyboard_buttons[3] if len(self.keyboard_buttons) >= 4 else None
        self._draw_guide("Gaze at predefined sentences", btn)
        self._guide_job = self.after(150, self._update_predefined_guide)

    def _finish_tutorial_from_predefined(self):
        self._tutorial_step = "familiarize"
        self._tutorial_input_paused = True
        if self._in_predefined_mode:
            self.predefined_sentence()
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        if hasattr(self, "status_bar"):
            self.status_bar.config(text="Tutorial familiarization: input is paused")
        self._destroy_guide_overlay()
        self._show_ui_tutorial_text(
            "Now familiarize yourself with the keyboard",
            5000,
            lambda: self._show_ui_tutorial_text(
                "Input is paused for now",
                5000,
                lambda: self.after(10000, self._show_finish_tutorial_button),
                target=None,
            ),
            target=None,
        )

    def _show_finish_tutorial_button(self):
        if self._finish_tutorial_btn is not None:
            return
        self._tutorial_step = "finish"
        btn = self._make_dwell_btn(
            self,
            self._finish_ui2_tutorial,
            text="Finish Tutorial",
            font=("Segoe UI", 20, "bold"),
            bg="#5865f2",
            fg="#ffffff",
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        btn.place(x=self._frame_gap(), y=self._frame_gap(), width=300, height=110)
        btn.lift()
        self._finish_tutorial_btn = btn
        self._dwell_reset_all()

    def _hide_finish_tutorial_button(self):
        if self._finish_tutorial_btn is None:
            return
        bid = id(self._finish_tutorial_btn)
        self.dwell_btn_meta.pop(bid, None)
        self.dwell_hover_ms.pop(bid, None)
        self.dwell_overlays.pop(bid, None)
        if self.dwell_hovered is self._finish_tutorial_btn:
            self.dwell_hovered = None
        try:
            self._finish_tutorial_btn.destroy()
        except Exception:
            pass
        self._finish_tutorial_btn = None

    def _finish_ui2_tutorial(self):
        self._tutorial_input_paused = False
        self._hide_finish_tutorial_button()
        self._tutorial_step = "done"
        self._show_ui_tutorial_text("Great! The keyboard is now ready\nGood luck!", 5000)

    def _tutorial_target_widget(self, target):
        if target == "keyboard":
            return self.letters_frame
        if target == "predictions":
            return self.predictive_container
        if target == "tts" and hasattr(self, "keyboard_buttons") and len(self.keyboard_buttons) >= 5:
            return self.keyboard_buttons[4]
        return None

    def _prediction_button(self, word):
        wanted = word.lower()
        for child in self.predictive_container.winfo_children():
            try:
                if child.cget("text").lower() == wanted:
                    return child
            except Exception:
                pass
        return self.predictive_container

    def _hide_tutorial_target_outline(self, widget):
        if config.DWELL_MODE != "async" or widget is None:
            return False
        try:
            active_btn = getattr(self, "_zoom_source_btn", None) or self.dwell_hovered
            if active_btn is None:
                return False
            if widget is active_btn:
                return self.dwell_hover_ms.get(id(active_btn), 0) > 0
            return self._point_in_widget(active_btn, widget.winfo_rootx(), widget.winfo_rooty())
        except Exception:
            return False

    def _ensure_guide_overlay(self):
        if self._guide_overlay is not None:
            return
        overlay = tk.Toplevel(self)
        overlay.withdraw()
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#010203", cursor=self.pointer_cursor)
        self._position_overlay(overlay)
        try:
            overlay.wm_attributes("-transparentcolor", "#010203")
        except Exception:
            pass
        try:
            overlay.wm_attributes("-disabled", True)
        except Exception:
            pass
        canvas = tk.Canvas(
            overlay,
            bg="#010203",
            highlightthickness=0,
            bd=0,
            cursor=self.pointer_cursor,
        )
        canvas.pack(fill="both", expand=True)
        self._guide_overlay = overlay
        self._guide_canvas = canvas
        overlay.deiconify()
        overlay.lift()

    def _destroy_guide_overlay(self):
        if self._guide_job is not None:
            try:
                self.after_cancel(self._guide_job)
            except Exception:
                pass
            self._guide_job = None
        if self._guide_overlay is not None:
            try:
                self._guide_overlay.destroy()
            except Exception:
                pass
        self._guide_overlay = None
        self._guide_canvas = None

    def _draw_guide(self, text, widget):
        self._ensure_guide_overlay()
        self._position_overlay(self._guide_overlay)
        canvas = self._guide_canvas
        if canvas is None:
            return
        canvas.delete("all")
        w = max(1, canvas.winfo_width())
        h = max(1, canvas.winfo_height())
        canvas.create_text(
            w // 2,
            90,
            text=text,
            fill="#ffffff",
            font=("Segoe UI", 42, "bold"),
            anchor="center",
        )
        if widget is None:
            return
        try:
            x1 = widget.winfo_rootx() - self.winfo_rootx()
            y1 = widget.winfo_rooty() - self.winfo_rooty()
            x2 = x1 + widget.winfo_width()
            y2 = y1 + widget.winfo_height()
        except Exception:
            return
        if x2 <= x1 or y2 <= y1:
            return
        if self._hide_tutorial_target_outline(widget):
            return
        pad = 8
        canvas.create_rectangle(x1 - pad, y1 - pad, x2 + pad, y2 + pad,
                                outline="#ffffff", width=5)
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2
        start_x = w // 2
        start_y = 140
        if target_y < 180:
            start_y = h // 2
        canvas.create_line(start_x, start_y, target_x, target_y,
                           fill="#ffffff", width=8, arrow=tk.LAST,
                           arrowshape=(28, 34, 12), smooth=True)

    def _show_ui_tutorial_text(self, text, duration_ms=5000, on_done=None, target=None):
        self._cancel_guide_job()
        self._destroy_guide_overlay()
        if self._tutorial_overlay is not None:
            return
        if self._tutorial_job is not None:
            try:
                self.after_cancel(self._tutorial_job)
            except Exception:
                pass
            self._tutorial_job = None
        self.update_idletasks()
        self._tutorial_prev_dwell = self.dwell_enabled
        self.dwell_enabled = False
        self._dwell_reset_all()

        overlay = tk.Toplevel(self)
        overlay.withdraw()
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.0)
        overlay.configure(bg="#000000", cursor=self.pointer_cursor)
        self._position_overlay(overlay)
        try:
            overlay.grab_set()
        except Exception:
            pass
        overlay.bind("<Button-1>", lambda _e: "break")
        overlay.bind("<Motion>", lambda _e: "break")
        overlay.bind("<KeyPress>", lambda _e: "break")

        canvas = tk.Canvas(
            overlay,
            bg="#000000",
            highlightthickness=0,
            bd=0,
            cursor=self.pointer_cursor,
        )
        canvas.pack(fill="both", expand=True)

        def draw_overlay(_event=None):
            canvas.delete("all")
            w = max(1, canvas.winfo_width())
            h = max(1, canvas.winfo_height())
            font_size = max(38, min(72, w // 19))

            widget = self._tutorial_target_widget(target)
            if widget is None:
                canvas.create_text(
                    w // 2,
                    h // 2,
                    text=text,
                    fill="#ffffff",
                    font=("Segoe UI", font_size, "bold"),
                    width=max(800, w - 260),
                    justify="center",
                    anchor="center",
                )
                return
            try:
                x1 = widget.winfo_rootx() - self.winfo_rootx()
                y1 = widget.winfo_rooty() - self.winfo_rooty()
                x2 = x1 + widget.winfo_width()
                y2 = y1 + widget.winfo_height()
            except Exception:
                return
            if x2 <= x1 or y2 <= y1:
                return

            text_y = h // 2
            if target == "keyboard":
                text_y = max(80, min(h // 4, y1 - 90))
            elif target == "predictions":
                text_y = min(h - 90, max((y2 + h) // 2, y2 + 120))
            elif target == "tts":
                text_y = min(h - 90, y2 + 120) if y2 < h // 2 else max(90, y1 - 120)

            canvas.create_text(
                w // 2,
                text_y,
                text=text,
                fill="#ffffff",
                font=("Segoe UI", font_size, "bold"),
                width=max(800, w - 260),
                justify="center",
                anchor="center",
            )

            pad = 10
            canvas.create_rectangle(
                x1 - pad, y1 - pad, x2 + pad, y2 + pad,
                outline="#ffffff",
                width=4,
            )

            target_x = (x1 + x2) // 2
            target_y = (y1 + y2) // 2
            start_x = w // 2
            start_y = text_y + 95
            if target_y < text_y:
                start_y = text_y - 95
            canvas.create_line(
                start_x, start_y, target_x, target_y,
                fill="#ffffff",
                width=8,
                arrow=tk.LAST,
                arrowshape=(28, 34, 12),
                smooth=True,
            )

        canvas.bind("<Configure>", draw_overlay)
        self.after(50, draw_overlay)
        self._tutorial_overlay = overlay
        self._tutorial_overlay_canvas = canvas
        overlay.deiconify()
        overlay.lift()
        self._fade_ui_tutorial_overlay(0.86, on_done=lambda: self._schedule_ui_tutorial_hide(duration_ms, on_done))

    def _fade_ui_tutorial_overlay(self, target_alpha, on_done=None, duration_ms=350, steps=12):
        if self._tutorial_overlay is None:
            if on_done:
                on_done()
            return
        try:
            start_alpha = float(self._tutorial_overlay.attributes("-alpha"))
        except Exception:
            start_alpha = 0.86
        delta = (target_alpha - start_alpha) / max(steps, 1)

        def tick(i=0):
            if self._tutorial_overlay is None:
                return
            alpha = target_alpha if i >= steps else start_alpha + delta * i
            try:
                self._tutorial_overlay.attributes("-alpha", max(0.0, min(0.86, alpha)))
            except Exception:
                pass
            if i >= steps:
                if on_done:
                    on_done()
                return
            self._tutorial_job = self.after(max(1, duration_ms // steps), lambda: tick(i + 1))

        tick()

    def _schedule_ui_tutorial_hide(self, duration_ms, on_done=None):
        self._tutorial_job = self.after(
            duration_ms,
            lambda: self._fade_ui_tutorial_overlay(
                0.0,
                on_done=lambda: self._hide_ui_tutorial_text(on_done),
            ),
        )

    def _hide_ui_tutorial_text(self, on_done=None):
        if self._tutorial_overlay is None:
            if on_done:
                on_done()
            return
        try:
            self._tutorial_overlay.grab_release()
        except Exception:
            pass
        try:
            self._tutorial_overlay.destroy()
        except Exception:
            pass
        self._tutorial_overlay = None
        self._tutorial_overlay_canvas = None
        if self._tutorial_prev_dwell is not None:
            self.dwell_enabled = self._tutorial_prev_dwell
        self._tutorial_prev_dwell = None
        self._dwell_reset_all()
        self._tutorial_job = None
        if on_done:
            on_done()

    # =========================================================================
    # WIDGET SETUP
    # =========================================================================
    def _frame_gap(self):
        return self.GAZE_FRAME_GAP_PX if self._gaze_tracking_active else 5

    def _top_control_gap(self):
        return 0

    def _create_widgets(self):
        theme = self.themes[self.current_theme]

        # ── Top area: text displays + PANIC BUTTON ────────────────────────────
        top_frame = tk.Frame(self, bg=theme["bg"])
        self.top_frame = top_frame
        top_frame.pack(fill="x", padx=self._top_control_gap(), pady=(0, 3))

        displays = tk.Frame(top_frame, bg=theme["bg"])
        self.displays_frame = displays
        displays.pack(side="left", fill="both", expand=True)

        self.input_display = tk.Text(displays, wrap="word", font=("Segoe UI", 18), height=2)
        self.input_display.pack(fill="both", expand=True)
        self.input_display.config(state="disabled")

        panic_bg = theme.get("panic_bg", "#8b0000")
        self.panic_btn = self._make_dwell_btn(
            top_frame, self.panic,
            text="PANIC\nBUTTON",
            font=("Segoe UI", 14, "bold"),
            bg=panic_bg, fg="white",
            relief="raised", bd=2, cursor="hand2", width=18,
        )
        self.panic_btn.pack(side="right", fill="y", padx=(8, 0))

        # ── Prediction bar ────────────────────────────────────────────────────
        self.predictive_container = tk.Frame(self, bg=theme["bg"])
        self.predictive_container.pack(fill="x", padx=self._top_control_gap(), pady=0)

        # ── Status bar (pack first with side=bottom so it anchors correctly) ───
        self.status_bar = ttk.Label(
            self,
            text="Gaze-based keyboard ready | S = Settings (caretaker)",
            relief="sunken", anchor="w", font=("Segoe UI", 8),
        )
        self.status_bar.pack(fill="x", side="bottom")

        # ── Shared content area — keyboard and predefined panel swap here ──────
        self.content_area = tk.Frame(self, bg=theme["bg"])
        self.content_area.pack(fill="both", expand=True, padx=self._top_control_gap(), pady=(0, self._frame_gap()))

        # Main grid: row 0 = func row, rows 1-3 = letters (all equal weight)
        self.main_grid = tk.Frame(self.content_area, bg=theme["bg"])
        self.main_grid.pack(fill="both", expand=True)
        for i in range(4):
            self.main_grid.grid_rowconfigure(i, weight=1, uniform="row")
        self.main_grid.grid_columnconfigure(0, weight=1)

        # Row 0: func row (always visible)
        self.func_row_frame = tk.Frame(self.main_grid, bg=theme["bg"])
        self.func_row_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._create_func_row(self.func_row_frame)

        # Rows 1-3: swappable area
        self.letters_frame    = tk.Frame(self.main_grid, bg=theme["bg"])
        self.predefined_frame = tk.Frame(self.main_grid, bg=theme["bg"])
        self.letters_frame.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=self._frame_gap())
        self._create_keyboard_area(self.letters_frame)

        self.apply_theme()
        self.update_display()
        self.after(500, self._dwell_start_trial)

    # =========================================================================
    # KEYBOARD LAYOUT
    # =========================================================================
    def _unregister_widgets(self, parent):
        """Remove parent descendants from dwell tracking before rebuilding them."""
        def descendants(widget):
            items = []
            for child in widget.winfo_children():
                items.append(child)
                items.extend(descendants(child))
            return items

        widgets = descendants(parent)

        for widget in widgets:
            bid = id(widget)
            self.dwell_btn_meta.pop(bid, None)
            self.dwell_hover_ms.pop(bid, None)
            self.dwell_overlays.pop(bid, None)

        self.keyboard_buttons = self.keyboard_buttons[:5] if hasattr(self, "keyboard_buttons") else []
        if hasattr(self, "_backspace_btn"):
            del self._backspace_btn
        if hasattr(self, "_clearall_btn"):
            del self._clearall_btn

        for widget in parent.winfo_children():
            widget.destroy()

    def _create_keyboard_area(self, parent):
        if self.ui_layout == "ui2":
            self._create_ui2_group_rows(parent)
        else:
            self._create_letter_rows(parent)

    def _create_func_row(self, parent):
        """Always-visible function row: ◄ ► SPACE Predefined 🔊"""
        theme    = self.themes[self.current_theme]
        func_bg  = theme.get("funckey_bg", theme["button_bg"])
        func_fg  = theme.get("funckey_fg", theme["button_fg"])
        func_abg = theme.get("funckey_active_bg", theme["button_active_bg"])
        parent.configure(bg=theme["bg"])

        self.keyboard_buttons = []   # func row occupies indices 0-4

        for w in parent.winfo_children():
            w.destroy()

        parent.grid_rowconfigure(0, weight=1)
        for col, w in enumerate([1, 1, 4, 3, 2]):
            parent.grid_columnconfigure(col, weight=w, uniform="fcol")

        middle_text = "Back" if self._ui2_letters_open else "⎵"
        middle_cmd = (lambda: self._create_ui2_group_rows(self.letters_frame)) if self._ui2_letters_open else self.finalize_word
        middle_size = 16 if self._ui2_letters_open else 22
        predefined_text = "Back\nto keys" if self._in_predefined_mode else "Predefined\nSentence"

        for col, (text, cmd, fsize) in enumerate([
            ("◄",                    self.move_word_left,      20),
            ("►",                    self.move_word_right,     20),
            (middle_text,             middle_cmd,                middle_size),
            (predefined_text,         self.predefined_sentence, 13),
            ("🔊",                   self.enter,               22),
        ]):
            btn = self._make_dwell_btn(
                parent, cmd,
                text=text, font=("Segoe UI", fsize, "bold"),
                bg=theme.get("tts_bg", func_bg) if col == 4 else func_bg,
                fg=func_fg,
                activebackground=theme.get("tts_active_bg", func_abg) if col == 4 else func_abg,
                activeforeground=func_fg,
                relief="raised", bd=1, cursor="hand2",
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=0, pady=0)
            self.keyboard_buttons.append(btn)

    def _create_letter_rows(self, parent):
        """Q-P / A-⌫ / Z-Clear all rows."""
        theme = self.themes[self.current_theme]
        parent.configure(bg=theme["bg"])
        self._unregister_widgets(parent)

        def btn_kw(**extra):
            return dict(
                bg=theme["button_bg"], fg=theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
                **extra,
            )

        main = tk.Frame(parent, bg=theme["bg"])
        main.pack(fill="both", expand=True)
        for i in range(3):
            main.grid_rowconfigure(i, weight=1, uniform="row")
        main.grid_columnconfigure(0, weight=1)

        for row_idx, chars in enumerate(["qwertyuiop", "asdfghjkl", "zxcvbnm"]):
            row = tk.Frame(main, bg=theme["bg"])
            row.grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)
            row.grid_rowconfigure(0, weight=1)

            for i, ch in enumerate(chars):
                row.grid_columnconfigure(i, weight=1, uniform="key")
                btn = self._make_dwell_btn(
                    row, lambda c=ch: self.insert_char(c),
                    text=ch.upper(), font=("Segoe UI", 22, "bold"),
                    **btn_kw(),
                )
                btn.grid(row=0, column=i, sticky="nsew", padx=1)
                self.keyboard_buttons.append(btn)

            if row_idx == 1:   # asdfghjkl → ⌫
                row.grid_columnconfigure(9, weight=1, uniform="key")
                bs = self._make_dwell_btn(
                    row, self.backspace,
                    text="⌫", font=("Segoe UI", 26, "bold"),
                    bg=theme.get("funckey_bg", theme["button_bg"]),
                    fg=theme.get("funckey_fg", theme["button_fg"]),
                    relief="raised", bd=1, cursor="hand2",
                )
                bs.grid(row=0, column=9, sticky="nsew", padx=1)
                self.keyboard_buttons.append(bs)
                self._backspace_btn = bs

            if row_idx == 2:   # zxcvbnm → Clear all
                row.grid_columnconfigure(7, weight=3, uniform="key")
                ca = self._make_dwell_btn(
                    row, self.clear_all,
                    text="Clear all", font=("Segoe UI", 16, "bold"),
                    bg=theme.get("funckey_bg", theme["button_bg"]),
                    fg=theme.get("funckey_fg", theme["button_fg"]),
                    relief="raised", bd=1, cursor="hand2",
                )
                ca.grid(row=0, column=7, sticky="nsew", padx=1)
                self.keyboard_buttons.append(ca)
                self._clearall_btn = ca

    def _create_ui2_group_rows(self, parent):
        """UI2 Design 6: grouped letter blocks."""
        theme = self.themes[self.current_theme]
        parent.configure(bg=theme["bg"])
        self._unregister_widgets(parent)
        self._ui2_letters_open = False
        self._ui2_group_buttons = {}
        self._ui2_letter_buttons = {}
        self._create_func_row(self.func_row_frame)

        def btn_kw(**extra):
            return dict(
                bg=theme["button_bg"], fg=theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
                **extra,
            )

        main = tk.Frame(parent, bg=theme["bg"])
        main.pack(fill="both", expand=True)
        for r in range(2):
            main.grid_rowconfigure(r, weight=1, uniform="ui2row")
        for c in range(4):
            main.grid_columnconfigure(c, weight=1, uniform="ui2col")

        cells = [
            ("A B C D", lambda: self._show_ui2_letters("abcd")),
            ("E F G H", lambda: self._show_ui2_letters("efgh")),
            ("I J K L", lambda: self._show_ui2_letters("ijkl")),
            ("M N O P", lambda: self._show_ui2_letters("mnop")),
            ("Q R S T U", lambda: self._show_ui2_letters("qrstu")),
            ("V W X Y Z", lambda: self._show_ui2_letters("vwxyz")),
            ("⌫",    self.backspace),
            ("Clear all", self.clear_all),
        ]

        for idx, (text, cmd) in enumerate(cells):
            group_key = text.replace(" ", "").lower()
            row, col = divmod(idx, 4)
            is_special = text in ("⌫", "Clear all")
            btn = self._make_dwell_btn(
                main, cmd,
                text=text, font=("Segoe UI", 22 if text != "Clear all" else 16, "bold"),
                bg=theme.get("funckey_bg", theme["button_bg"]) if is_special else theme["button_bg"],
                fg=theme.get("funckey_fg", theme["button_fg"]) if is_special else theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
            )
            btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            self.keyboard_buttons.append(btn)
            if text == "⌫":
                self._backspace_btn = btn
            elif text == "Clear all":
                self._clearall_btn = btn
            else:
                self._ui2_group_buttons[group_key] = btn

        self._dwell_reset_all()

    def _show_ui2_letters(self, letters):
        """UI2 Design 7: large individual letter choices for a selected group."""
        theme = self.themes[self.current_theme]
        self.letters_frame.configure(bg=theme["bg"])
        self._unregister_widgets(self.letters_frame)
        self._ui2_letters_open = True
        self._ui2_letter_buttons = {}
        self._create_func_row(self.func_row_frame)

        main = tk.Frame(self.letters_frame, bg=theme["bg"])
        main.pack(fill="both", expand=True)
        use_quadrants = len(letters) == 4
        if use_quadrants:
            for row in range(2):
                main.grid_rowconfigure(row, weight=1, uniform="ui2letterrow")
            for col in range(2):
                main.grid_columnconfigure(col, weight=1, uniform="ui2lettercol")
        else:
            main.grid_rowconfigure(0, weight=1)
            for col in range(len(letters)):
                main.grid_columnconfigure(col, weight=1, uniform="ui2letters")

        for idx, ch in enumerate(letters):
            row, col = divmod(idx, 2) if use_quadrants else (0, idx)
            btn = self._make_dwell_btn(
                main, lambda c=ch: self._insert_ui2_char(c),
                text=ch.upper(), font=("Segoe UI", 28, "bold"),
                bg=theme["button_bg"], fg=theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
            )
            btn.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
            self.keyboard_buttons.append(btn)
            self._ui2_letter_buttons[ch] = btn

        self._dwell_reset_all()
        self.status_bar.config(text=f"UI2 letter group: {letters.upper()}")

    def _insert_ui2_char(self, char):
        self.insert_char(char)
        self.after(220, lambda: self._create_ui2_group_rows(self.letters_frame))

    # =========================================================================
    # PREDEFINED SENTENCE PANEL
    # =========================================================================
    def _unregister_predefined_buttons(self):
        """Remove destroyed predefined panel buttons from the dwell engine."""
        for btn in list(self.predefined_func_buttons):
            bid = id(btn)
            self.dwell_btn_meta.pop(bid, None)
            self.dwell_hover_ms.pop(bid, None)
            self.dwell_overlays.pop(bid, None)
        self.predefined_func_buttons = []
        # Also purge any other registered buttons whose widget no longer exists
        for bid in list(self.dwell_btn_meta.keys()):
            widget, _ = self.dwell_btn_meta[bid]
            try:
                if not widget.winfo_exists():
                    self.dwell_btn_meta.pop(bid, None)
                    self.dwell_hover_ms.pop(bid, None)
                    self.dwell_overlays.pop(bid, None)
            except Exception:
                self.dwell_btn_meta.pop(bid, None)
                self.dwell_hover_ms.pop(bid, None)
                self.dwell_overlays.pop(bid, None)

    def _create_predefined_panel(self, parent):
        """Build the predefined sentence view inside parent frame."""
        theme = self.themes[self.current_theme]

        # Destroy old contents and unregister their dwell entries
        self._unregister_predefined_buttons()
        for w in parent.winfo_children():
            w.destroy()

        # ── Sentence buttons ──────────────────────────────────────────────────
        sentences = self._get_predefined_sentences()[:MAX_PREDEFINED_SENTENCES]
        content   = tk.Frame(parent, bg=theme["bg"])
        content.pack(fill="both", expand=True, padx=1, pady=1)

        if not sentences:
            tk.Label(
                content,
                text="No predefined sentences yet.\nGaze at text to speech 3x to auto-save one.",
                font=("Segoe UI", 18), bg=theme["bg"], fg=theme["suggestion_fg"],
                justify="center",
            ).pack(expand=True)
            return

        COLS = 3
        row_frame = None
        for idx, sentence in enumerate(sentences):
            if idx % COLS == 0:
                row_frame = tk.Frame(content, bg=theme["bg"])
                row_frame.pack(fill="both", expand=True, pady=2)
                for c in range(COLS):
                    row_frame.grid_columnconfigure(c, weight=1, uniform="sc")
                row_frame.grid_rowconfigure(0, weight=1)
            col = idx % COLS
            btn = self._make_dwell_btn(
                row_frame,
                lambda s=sentence: self._speak_predefined(s),
                text=sentence,
                font=("Segoe UI", 16, "bold"),
                bg=theme["button_bg"], fg=theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
                wraplength=380,
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=4)
            self.predefined_func_buttons.append(btn)

    def _speak_predefined(self, sentence):
        """Load a predefined sentence into the output and clear input."""
        if self._tutorial_step == "predefined":
            self._tutorial_predefined_selected = True
        self.output_words  = sentence.split()
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        self.predefined_sentence()   # switch back to keyboard
        self.status_bar.config(text=f"Loaded: '{sentence}'")

    def predefined_sentence(self):
        """Toggle between letter keys and predefined sentence panel."""
        if self._in_predefined_mode:
            self.predefined_frame.grid_remove()
            self.letters_frame.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=self._frame_gap())
            self._in_predefined_mode = False
            self._create_func_row(self.func_row_frame)
            self._update_current_word_overlay()
            self._dwell_reset_all()
            self.status_bar.config(text="Keyboard mode")
        else:
            self._hide_current_word_overlay()
            if self.ui_layout == "ui2" and self._ui2_letters_open:
                self._create_ui2_group_rows(self.letters_frame)
            self.letters_frame.grid_remove()
            self._create_predefined_panel(self.predefined_frame)
            self.predefined_frame.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=self._frame_gap())
            self._in_predefined_mode = True
            self._create_func_row(self.func_row_frame)
            self._dwell_reset_all()
            self.status_bar.config(text="Predefined sentences")

    # ── Sentence count tracking ───────────────────────────────────────────────
    def _load_sentence_counts(self):
        if os.path.exists(PREDEFINED_FILE):
            try:
                with open(PREDEFINED_FILE, "r", encoding="utf-8") as f:
                    self.sentence_counts = json.load(f)
                return
            except Exception:
                pass
        self.sentence_counts = {}

    def _save_sentence_counts(self):
        try:
            with open(PREDEFINED_FILE, "w", encoding="utf-8") as f:
                json.dump(self.sentence_counts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Could not save predefined sentences: {e}")

    def _get_predefined_sentences(self):
        return [s for s, c in self.sentence_counts.items() if c >= PREDEFINED_THRESHOLD]

    # =========================================================================
    # DISPLAY
    # =========================================================================
    def update_display(self):
        theme = self.themes[self.current_theme]
        # Fetch suggestions
        if self.current_input:
            ctx_words   = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
            context     = get_context_words(" ".join(ctx_words), n=2)
            suggestions = ngram_model.get_completion_suggestions(self.current_input, context, max_results=5)
            if suggestions:
                self.current_completion      = suggestions[0]
                self.alternative_suggestions = suggestions[1:5]
            else:
                self.current_completion      = self.current_input
                self.alternative_suggestions = []
        else:
            self.current_completion      = ""
            self.alternative_suggestions = []

        # ── Input display ─────────────────────────────────────────────────────
        self.input_display.config(state="normal")
        self.input_display.delete("1.0", "end")
        for i, word in enumerate(self.output_words):
            if i == self.output_cursor and self.current_input:
                self.input_display.insert("end", self.current_input, "editing")
                self.input_display.insert("end", "|", "cursor")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
            elif i == self.output_cursor:
                self.input_display.insert("end", word, "highlighted")
                self.input_display.insert("end", "|", "cursor")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
            else:
                self.input_display.insert("end", word, "normal")
                if i < len(self.output_words) - 1:
                    self.input_display.insert("end", " ")
        if self.output_cursor == -1 or self.output_cursor >= len(self.output_words):
            if self.output_words:
                self.input_display.insert("end", " ")
            if self.current_input:
                self.input_display.insert("end", self.current_input, "editing")
                self.input_display.insert("end", "|", "cursor")
            else:
                self.input_display.insert("end", "|", "cursor")
        self.input_display.tag_config("cursor",      foreground="red",             font=("Segoe UI", 32, "bold"))
        self.input_display.tag_config("highlighted",  foreground=theme["text_fg"],  background="yellow", font=("Segoe UI", 32))
        self.input_display.tag_config("editing",      foreground=theme["text_fg"],  font=("Segoe UI", 32))
        self.input_display.tag_config("normal",       foreground=theme["text_fg"],  font=("Segoe UI", 32))
        self.input_display.config(state="disabled")

        self._update_current_word_overlay()
        self.update_predictions()

    # =========================================================================
    # PREDICTIONS BAR
    # =========================================================================
    def update_predictions(self):
        for w in self.predictive_container.winfo_children():
            w.destroy()

        lang = config.PREDICTION_LANGUAGE
        if self.current_input:
            # Completion mode — suggest completions for the partial word being typed
            ctx_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
            context   = ctx_words[-2:] if len(ctx_words) >= 2 else ctx_words
            words     = ngram_model.get_completion_suggestions(self.current_input, context, max_results=4, language=lang)
            items     = [("word", word) for word in words]
        else:
            # Next-word mode — use the last 2 committed words directly as context
            context = self.output_words[-2:] if len(self.output_words) >= 2 else self.output_words
            phrase_items = []
            if cnn_phrase_model is not None and getattr(config, "ENABLE_CNN_PHRASE_SUGGESTIONS", True):
                try:
                    phrase_items = cnn_phrase_model.get_phrase_suggestions(
                        context,
                        max_results=getattr(config, "CNN_PHRASE_MAX_RESULTS", 2),
                        language=lang,
                    )
                except Exception:
                    phrase_items = []
            words   = ngram_model.get_next_word_suggestions(context, max_results=4, language=lang)
            if (
                self._ui_tutorial_enabled
                and self._tutorial_step in ("type_hello", "select_there", "tts_hello", "edit_hello", "tts_hi")
                and [w.lower() for w in self.output_words] == ["hello"]
                and "there" not in [w.lower() for w in words]
            ):
                words = ["there"] + words[:3]
            items = [("phrase", phrase) for phrase in phrase_items]
            phrase_words = {
                word
                for phrase in phrase_items
                for word in phrase.get("missing_words", [])
            }
            for word in words:
                if word not in phrase_words:
                    items.append(("word", word))
                if len(items) >= 4:
                    break

        theme = self.themes[self.current_theme]
        for kind, item in items[:4]:
            if kind == "phrase":
                button_text = item["phrase"]
                command = lambda phrase=item: self.apply_phrase_prediction(phrase)
                font = ("Segoe UI", 18, "bold")
                ipadx = 14
            else:
                button_text = item
                command = lambda word=item: (
                    self.apply_completion(word) if self.current_input else self.apply_prediction(word)
                )
                font = ("Segoe UI", 28, "bold")
                ipadx = 26
            btn = self._make_dwell_btn(
                self.predictive_container,
                command,
                text=button_text,
                font=font,
                relief="raised", bd=2, cursor="hand2",
                bg=theme["button_bg"], fg=theme["button_fg"],
            )
            btn.pack(side="left", padx=0, ipadx=ipadx, ipady=38, expand=True, fill="both")

    def apply_completion(self, word):
        """User selected a completion suggestion while typing (before space)."""
        self._commit_word(word)

    def apply_prediction(self, word):
        """User selected a next-word prediction (after space has been pressed)."""
        context = self.output_words[-2:] if len(self.output_words) >= 2 else self.output_words
        self.output_words.append(word)
        self.output_cursor = -1
        ngram_model.track_word_usage(word, context)
        self.update_display()
        self.status_bar.config(text=f"Predicted: '{word}'")

    def apply_phrase_prediction(self, phrase_result):
        """User selected a phrase suggestion; append only words not already typed."""
        missing_words = phrase_result.get("missing_words", [])
        if not missing_words:
            return
        context = self.output_words[-2:] if len(self.output_words) >= 2 else self.output_words
        for word in missing_words:
            ngram_model.track_word_usage(word, context)
            self.output_words.append(word)
            context = (context + [word])[-2:]
        self.output_cursor = -1
        self.current_input = ""
        self.current_completion = ""
        self.alternative_suggestions = []
        self.update_display()
        phrase = phrase_result.get("phrase", " ".join(missing_words))
        self.status_bar.config(text=f"Phrase: '{phrase}'")

    # =========================================================================
    # WORD COMMIT
    # =========================================================================
    def _commit_word(self, word):
        ctx_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
        context   = ctx_words[-2:] if len(ctx_words) >= 2 else ctx_words
        if self.current_input and self.current_input != word:
            ngram_model.learn_from_user_typing(self.current_input, word)
        ngram_model.track_word_usage(word, context)
        if self.output_cursor != -1 and self.output_cursor < len(self.output_words):
            self.output_words[self.output_cursor] = word
            self.output_cursor += 1
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
        else:
            self.output_words.append(word)
            self.output_cursor = -1
        self.current_input           = ""
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.update_display()
        self.status_bar.config(text=f"Selected: '{word}'")

    # =========================================================================
    # INPUT HANDLERS
    # =========================================================================
    def insert_char(self, char):
        self.current_input += char
        self.update_display()
        self.status_bar.config(text=f"Typing: '{self.current_input}'")

    def backspace(self):
        if self.current_input:
            self.current_input           = self.current_input[:-1]
            self.current_completion      = ""
            self.alternative_suggestions = []
            self.update_display()
            self.status_bar.config(text="Backspace")
        elif self.output_cursor == -1 and self.output_words:
            last = self.output_words.pop()
            self.current_input = last[:-1] if len(last) > 1 else ""
            self.update_display()
            self.status_bar.config(text=f"Editing: '{last}' → '{self.current_input}'")
        elif self.output_cursor != -1 and self.output_cursor < len(self.output_words):
            deleted = self.output_words.pop(self.output_cursor)
            if self.output_cursor >= len(self.output_words):
                self.output_cursor = -1
            self.update_display()
            self.status_bar.config(text=f"Deleted word: '{deleted}'")
        else:
            self.status_bar.config(text="Nothing to delete")

    def finalize_word(self):
        """SPACE — commits the literal typed input, never auto-completes."""
        if not self.current_input:
            self.status_bar.config(text="Nothing to finalize")
            return
        word = self.current_input   # always use exactly what was typed
        self._commit_word(word)
        self.status_bar.config(text=f"Added '{word}'")

    def enter(self):
        """🔊 — finalize current input, speak (TTS placeholder), track count, then clear."""
        tutorial_tts = self._tutorial_step if (
            self._ui_tutorial_enabled
            and self._tutorial_step in ("tts_hello", "tts_hi")
        ) else None
        if self.current_input:
            self.finalize_word()
        output_text = " ".join(self.output_words).strip()
        if output_text:
            speak(output_text)
            # Track usage count — auto-save to predefined after threshold
            self.sentence_counts[output_text] = self.sentence_counts.get(output_text, 0) + 1
            self._save_sentence_counts()
            count = self.sentence_counts[output_text]
            if count == PREDEFINED_THRESHOLD:
                self.status_bar.config(text=f"✓ Auto-saved to predefined: '{output_text}'")
            else:
                remaining = max(0, PREDEFINED_THRESHOLD - count)
                suffix = f" ({remaining} more to auto-save)" if remaining > 0 else ""
                self.status_bar.config(text=f"Spoken{suffix}")
        else:
            self.status_bar.config(text="Nothing to speak")
        self.update_display()
        if tutorial_tts == "tts_hello":
            self._tutorial_step = "edit_hello"
            self._destroy_guide_overlay()
            self._start_edit_hello_intro()
        elif tutorial_tts == "tts_hi":
            self._tutorial_step = "save_sentence"
            self.output_cursor = -1
            self.current_input = ""
            self.update_display()
            self._destroy_guide_overlay()
            self._start_save_sentence_intro()

    def clear_all(self):
        self.output_words            = []
        self.output_cursor           = -1
        self.current_input           = ""
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.update_display()
        self.status_bar.config(text="All cleared")


    def panic(self):
        """Start emergency alarm. Stop is intentionally keyboard-only: press 1."""
        if self._panic_active:
            self.status_bar.config(text="Alarm active | press 1 to stop")
            return
        panic_sound.start()
        self._panic_active = True
        self.dwell_enabled = False
        self.dwell_hovered = None
        self._dwell_reset_all()
        self._show_panic_stop_overlay()
        self.status_bar.config(text="Alarm active | press 1 to stop")

    def _stop_panic_shortcut(self, _event=None):
        if self._panic_active:
            panic_sound.stop()
            self._panic_active = False
            self._hide_panic_stop_overlay()
            self.dwell_enabled = True
            self.dwell_hovered = None
            self._dwell_reset_all()
            self.status_bar.config(text="Alarm stopped")
            return "break"
        return None

    def _show_panic_stop_overlay(self):
        self._hide_panic_stop_overlay()
        overlay = tk.Toplevel(self)
        overlay.withdraw()
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#050000", cursor=self.pointer_cursor, highlightthickness=0, bd=0)
        try:
            overlay.wm_attributes("-disabled", True)
        except Exception:
            pass
        self._position_overlay(overlay)

        label = tk.Label(
            overlay,
            text="PRESS 1 TO STOP",
            bg="#050000",
            fg="#ff2020",
            font=("Segoe UI", 72, "bold"),
            bd=0,
            highlightthickness=0,
        )
        label.place(relx=0.5, rely=0.52, anchor="center")
        self._panic_overlay = overlay
        self._panic_overlay_label = label
        self._panic_blink_visible = True
        overlay.deiconify()
        overlay.lift()
        self._blink_panic_stop_overlay()

    def _blink_panic_stop_overlay(self):
        if not self._panic_active or self._panic_overlay_label is None:
            return
        self._panic_blink_visible = not self._panic_blink_visible
        self._panic_overlay_label.config(fg="#ff2020" if self._panic_blink_visible else "#050000")
        if self._panic_overlay is not None:
            self._position_overlay(self._panic_overlay)
        self._panic_blink_job = self.after(450, self._blink_panic_stop_overlay)

    def _hide_panic_stop_overlay(self):
        if self._panic_blink_job is not None:
            try:
                self.after_cancel(self._panic_blink_job)
            except Exception:
                pass
            self._panic_blink_job = None
        if self._panic_overlay is not None:
            try:
                self._panic_overlay.destroy()
            except Exception:
                pass
        self._panic_overlay = None
        self._panic_overlay_label = None

    # =========================================================================
    # NAVIGATION
    # =========================================================================
    def move_word_left(self):
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        if self.output_cursor == -1:
            self.output_cursor = len(self.output_words) - 1
        elif self.output_cursor > 0:
            self.output_cursor -= 1
        self.current_input = ""
        self.update_display()
        self.status_bar.config(
            text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'"
        )

    def move_word_right(self):
        if not self.output_words:
            self.status_bar.config(text="No words in output")
            return
        if self.output_cursor == -1:
            self.status_bar.config(text="Already at end")
            return
        if self.output_cursor < len(self.output_words) - 1:
            self.output_cursor += 1
        else:
            self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        word_info = (
            f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'"
            if self.output_cursor != -1 else "Cursor at end — ready for new word"
        )
        self.status_bar.config(text=word_info)

    # =========================================================================
    # THEME
    # =========================================================================
    def apply_theme(self):
        theme = self.themes[self.current_theme]
        self.configure(bg=theme["bg"])
        self.input_display.config(bg=theme["input_bg"], fg=theme["text_fg"],
                                  insertbackground=theme["text_fg"])
        self.predictive_container.config(bg=theme["bg"])
        for frame_name in (
            "top_frame", "displays_frame", "content_area", "main_grid",
            "func_row_frame", "letters_frame", "predefined_frame",
        ):
            frame = getattr(self, frame_name, None)
            if frame is not None:
                try:
                    frame.configure(bg=theme["bg"])
                except Exception:
                    pass
        panic_bg = theme.get("panic_bg", "#660002")
        if hasattr(self, 'panic_btn'):
            self.panic_btn.config(text="PANIC\nBUTTON", bg=panic_bg, fg="white",
                                  font=("Segoe UI", 14, "bold"))
            self._apply_button_chrome(self.panic_btn)
        # keyboard_buttons[0..4] are the function row; rest are letter keys
        func_bg    = theme.get("funckey_bg",    theme["button_bg"])
        func_fg    = theme.get("funckey_fg",    theme["button_fg"])
        func_abg   = theme.get("funckey_active_bg", theme["button_active_bg"])
        special = set()
        if hasattr(self, '_backspace_btn'): special.add(id(self._backspace_btn))
        if hasattr(self, '_clearall_btn'):  special.add(id(self._clearall_btn))
        if hasattr(self, 'keyboard_buttons'):
            for i, btn in enumerate(self.keyboard_buttons):
                if i < 5 or id(btn) in special:
                    btn_bg = theme.get("tts_bg", func_bg) if i == 4 else func_bg
                    btn_abg = theme.get("tts_active_bg", func_abg) if i == 4 else func_abg
                    btn.config(bg=btn_bg, fg=func_fg,
                               activebackground=btn_abg, activeforeground=func_fg)
                else:
                    btn.config(bg=theme["button_bg"], fg=theme["button_fg"],
                               activebackground=theme["button_active_bg"],
                               activeforeground=theme["button_fg"])
                self._apply_button_chrome(btn)
        for widget in self.predictive_container.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(
                    bg=theme["button_bg"], fg=theme["button_fg"],
                    activebackground=theme["button_active_bg"],
                    activeforeground=theme["button_fg"],
                )
                self._apply_button_chrome(widget)

    def _toggle_zoom(self, enabled):
        config.DWELL_ZOOM = enabled
        self.status_bar.config(text=f"Key zoom: {'ON' if enabled else 'OFF'}")

    def _set_dwell_mode(self, mode):
        config.DWELL_MODE = mode
        self._dwell_reset_all()
        self.status_bar.config(text=f"Dwell mode: {mode.capitalize()}")

    def change_theme(self, theme, settings_window=None):
        self.current_theme = theme
        self.apply_theme()
        if hasattr(self, "func_row_frame"):
            self._create_func_row(self.func_row_frame)
        if hasattr(self, "letters_frame"):
            self._create_keyboard_area(self.letters_frame)
        if settings_window:
            self._close_settings_window()
        self.update_display()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")

    def _toggle_settings_dwell(self, enabled):
        self._toggle_dwell(enabled)
        if self._settings_open:
            self.dwell_enabled = False
            self.dwell_hovered = None
            self._dwell_reset_all()

    # =========================================================================
    # SETTINGS PANEL  (caretaker opens via physical 'S' key)
    # =========================================================================
    def show_settings(self):
        if self._settings_open:
            return
        self._show_system_pointer()
        self._pause_gaze_input_for_settings()
        settings_win = tk.Toplevel(self)
        self._settings_window = settings_win
        settings_win.title("Settings")
        settings_win.geometry("440x640")
        settings_win.resizable(False, True)
        settings_win.transient(self)
        settings_win.configure(bg="#1e1f22")
        settings_win.grab_set()
        self._apply_arrow_cursor(settings_win)
        settings_win.bind("<KeyPress-s>", self._close_settings_window)
        settings_win.bind("<KeyPress-S>", self._close_settings_window)

        settings_win.protocol("WM_DELETE_WINDOW", self._close_settings_window)

        settings_bg = "#1e1f22"
        card_bg = "#2b2d31"
        text_fg = "#dcddde"
        muted_fg = "#9da1a6"
        accent = "#5865f2"
        button_bg = "#313338"
        button_hover = "#3a3d43"
        try:
            style = ttk.Style(settings_win)
            style.theme_use("clam")
        except Exception:
            style = ttk.Style(settings_win)
        style.configure("Settings.TFrame", background=settings_bg)
        style.configure("Settings.TLabel", background=card_bg, foreground=text_fg)
        style.configure("SettingsMuted.TLabel", background=card_bg, foreground=muted_fg)
        style.configure("Settings.TLabelframe", background=card_bg, bordercolor="#42454a", relief="solid")
        style.configure("Settings.TLabelframe.Label", background=card_bg, foreground=text_fg,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Settings.TCheckbutton", background=card_bg, foreground=text_fg)
        style.map("Settings.TCheckbutton", background=[("active", card_bg)], foreground=[("active", "#ffffff")])
        style.configure("Settings.TRadiobutton", background=card_bg, foreground=text_fg)
        style.map("Settings.TRadiobutton", background=[("active", card_bg)], foreground=[("active", "#ffffff")])
        style.configure("Settings.TButton", background=button_bg, foreground=text_fg,
                        bordercolor="#42454a", focusthickness=0, padding=(12, 7))
        style.map("Settings.TButton", background=[("active", button_hover), ("pressed", accent)],
                  foreground=[("active", "#ffffff"), ("pressed", "#ffffff")])
        style.configure("Settings.Horizontal.TScale", background=card_bg, troughcolor="#171719")
        style.configure("Settings.Vertical.TScrollbar", background="#313338", troughcolor="#171719",
                        bordercolor="#171719", arrowcolor=text_fg, darkcolor="#313338",
                        lightcolor="#313338")

        # Scrollable container
        outer = tk.Frame(settings_win, bg=settings_bg)
        outer.pack(fill="both", expand=True)
        canvas  = tk.Canvas(outer, borderwidth=0, highlightthickness=0, bg=settings_bg)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview,
                                  style="Settings.Vertical.TScrollbar")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=settings_bg)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)

        def _on_frame(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_frame)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        inner.bind("<Destroy>", lambda _e: canvas.unbind_all("<MouseWheel>"))
        win = inner  # point all subsequent widgets at the scrollable inner frame

        # Theme
        tf = ttk.LabelFrame(win, text="Theme", padding=12, style="Settings.TLabelframe")
        tf.pack(fill="x", padx=20, pady=(15, 8))
        ttk.Label(tf, text="Select Theme:", font=("Segoe UI", 11),
                  style="Settings.TLabel").pack(anchor="w", pady=(0, 8))
        btn_row = tk.Frame(tf, bg=card_bg)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Light Mode", style="Settings.TButton",
                   command=lambda: self.change_theme("light", settings_win), width=18).pack(side="left", padx=(0, 10), ipady=8)
        ttk.Button(btn_row, text="Dark Mode", style="Settings.TButton",
                   command=lambda: self.change_theme("dark",  settings_win), width=18).pack(side="left", ipady=8)
        ttk.Label(tf, text=f"Current: {self.current_theme.capitalize()} Mode",
                  font=("Segoe UI", 9, "italic"),
                  style="SettingsMuted.TLabel").pack(anchor="w", pady=(8, 0))

        # Dwell Mode
        mf = ttk.LabelFrame(win, text="Dwell Mode", padding=12, style="Settings.TLabelframe")
        mf.pack(fill="x", padx=20, pady=(0, 8))
        mode_var = tk.StringVar(value=config.DWELL_MODE)
        ttk.Radiobutton(
            mf, text="Synchronous — accumulates across trial window, winner fires at end",
            variable=mode_var, value="sync",
            command=lambda: self._set_dwell_mode("sync"),
            style="Settings.TRadiobutton",
        ).pack(anchor="w")
        ttk.Radiobutton(
            mf, text="Asynchronous — zoom-on-dwell, fires as soon as threshold is reached",
            variable=mode_var, value="async",
            command=lambda: self._set_dwell_mode("async"),
            style="Settings.TRadiobutton",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            mf,
            text="Async mode works best with Key Zoom enabled.",
            font=("Segoe UI", 8, "italic"),
            style="SettingsMuted.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        # Dwell
        df = ttk.LabelFrame(win, text="Hover Dwell Input", padding=12, style="Settings.TLabelframe")
        df.pack(fill="x", padx=20, pady=8)
        dwell_var = tk.BooleanVar(value=config.DWELL_ENABLED)
        ttk.Checkbutton(df, text="Enable hover dwell input", variable=dwell_var,
                        command=lambda: self._toggle_settings_dwell(dwell_var.get()),
                        style="Settings.TCheckbutton").pack(anchor="w")
        ttk.Label(df, text="Hold duration — how long to hover before key fires:",
                  font=("Segoe UI", 9), style="Settings.TLabel").pack(anchor="w", pady=(12, 2))
        min_ms_var = tk.IntVar(value=config.DWELL_MIN_MS)
        min_row    = tk.Frame(df, bg=card_bg)
        min_row.pack(fill="x")
        min_label  = ttk.Label(min_row, text=f"{config.DWELL_MIN_MS} ms", width=7,
                               style="Settings.TLabel")
        min_label.pack(side="right")

        def on_slider(val):
            v = int(float(val) // 50) * 50
            min_label.config(text=f"{v} ms")
            min_ms_var.set(v)

        ttk.Scale(min_row, from_=200, to=1500, orient="horizontal",
                  style="Settings.Horizontal.TScale",
                  variable=min_ms_var, command=on_slider).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(df, text="Apply", style="Settings.TButton",
                   command=lambda: self._apply_min_hover(min_ms_var.get())).pack(anchor="e", pady=(8, 0))
        ttk.Label(df, text="Moving off a key resets its progress to zero.",
                  font=("Segoe UI", 8, "italic"),
                  style="SettingsMuted.TLabel").pack(anchor="w", pady=(6, 0))

        ttk.Label(df, text="Post-fire cooldown — pause after a key fires:",
                  font=("Segoe UI", 9), style="Settings.TLabel").pack(anchor="w", pady=(10, 2))
        cooldown_var   = tk.IntVar(value=config.DWELL_COOLDOWN_MS)
        cooldown_row   = tk.Frame(df, bg=card_bg)
        cooldown_row.pack(fill="x")
        cooldown_label = ttk.Label(cooldown_row, text=f"{config.DWELL_COOLDOWN_MS} ms", width=7,
                                   style="Settings.TLabel")
        cooldown_label.pack(side="right")

        def on_cooldown(val):
            v = int(float(val) // 50) * 50
            cooldown_label.config(text=f"{v} ms")
            cooldown_var.set(v)
            config.DWELL_COOLDOWN_MS = v

        ttk.Scale(cooldown_row, from_=0, to=2000, orient="horizontal",
                  style="Settings.Horizontal.TScale",
                  variable=cooldown_var, command=on_cooldown).pack(
                  side="left", fill="x", expand=True, padx=(0, 6))

        # Zoom
        zf = ttk.LabelFrame(win, text="Key Zoom (Gaze Stabiliser)", padding=12, style="Settings.TLabelframe")
        zf.pack(fill="x", padx=20, pady=8)
        zoom_var = tk.BooleanVar(value=config.DWELL_ZOOM)
        ttk.Checkbutton(
            zf, text="Zoom hovered key as dwell accumulates",
            variable=zoom_var,
            command=lambda: self._toggle_zoom(zoom_var.get()),
            style="Settings.TCheckbutton",
        ).pack(anchor="w")
        ttk.Label(
            zf,
            text="Enlarges the key you're gazing at to widen its hit area.",
            font=("Segoe UI", 8, "italic"),
            style="SettingsMuted.TLabel",
        ).pack(anchor="w", pady=(4, 8))

        ttk.Label(zf, text="Zoom start delay — how long to settle before zoom appears:",
                  font=("Segoe UI", 9), style="Settings.TLabel").pack(anchor="w")
        zoom_delay_var = tk.IntVar(value=config.DWELL_ZOOM_DELAY_MS)
        zoom_delay_row = tk.Frame(zf, bg=card_bg)
        zoom_delay_row.pack(fill="x", pady=(2, 0))
        zoom_delay_label = ttk.Label(zoom_delay_row, text=f"{config.DWELL_ZOOM_DELAY_MS} ms", width=7,
                                     style="Settings.TLabel")
        zoom_delay_label.pack(side="right")

        def on_zoom_delay(val):
            v = int(float(val) // 50) * 50
            zoom_delay_label.config(text=f"{v} ms")
            zoom_delay_var.set(v)
            config.DWELL_ZOOM_DELAY_MS = v

        ttk.Scale(zoom_delay_row, from_=0, to=800, orient="horizontal",
                  style="Settings.Horizontal.TScale",
                  variable=zoom_delay_var, command=on_zoom_delay).pack(
                  side="left", fill="x", expand=True, padx=(0, 6))

        # Prediction Language
        lf = ttk.LabelFrame(win, text="Prediction Language", padding=12, style="Settings.TLabelframe")
        lf.pack(fill="x", padx=20, pady=(0, 8))
        lang_var = tk.StringVar(value=config.PREDICTION_LANGUAGE)

        def _set_lang(val):
            config.PREDICTION_LANGUAGE = val
            self.update_predictions()

        ttk.Radiobutton(lf, text="Both (Filipino + English)",
                        variable=lang_var, value="both",
                        command=lambda: _set_lang("both"),
                        style="Settings.TRadiobutton").pack(anchor="w")
        ttk.Radiobutton(lf, text="Filipino only",
                        variable=lang_var, value="filipino",
                        command=lambda: _set_lang("filipino"),
                        style="Settings.TRadiobutton").pack(anchor="w", pady=(4, 0))
        ttk.Radiobutton(lf, text="English only",
                        variable=lang_var, value="english",
                        command=lambda: _set_lang("english"),
                        style="Settings.TRadiobutton").pack(anchor="w", pady=(4, 0))
        ttk.Label(lf, text="Filters autocomplete and next-word predictions.",
                  font=("Segoe UI", 8, "italic"),
                  style="SettingsMuted.TLabel").pack(anchor="w", pady=(6, 0))

        ttk.Button(win, text="Close", command=self._close_settings_window,
                   style="Settings.TButton").pack(pady=(8, 12))
        self._apply_arrow_cursor(settings_win)
