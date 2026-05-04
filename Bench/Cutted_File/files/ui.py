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
            "button_bg":        "#e0e0e0",
            "button_fg":        "black",
            "button_active_bg": "#d0d0d0",
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

        lbl = tk.Label(parent, relief=relief, bd=bd, **kwargs)
        lbl.bind('<Button-1>', lambda _e, c=command: c())
        self._dwell_register(lbl, command)
        return lbl

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
        self._pointer_overlay        = None
        self._pointer_canvas         = None
        self._pointer_job            = None
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
        if self._tutorial_job is not None:
            try:
                self.after_cancel(self._tutorial_job)
            except Exception:
                pass
            self._tutorial_job = None
        self._destroy_guide_overlay()
        self._hide_ui_tutorial_text()
        self._hide_finish_tutorial_button()
        self._show_system_cursor()
        self._destroy_pointer_overlay()
        super().destroy()

    def _keyboard_only_gaze_shortcut(self, event=None):
        if hasattr(self, "status_bar"):
            key = event.keysym.upper() if event else ""
            self.status_bar.config(text=f"{key} is available in gaze mode only")
        return "break"

    def _open_settings_shortcut(self, _event=None):
        if not self._settings_open:
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
            font=("Segoe UI", 14, "bold"),
            bg="#5865f2",
            fg="#ffffff",
            relief="raised",
            bd=2,
            cursor="hand2",
        )
        btn.place(x=self._frame_gap(), y=self._frame_gap(), width=190, height=54)
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

    def _create_widgets(self):
        theme = self.themes[self.current_theme]

        # ── Top area: text displays + PANIC BUTTON ────────────────────────────
        top_frame = tk.Frame(self, bg=theme["bg"])
        top_frame.pack(fill="x", padx=self._frame_gap(), pady=(self._frame_gap(), 3))

        displays = tk.Frame(top_frame, bg=theme["bg"])
        displays.pack(side="left", fill="both", expand=True)

        self.input_display = tk.Text(
            displays, wrap="word", font=("Segoe UI", 18), height=2
        )
        self.input_display.pack(fill="both", expand=True)
        self.input_display.config(state="disabled")

        panic_bg = theme.get("panic_bg", "#8b0000")
        self.panic_btn = self._make_dwell_btn(
            top_frame, self.panic,
            text="PANIC\nBUTTON",
            font=("Segoe UI", 14, "bold"),
            bg=panic_bg, fg="white",
            relief="raised", bd=2, cursor="hand2", width=10,
        )
        self.panic_btn.pack(side="right", fill="y", padx=(8, 0))

        # ── Prediction bar ────────────────────────────────────────────────────
        self.predictive_container = tk.Frame(self, bg=theme["bg"])
        self.predictive_container.pack(fill="x", padx=self._frame_gap(), pady=3)

        # ── Status bar (pack first with side=bottom so it anchors correctly) ───
        self.status_bar = ttk.Label(
            self,
            text="Gaze-based keyboard ready | S = Settings (caretaker)",
            relief="sunken", anchor="w", font=("Segoe UI", 8),
        )
        self.status_bar.pack(fill="x", side="bottom")

        # ── Shared content area — keyboard and predefined panel swap here ──────
        self.content_area = tk.Frame(self, bg=theme["bg"])
        self.content_area.pack(fill="both", expand=True, padx=self._frame_gap(), pady=(3, self._frame_gap()))

        # Main grid: row 0 = func row, rows 1-3 = letters (all equal weight)
        self.main_grid = tk.Frame(self.content_area, bg=theme["bg"])
        self.main_grid.pack(fill="both", expand=True)
        for i in range(4):
            self.main_grid.grid_rowconfigure(i, weight=1, uniform="row")
        self.main_grid.grid_columnconfigure(0, weight=1)

        # Row 0: func row (always visible)
        self.func_row_frame = tk.Frame(self.main_grid, bg=theme["bg"])
        self.func_row_frame.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._create_func_row(self.func_row_frame)

        # Rows 1-3: swappable area
        self.letters_frame    = tk.Frame(self.main_grid, bg=theme["bg"])
        self.predefined_frame = tk.Frame(self.main_grid, bg=theme["bg"])
        self.letters_frame.grid(row=1, column=0, rowspan=3, sticky="nsew")
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

        self.keyboard_buttons = []   # func row occupies indices 0-4

        for w in parent.winfo_children():
            w.destroy()

        parent.grid_rowconfigure(0, weight=1)
        for col, w in enumerate([1, 1, 5, 3, 1]):
            parent.grid_columnconfigure(col, weight=w, uniform="fcol")

        for col, (text, cmd, fsize) in enumerate([
            ("◄",                    self.move_word_left,      20),
            ("►",                    self.move_word_right,     20),
            ("⎵",                    self.finalize_word,       22),
            ("Predefined\nSentence", self.predefined_sentence, 13),
            ("🔊",                   self.enter,               22),
        ]):
            btn = self._make_dwell_btn(
                parent, cmd,
                text=text, font=("Segoe UI", fsize, "bold"),
                bg=func_bg, fg=func_fg,
                activebackground=func_abg, activeforeground=func_fg,
                relief="raised", bd=1, cursor="hand2",
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=1)
            self.keyboard_buttons.append(btn)

    def _create_letter_rows(self, parent):
        """Q-P / A-⌫ / Z-Clear all rows."""
        theme = self.themes[self.current_theme]
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
        self._unregister_widgets(parent)
        self._ui2_group_buttons = {}
        self._ui2_letter_buttons = {}

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
            ("ABCD", lambda: self._show_ui2_letters("abcd")),
            ("EFGH", lambda: self._show_ui2_letters("efgh")),
            ("IJKL", lambda: self._show_ui2_letters("ijkl")),
            ("MNOP", lambda: self._show_ui2_letters("mnop")),
            ("QRSTU", lambda: self._show_ui2_letters("qrstu")),
            ("VWXYZ", lambda: self._show_ui2_letters("vwxyz")),
            ("⌫",    self.backspace),
            ("Clear all", self.clear_all),
        ]

        for idx, (text, cmd) in enumerate(cells):
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
                self._ui2_group_buttons[text.lower()] = btn

        self._dwell_reset_all()

    def _show_ui2_letters(self, letters):
        """UI2 Design 7: large individual letter choices for a selected group."""
        theme = self.themes[self.current_theme]
        self._unregister_widgets(self.letters_frame)
        self._ui2_letter_buttons = {}

        main = tk.Frame(self.letters_frame, bg=theme["bg"])
        main.pack(fill="both", expand=True)
        main.grid_rowconfigure(0, weight=1)
        for col in range(len(letters)):
            main.grid_columnconfigure(col, weight=1, uniform="ui2letters")

        for col, ch in enumerate(letters):
            btn = self._make_dwell_btn(
                main, lambda c=ch: self._insert_ui2_char(c),
                text=ch.upper(), font=("Segoe UI", 28, "bold"),
                bg=theme["button_bg"], fg=theme["button_fg"],
                relief="raised", bd=1, cursor="hand2",
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=2, pady=2)
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
            self.letters_frame.grid(row=1, column=0, rowspan=3, sticky="nsew")
            self._in_predefined_mode = False
            self._dwell_reset_all()
            self.status_bar.config(text="Keyboard mode")
        else:
            self.letters_frame.grid_remove()
            self._create_predefined_panel(self.predefined_frame)
            self.predefined_frame.grid(row=1, column=0, rowspan=3, sticky="nsew")
            self._in_predefined_mode = True
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
            handler   = self.apply_completion
        else:
            # Next-word mode — use the last 2 committed words directly as context
            context = self.output_words[-2:] if len(self.output_words) >= 2 else self.output_words
            words   = ngram_model.get_next_word_suggestions(context, max_results=4, language=lang)
            handler = self.apply_prediction
            if (
                self._ui_tutorial_enabled
                and self._tutorial_step in ("type_hello", "select_there", "tts_hello", "edit_hello", "tts_hi")
                and [w.lower() for w in self.output_words] == ["hello"]
                and "there" not in [w.lower() for w in words]
            ):
                words = ["there"] + words[:3]

        theme = self.themes[self.current_theme]
        for word in words:
            btn = self._make_dwell_btn(
                self.predictive_container,
                lambda w=word: handler(w),
                text=word,
                font=("Segoe UI", 28, "bold"),
                relief="raised", bd=2, cursor="hand2",
                bg=theme["button_bg"], fg=theme["button_fg"],
            )
            btn.pack(side="left", padx=3, ipadx=26, ipady=38, expand=True, fill="both")

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
        """Toggle emergency alarm on/off."""
        if self._panic_active:
            panic_sound.stop()
            self._panic_active = False
            self.panic_btn.config(bg=self.themes[self.current_theme].get("panic_bg", "#660002"))
            self.status_bar.config(text="Alarm stopped")
        else:
            panic_sound.start()
            self._panic_active = True
            self.panic_btn.config(bg="#ff0000")
            self.status_bar.config(text="🚨 ALARM ACTIVE — press PANIC again to stop")

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
        panic_bg = theme.get("panic_bg", "#660002")
        if hasattr(self, 'panic_btn'):
            self.panic_btn.config(bg=panic_bg, fg="white")
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
                    btn.config(bg=func_bg, fg=func_fg,
                               activebackground=func_abg, activeforeground=func_fg)
                else:
                    btn.config(bg=theme["button_bg"], fg=theme["button_fg"],
                               activebackground=theme["button_active_bg"],
                               activeforeground=theme["button_fg"])
        for widget in self.predictive_container.winfo_children():
            if isinstance(widget, tk.Button):
                widget.config(
                    bg=theme["button_bg"], fg=theme["button_fg"],
                    activebackground=theme["button_active_bg"],
                    activeforeground=theme["button_fg"],
                )

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
        if settings_window:
            try:
                settings_window.grab_release()
            except Exception:
                pass
            settings_window.destroy()
            self._restore_main_cursor()
            self._take_focus()
        self.update_display()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")

    # =========================================================================
    # SETTINGS PANEL  (caretaker opens via physical 'S' key)
    # =========================================================================
    def show_settings(self):
        self._show_system_pointer()
        settings_win = tk.Toplevel(self)
        settings_win.title("Settings")
        settings_win.geometry("440x640")
        settings_win.resizable(False, True)
        settings_win.transient(self)
        settings_win.grab_set()
        self._apply_arrow_cursor(settings_win)

        def _close_settings():
            try:
                settings_win.grab_release()
            except Exception:
                pass
            settings_win.destroy()
            self._restore_main_cursor()
            self._take_focus()

        settings_win.protocol("WM_DELETE_WINDOW", _close_settings)

        # Scrollable container
        outer = tk.Frame(settings_win)
        outer.pack(fill="both", expand=True)
        canvas  = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas)
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
        tf = ttk.LabelFrame(win, text="Theme", padding=12)
        tf.pack(fill="x", padx=20, pady=(15, 8))
        ttk.Label(tf, text="Select Theme:", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 8))
        btn_row = tk.Frame(tf)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="☀ Light Mode",
                   command=lambda: self.change_theme("light", settings_win), width=18).pack(side="left", padx=(0, 10), ipady=8)
        ttk.Button(btn_row, text="🌙 Dark Mode",
                   command=lambda: self.change_theme("dark",  settings_win), width=18).pack(side="left", ipady=8)
        ttk.Label(tf, text=f"Current: {self.current_theme.capitalize()} Mode",
                  font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(8, 0))

        # Dwell Mode
        mf = ttk.LabelFrame(win, text="Dwell Mode", padding=12)
        mf.pack(fill="x", padx=20, pady=(0, 8))
        mode_var = tk.StringVar(value=config.DWELL_MODE)
        ttk.Radiobutton(
            mf, text="Synchronous — accumulates across trial window, winner fires at end",
            variable=mode_var, value="sync",
            command=lambda: self._set_dwell_mode("sync"),
        ).pack(anchor="w")
        ttk.Radiobutton(
            mf, text="Asynchronous — zoom-on-dwell, fires as soon as threshold is reached",
            variable=mode_var, value="async",
            command=lambda: self._set_dwell_mode("async"),
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            mf,
            text="Async mode works best with Key Zoom enabled.",
            font=("Segoe UI", 8, "italic"), foreground="gray",
        ).pack(anchor="w", pady=(6, 0))

        # Dwell
        df = ttk.LabelFrame(win, text="Hover Dwell Input", padding=12)
        df.pack(fill="x", padx=20, pady=8)
        dwell_var = tk.BooleanVar(value=self.dwell_enabled)
        ttk.Checkbutton(df, text="Enable hover dwell input", variable=dwell_var,
                        command=lambda: self._toggle_dwell(dwell_var.get())).pack(anchor="w")
        ttk.Label(df, text="Hold duration — how long to hover before key fires:",
                  font=("Segoe UI", 9)).pack(anchor="w", pady=(12, 2))
        min_ms_var = tk.IntVar(value=config.DWELL_MIN_MS)
        min_row    = tk.Frame(df)
        min_row.pack(fill="x")
        min_label  = ttk.Label(min_row, text=f"{config.DWELL_MIN_MS} ms", width=7)
        min_label.pack(side="right")

        def on_slider(val):
            v = int(float(val) // 50) * 50
            min_label.config(text=f"{v} ms")
            min_ms_var.set(v)

        ttk.Scale(min_row, from_=200, to=1500, orient="horizontal",
                  variable=min_ms_var, command=on_slider).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(df, text="Apply",
                   command=lambda: self._apply_min_hover(min_ms_var.get())).pack(anchor="e", pady=(8, 0))
        ttk.Label(df, text="Moving off a key resets its progress to zero.",
                  font=("Segoe UI", 8, "italic"), foreground="gray").pack(anchor="w", pady=(6, 0))

        ttk.Label(df, text="Post-fire cooldown — pause after a key fires:",
                  font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 2))
        cooldown_var   = tk.IntVar(value=config.DWELL_COOLDOWN_MS)
        cooldown_row   = tk.Frame(df)
        cooldown_row.pack(fill="x")
        cooldown_label = ttk.Label(cooldown_row, text=f"{config.DWELL_COOLDOWN_MS} ms", width=7)
        cooldown_label.pack(side="right")

        def on_cooldown(val):
            v = int(float(val) // 50) * 50
            cooldown_label.config(text=f"{v} ms")
            cooldown_var.set(v)
            config.DWELL_COOLDOWN_MS = v

        ttk.Scale(cooldown_row, from_=0, to=2000, orient="horizontal",
                  variable=cooldown_var, command=on_cooldown).pack(
                  side="left", fill="x", expand=True, padx=(0, 6))

        # Zoom
        zf = ttk.LabelFrame(win, text="Key Zoom (Gaze Stabiliser)", padding=12)
        zf.pack(fill="x", padx=20, pady=8)
        zoom_var = tk.BooleanVar(value=config.DWELL_ZOOM)
        ttk.Checkbutton(
            zf, text="Zoom hovered key as dwell accumulates",
            variable=zoom_var,
            command=lambda: self._toggle_zoom(zoom_var.get()),
        ).pack(anchor="w")
        ttk.Label(
            zf,
            text="Enlarges the key you're gazing at to widen its hit area.",
            font=("Segoe UI", 8, "italic"), foreground="gray",
        ).pack(anchor="w", pady=(4, 8))

        ttk.Label(zf, text="Zoom start delay — how long to settle before zoom appears:",
                  font=("Segoe UI", 9)).pack(anchor="w")
        zoom_delay_var = tk.IntVar(value=config.DWELL_ZOOM_DELAY_MS)
        zoom_delay_row = tk.Frame(zf)
        zoom_delay_row.pack(fill="x", pady=(2, 0))
        zoom_delay_label = ttk.Label(zoom_delay_row, text=f"{config.DWELL_ZOOM_DELAY_MS} ms", width=7)
        zoom_delay_label.pack(side="right")

        def on_zoom_delay(val):
            v = int(float(val) // 50) * 50
            zoom_delay_label.config(text=f"{v} ms")
            zoom_delay_var.set(v)
            config.DWELL_ZOOM_DELAY_MS = v

        ttk.Scale(zoom_delay_row, from_=0, to=800, orient="horizontal",
                  variable=zoom_delay_var, command=on_zoom_delay).pack(
                  side="left", fill="x", expand=True, padx=(0, 6))

        # Prediction Language
        lf = ttk.LabelFrame(win, text="Prediction Language", padding=12)
        lf.pack(fill="x", padx=20, pady=(0, 8))
        lang_var = tk.StringVar(value=config.PREDICTION_LANGUAGE)

        def _set_lang(val):
            config.PREDICTION_LANGUAGE = val
            self.update_predictions()

        ttk.Radiobutton(lf, text="Both (Filipino + English)",
                        variable=lang_var, value="both",
                        command=lambda: _set_lang("both")).pack(anchor="w")
        ttk.Radiobutton(lf, text="Filipino only",
                        variable=lang_var, value="filipino",
                        command=lambda: _set_lang("filipino")).pack(anchor="w", pady=(4, 0))
        ttk.Radiobutton(lf, text="English only",
                        variable=lang_var, value="english",
                        command=lambda: _set_lang("english")).pack(anchor="w", pady=(4, 0))
        ttk.Label(lf, text="Filters autocomplete and next-word predictions.",
                  font=("Segoe UI", 8, "italic"), foreground="gray").pack(anchor="w", pady=(6, 0))

        ttk.Button(win, text="Close", command=_close_settings).pack(pady=(8, 12))
        self._apply_arrow_cursor(settings_win)
