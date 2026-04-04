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

PREDEFINED_FILE      = "predefined_sentences.json"
PREDEFINED_THRESHOLD = 3   # times spoken before auto-saving


class FilipinoKeyboard(tk.Tk, DwellMixin):

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

    # ── Override dwell flash to restore correct per-button colour ─────────────
    def _dwell_flash(self, btn):
        theme = self.themes[self.current_theme]
        func_keys = (self.keyboard_buttons[:5] if hasattr(self, 'keyboard_buttons') else []) + \
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
            btn.after(200, lambda: btn.config(bg=restore_bg, fg=restore_fg))
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

        lbl = tk.Label(parent, relief=relief, bd=bd, **kwargs)
        lbl.bind('<Button-1>', lambda _e, c=command: c())
        self._dwell_register(lbl, command)
        return lbl

    def __init__(self):
        super().__init__()
        self.title("Filipino Keyboard - Gaze-Based")
        self.attributes('-fullscreen', True)
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))
        self.bind('<s>', lambda e: self.show_settings())   # caretaker shortcut

        self.current_theme           = "dark"
        self.themes                  = self.THEMES
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.current_input           = ""
        self.output_words            = []
        self.output_cursor           = -1
        self._in_predefined_mode     = False
        self.predefined_func_buttons = []   # function row in predefined panel

        self._dwell_init()
        self._load_sentence_counts()
        self._create_widgets()

    # =========================================================================
    # WIDGET SETUP
    # =========================================================================
    def _create_widgets(self):
        theme = self.themes[self.current_theme]

        # ── Top area: text displays + PANIC BUTTON ────────────────────────────
        top_frame = tk.Frame(self, bg=theme["bg"])
        top_frame.pack(fill="x", padx=5, pady=(5, 3))

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
        self.predictive_container.pack(fill="x", padx=5, pady=3)

        # ── Status bar (pack first with side=bottom so it anchors correctly) ───
        self.status_bar = ttk.Label(
            self,
            text="Gaze-based keyboard ready | S = Settings (caretaker)",
            relief="sunken", anchor="w", font=("Segoe UI", 8),
        )
        self.status_bar.pack(fill="x", side="bottom")

        # ── Shared content area — keyboard and predefined panel swap here ──────
        self.content_area = tk.Frame(self, bg=theme["bg"])
        self.content_area.pack(fill="both", expand=True, padx=5, pady=(3, 5))

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
        self._create_letter_rows(self.letters_frame)

        self.apply_theme()
        self.update_display()
        self.after(500, self._dwell_start_trial)

    # =========================================================================
    # KEYBOARD LAYOUT
    # =========================================================================
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
                    **btn_kw(),
                )
                bs.grid(row=0, column=9, sticky="nsew", padx=1)
                self.keyboard_buttons.append(bs)

            if row_idx == 2:   # zxcvbnm → Clear all
                row.grid_columnconfigure(7, weight=3, uniform="key")
                ca = self._make_dwell_btn(
                    row, self.clear_all,
                    text="Clear all", font=("Segoe UI", 16, "bold"),
                    **btn_kw(),
                )
                ca.grid(row=0, column=7, sticky="nsew", padx=1)
                self.keyboard_buttons.append(ca)

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
        sentences = self._get_predefined_sentences()
        content   = tk.Frame(parent, bg=theme["bg"])
        content.pack(fill="both", expand=True, padx=1, pady=1)

        if not sentences:
            tk.Label(
                content,
                text="No predefined sentences yet.\nSpeak a sentence 3× to auto-save it.",
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

    def _speak_predefined(self, sentence):
        """Load a predefined sentence into the output and clear input."""
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

        if self.current_input:
            # Completion mode — suggest completions for the partial word being typed
            ctx_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
            context   = get_context_words(" ".join(ctx_words), n=2)
            words     = ngram_model.get_completion_suggestions(self.current_input, context, max_results=4)
            handler   = self.apply_completion
        else:
            # Next-word mode — suggest likely following words
            context = get_context_words(" ".join(self.output_words), n=2)
            words   = ngram_model.get_next_word_suggestions(context, max_results=4)
            handler = self.apply_prediction

        theme = self.themes[self.current_theme]
        for word in words:
            btn = self._make_dwell_btn(
                self.predictive_container,
                lambda w=word: handler(w),
                text=word,
                font=("Segoe UI", 22, "bold"),
                relief="raised", bd=2, cursor="hand2",
                bg=theme["button_bg"], fg=theme["button_fg"],
            )
            btn.pack(side="left", padx=3, ipadx=20, ipady=30, expand=True, fill="both")

    def apply_completion(self, word):
        """User selected a completion suggestion while typing (before space)."""
        self._commit_word(word)

    def apply_prediction(self, word):
        """User selected a next-word prediction (after space has been pressed)."""
        context = get_context_words(" ".join(self.output_words), n=2)
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
        context   = get_context_words(" ".join(ctx_words), n=2)
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
        if self.current_input:
            self.finalize_word()
        output_text = " ".join(self.output_words).strip()
        if output_text:
            print(f"🔊 TTS (not yet implemented): {output_text}")
            # Track usage count — auto-save to predefined after threshold
            self.sentence_counts[output_text] = self.sentence_counts.get(output_text, 0) + 1
            self._save_sentence_counts()
            count = self.sentence_counts[output_text]
            if count == PREDEFINED_THRESHOLD:
                self.status_bar.config(text=f"✓ Auto-saved to predefined: '{output_text}'")
            else:
                remaining = max(0, PREDEFINED_THRESHOLD - count)
                suffix = f" ({remaining} more to auto-save)" if remaining > 0 else ""
                self.status_bar.config(text=f"Spoken and cleared{suffix}")
        else:
            self.status_bar.config(text="Spoken and cleared")
        self.output_words  = []
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()

    def clear_all(self):
        self.output_words            = []
        self.output_cursor           = -1
        self.current_input           = ""
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.update_display()
        self.status_bar.config(text="All cleared")


    def panic(self):
        """Placeholder — panic button (to be implemented)."""
        self.status_bar.config(text="PANIC — coming soon")

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
        if hasattr(self, 'keyboard_buttons'):
            for i, btn in enumerate(self.keyboard_buttons):
                if i < 5:   # function row
                    btn.config(bg=func_bg, fg=func_fg,
                               activebackground=func_abg, activeforeground=func_fg)
                else:        # letter keys
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

    def change_theme(self, theme, settings_window=None):
        self.current_theme = theme
        self.apply_theme()
        if settings_window:
            settings_window.destroy()
        self.update_display()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")

    # =========================================================================
    # SETTINGS PANEL  (caretaker opens via physical 'S' key)
    # =========================================================================
    def show_settings(self):
        win = tk.Toplevel(self)
        win.title("Settings")
        win.geometry("420x380")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        # Theme
        tf = ttk.LabelFrame(win, text="Theme", padding=12)
        tf.pack(fill="x", padx=20, pady=(15, 8))
        ttk.Label(tf, text="Select Theme:", font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 8))
        btn_row = tk.Frame(tf)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="☀ Light Mode",
                   command=lambda: self.change_theme("light", win), width=18).pack(side="left", padx=(0, 10), ipady=8)
        ttk.Button(btn_row, text="🌙 Dark Mode",
                   command=lambda: self.change_theme("dark",  win), width=18).pack(side="left", ipady=8)
        ttk.Label(tf, text=f"Current: {self.current_theme.capitalize()} Mode",
                  font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(8, 0))

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

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(8, 12))
