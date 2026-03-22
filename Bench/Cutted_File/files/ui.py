# =============================================================================
# ui.py — FilipinoKeyboard UI (display, keyboard layout, popup, settings)
# =============================================================================

import tkinter as tk
from tkinter import ttk

import config
from dwell import DwellMixin
from model import ngram_model, get_context_words


class FilipinoKeyboard(tk.Tk, DwellMixin):

    THEMES = {
        "light": {
            "bg":               "#f0f0f0",
            "output_bg":        "#ffffff",
            "input_bg":         "#f9f9f9",
            "text_fg":          "black",
            "suggestion_fg":    "gray",
            "popup_bg":         "#fff9e6",
            "popup_border":     "#ffcc00",
            "popup_top_bg":     "#d4edda",
            "popup_top_fg":     "#155724",
            "button_bg":        "#e0e0e0",
            "button_fg":        "black",
            "button_active_bg": "#d0d0d0",
            "dwell_bar":        "#00cc44",
            "dwell_bg":         "#c8f0d8",
        },
        "dark": {
            "bg":               "#36393f",
            "output_bg":        "#2f3136",
            "input_bg":         "#40444b",
            "text_fg":          "#dcddde",
            "suggestion_fg":    "#8e9297",
            "popup_bg":         "#202225",
            "popup_border":     "#5865f2",
            "popup_top_bg":     "#1e3a27",
            "popup_top_fg":     "#88ffaa",
            "button_bg":        "#4f545c",
            "button_fg":        "#ffffff",
            "button_active_bg": "#5865f2",
            "dwell_bar":        "#55ff88",
            "dwell_bg":         "#1a3a28",
        },
    }

    def __init__(self):
        super().__init__()
        self.title("Filipino Keyboard - Live Autocomplete (Gaze-Based)")
        self.attributes('-fullscreen', True)
        self.bind('<Escape>', lambda e: self.attributes('-fullscreen', False))

        self.current_theme           = "light"
        self.themes                  = self.THEMES
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.current_input           = ""
        self.output_words            = []
        self.output_cursor           = -1
        self.alt_popup               = None
        self.shortcut_dialog         = None   # tracks open shortcut dialog

        self._dwell_init()
        self._create_widgets()

    # =========================================================================
    # WIDGET SETUP
    # =========================================================================
    def _create_widgets(self):
        self.output_display = tk.Text(self, wrap="word", font=("Segoe UI", 18), height=2)
        self.output_display.pack(fill="x", padx=5, pady=(5, 3))
        self.output_display.config(state="disabled")

        self.input_display = tk.Text(self, wrap="word", font=("Segoe UI", 16), height=1)
        self.input_display.pack(fill="x", padx=5, pady=3)
        self.input_display.config(state="disabled")

        self.predictive_container = ttk.Frame(self)
        self.predictive_container.pack(fill="x", padx=5, pady=3)

        keyboard_frame = tk.Frame(self, bg=self.themes[self.current_theme]["bg"])
        keyboard_frame.pack(fill="both", expand=True, padx=5, pady=(3, 5))
        self._create_keyboard(keyboard_frame)

        self.status_bar = ttk.Label(
            self,
            text="Type → SPACE to add word → ENTER to speak & clear | ◄► navigate words",
            relief="sunken", anchor="w", font=("Segoe UI", 8)
        )
        self.status_bar.pack(fill="x", side="bottom")

        self.apply_theme()
        self.update_display()
        self.after(500, self._dwell_start_trial)

    # =========================================================================
    # KEYBOARD LAYOUT
    # =========================================================================
    def _create_keyboard(self, parent):
        self.keyboard_buttons = []
        theme = self.themes[self.current_theme]

        main = tk.Frame(parent, bg=theme["bg"])
        main.pack(fill="both", expand=True)
        for i in range(5):
            main.grid_rowconfigure(i, weight=1, uniform="row")
        main.grid_columnconfigure(0, weight=1)

        # Row 0: ◄  CLEAR ALL  SETTINGS  ►
        row0 = tk.Frame(main, bg=theme["bg"])
        row0.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        row0.grid_rowconfigure(0, weight=1)
        for col, w in enumerate([1, 3, 3, 1]):
            row0.grid_columnconfigure(col, weight=w, uniform="func")
        for col, (text, cmd) in enumerate([
            ("◄",         self.move_word_left),
            ("CLEAR ALL", self.clear_all),
            ("⚙ SETTINGS",self.show_settings),
            ("►",         self.move_word_right),
        ]):
            btn = self._make_dwell_btn(
                row0, cmd, text=text,
                font=("Segoe UI", 20 if col in (0, 3) else 16, "bold"),
                relief="raised", bd=1, cursor="hand2"
            )
            btn.grid(row=0, column=col, sticky="nsew")
            self.keyboard_buttons.append(btn)

        # Rows 1-3: letter keys
        for row_idx, chars in enumerate(["qwertyuiop", "asdfghjkl", "zxcvbnm"], start=1):
            row = tk.Frame(main, bg=theme["bg"])
            row.grid(row=row_idx, column=0, sticky="nsew", padx=1, pady=1)
            row.grid_rowconfigure(0, weight=1)
            for i, ch in enumerate(chars):
                row.grid_columnconfigure(i, weight=1, uniform="key")
                btn = self._make_dwell_btn(
                    row, lambda c=ch: self.insert_char(c),
                    text=ch.upper(), font=("Segoe UI", 22, "bold"),
                    relief="raised", bd=1, cursor="hand2"
                )
                btn.grid(row=0, column=i, sticky="nsew")
                self.keyboard_buttons.append(btn)
            # Backspace on row 3
            if row_idx == 3:
                row.grid_columnconfigure(7, weight=2, uniform="key")
                bs = self._make_dwell_btn(
                    row, self.backspace, text="⌫",
                    font=("Segoe UI", 26, "bold"), relief="raised", bd=1, cursor="hand2"
                )
                bs.grid(row=0, column=7, sticky="nsew")
                self.keyboard_buttons.append(bs)

        # Row 4: SPACE + ENTER
        row4 = tk.Frame(main, bg=theme["bg"])
        row4.grid(row=4, column=0, sticky="nsew", padx=1, pady=1)
        row4.grid_rowconfigure(0, weight=1)
        row4.grid_columnconfigure(0, weight=85, uniform="bottom")
        row4.grid_columnconfigure(1, weight=15, uniform="bottom")
        sp = self._make_dwell_btn(row4, self.finalize_word, text="SPACE",
                                  font=("Segoe UI", 22, "bold"), relief="raised", bd=1, cursor="hand2")
        sp.grid(row=0, column=0, sticky="nsew")
        self.keyboard_buttons.append(sp)
        en = self._make_dwell_btn(row4, self.enter, text="↵",
                                  font=("Segoe UI", 26, "bold"), relief="raised", bd=1, cursor="hand2")
        en.grid(row=0, column=1, sticky="nsew")
        self.keyboard_buttons.append(en)

    # =========================================================================
    # DISPLAY
    # =========================================================================
    def update_display(self):
        theme = self.themes[self.current_theme]

        # Fetch suggestions
        if self.current_input:
            ctx_words = self.output_words[:self.output_cursor] if self.output_cursor != -1 else self.output_words
            context   = get_context_words(" ".join(ctx_words), n=2)
            suggestions = ngram_model.get_completion_suggestions(self.current_input, context, max_results=5)
            print(f"🔍 Input: '{self.current_input}' → Suggestions: {suggestions}")
            if suggestions:
                self.current_completion      = suggestions[0]
                self.alternative_suggestions = suggestions[1:5]
            else:
                self.current_completion      = self.current_input
                self.alternative_suggestions = []
        else:
            self.current_completion      = ""
            self.alternative_suggestions = []

        # Output display
        self.output_display.config(state="normal")
        self.output_display.delete("1.0", "end")
        for i, word in enumerate(self.output_words):
            is_cursor = (i == self.output_cursor)
            if is_cursor and self.current_input:
                self.output_display.insert("end", self.current_completion or self.current_input, "highlighted_word")
            elif is_cursor:
                self.output_display.insert("end", word, "highlighted_word")
            else:
                self.output_display.insert("end", word, "normal")
            if i < len(self.output_words) - 1:
                self.output_display.insert("end", " ")
        if self.output_cursor == -1 or self.output_cursor >= len(self.output_words):
            if self.output_words:
                self.output_display.insert("end", " ")
            if self.current_input:
                typed = self.current_input
                completion_rest = (self.current_completion[len(typed):]
                                   if self.current_completion.startswith(typed) else "")
                self.output_display.insert("end", typed, "input")
                if completion_rest:
                    self.output_display.insert("end", completion_rest, "suggestion")
        self.output_display.tag_config("cursor",           foreground="red",              font=("Segoe UI", 18, "bold"))
        self.output_display.tag_config("normal",           foreground=theme["text_fg"])
        self.output_display.tag_config("typing",           foreground=theme["text_fg"])
        self.output_display.tag_config("suggestion",       foreground=theme["suggestion_fg"])
        self.output_display.tag_config("input",            foreground=theme["text_fg"])
        self.output_display.tag_config("highlighted_word", foreground=theme["text_fg"],
                                       background="#ffe066" if self.current_theme == "light" else "#5865f2")
        self.output_display.config(state="disabled")

        # Input display
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
        self.input_display.tag_config("cursor",      foreground="red",   font=("Segoe UI", 16, "bold"))
        self.input_display.tag_config("highlighted",  foreground=theme["text_fg"], background="yellow")
        self.input_display.tag_config("editing",      foreground=theme["text_fg"])
        self.input_display.tag_config("normal",       foreground=theme["text_fg"])
        self.input_display.config(state="disabled")

        # Popup & predictions
        if self.current_input and self.alternative_suggestions:
            self.show_alternative_popup()
        else:
            self.close_popup()
        self.update_predictions()

    # =========================================================================
    # POPUP
    # =========================================================================
    def show_alternative_popup(self):
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None
        if not self.alternative_suggestions:
            return
        theme = self.themes[self.current_theme]
        self.alt_popup = tk.Toplevel(self)
        self.alt_popup.overrideredirect(True)
        self.alt_popup.attributes('-topmost', True)
        border = tk.Frame(self.alt_popup, bg=theme["popup_border"])
        border.pack(fill="both", expand=True)
        inner = tk.Frame(border, bg=theme["popup_bg"])
        inner.pack(fill="both", expand=True, padx=2, pady=2)
        all_words = [self.current_completion] + list(self.alternative_suggestions[:4])
        for idx, word in enumerate(all_words):
            is_top = (idx == 0)
            btn = tk.Button(
                inner, text=word,
                command=lambda w=word: self.apply_alternative_from_popup(w),
                font=("Segoe UI", 20, "bold") if is_top else ("Segoe UI", 18),
                relief="flat", bd=0, padx=20, pady=18, cursor="hand2",
                bg=theme["popup_top_bg"] if is_top else theme["popup_bg"],
                fg=theme["popup_top_fg"] if is_top else theme["text_fg"],
                activebackground=theme["button_active_bg"],
                activeforeground=theme["button_fg"],
            )
            btn.pack(side="left", fill="y")
            tk.Frame(inner, bg=theme["popup_border"], width=1).pack(side="left", fill="y")

        # ＋ button — always shown so users can add or redefine shortcuts freely
        add_btn = tk.Button(
            inner, text="＋",
            command=lambda: self._open_shortcut_dialog(self.current_input),
            font=("Segoe UI", 20, "bold"),
            relief="flat", bd=0, padx=20, pady=18, cursor="hand2",
            bg=theme["popup_bg"], fg="#4caf50",
            activebackground=theme["button_active_bg"],
            activeforeground="#4caf50",
        )
        add_btn.pack(side="left", fill="y")
        self.alt_popup.update_idletasks()
        self._position_popup()
        self.alt_popup.after(8000, self.close_popup)

    def _position_popup(self):
        if not self.alt_popup:
            return
        try:
            self.output_display.update_idletasks()
            char_offset = sum(len(w) + 1 for i, w in enumerate(self.output_words)
                              if not (i == self.output_cursor and self.current_input))
            start_idx = f"1.{char_offset}"
            end_idx   = f"1.{char_offset + len(self.current_completion)}"
            bbox_end   = self.output_display.bbox(end_idx)
            bbox_start = self.output_display.bbox(start_idx)
            wx, wy     = self.output_display.winfo_rootx(), self.output_display.winfo_rooty()
            pw, ph     = self.alt_popup.winfo_width(), self.alt_popup.winfo_height()
            sw, sh     = self.winfo_screenwidth(), self.winfo_screenheight()
            if bbox_end:
                ex, ey, ew, eh = bbox_end
                x, y = wx + ex + ew + 6, wy + ey + eh + 4
            elif bbox_start:
                sx, sy, sw2, sh2 = bbox_start
                x, y = wx + sx, wy + sy + sh2 + 4
            else:
                x, y = wx + 20, wy + self.output_display.winfo_height() + 4
            x = max(10, min(x, sw - pw - 10))
            y = max(0,  min(y, sh - ph - 10))
            self.alt_popup.geometry(f"+{x}+{y}")
        except Exception as e:
            print(f"⚠ Popup positioning error: {e}")

    def _open_shortcut_dialog(self, prefix):
        """
        Opens a small banner at the top of the screen showing the shortcut
        prompt and a live display of what the user is typing.
        The MAIN keyboard below remains fully active — insert_char() and
        backspace() redirect their input here while the dialog is open.
        SAVE / CANCEL are dwell buttons so the user never needs a physical key.
        """
        if self.shortcut_dialog and self.shortcut_dialog.winfo_exists():
            return
        self.close_popup()

        # Store the prefix so insert_char / backspace can reference it
        self._shortcut_prefix       = prefix
        self._shortcut_expansion_var = tk.StringVar()

        theme  = self.themes[self.current_theme]

        # Use a Toplevel pinned to the top of the screen, non-blocking
        # (no grab_set) so the main keyboard stays clickable/dwellable
        dialog = tk.Toplevel(self)
        dialog.title("")
        dialog.overrideredirect(True)
        dialog.attributes('-topmost', True)
        self.shortcut_dialog = dialog

        # ── Banner frame ──────────────────────────────────────────────────────
        border = tk.Frame(dialog, bg=theme["popup_border"], bd=0)
        border.pack(fill="both", expand=True)
        inner  = tk.Frame(border, bg=theme["popup_bg"], padx=16, pady=10)
        inner.pack(fill="both", expand=True, padx=2, pady=2)

        tk.Label(
            inner,
            text=f'Use the keyboard to type what  "{prefix}"  means, then SAVE:',
            font=("Segoe UI", 13),
            bg=theme["popup_bg"], fg=theme["text_fg"],
        ).pack(anchor="w", pady=(0, 6))

        # Live expansion display
        disp = tk.Label(
            inner,
            textvariable=self._shortcut_expansion_var,
            font=("Segoe UI", 26, "bold"),
            bg=theme["input_bg"], fg=theme["text_fg"],
            relief="sunken", bd=2,
            anchor="w", padx=10, width=20,
        )
        disp.pack(side="left", fill="y", padx=(0, 16))

        # SAVE and CANCEL as dwell buttons so gaze works normally
        def save_shortcut():
            expansion = self._shortcut_expansion_var.get().strip().lower()
            if not expansion:
                self.status_bar.config(text="⚠ Type the full word first.")
                return
            if len(prefix) >= len(expansion):
                self.status_bar.config(
                    text=f"⚠ '{expansion}' must be longer than '{prefix}'."
                )
                return
            ngram_model.learn_from_user_typing(prefix, expansion, force=True)
            ngram_model.load_user_learning()   # reload so shortcut works immediately
            self.status_bar.config(text=f"✓ Shortcut saved: '{prefix}' → '{expansion}'")
            self._close_shortcut_dialog()

        def cancel():
            self._close_shortcut_dialog()

        save_btn = self._make_dwell_btn(
            inner, save_shortcut,
            text="✓ SAVE",
            font=("Segoe UI", 14, "bold"),
            bg="#4caf50", fg="white",
            activebackground="#388e3c",
            relief="raised", bd=2, cursor="hand2",
            padx=20, pady=10,
        )
        save_btn.pack(side="left", padx=(0, 8))

        cancel_btn = self._make_dwell_btn(
            inner, cancel,
            text="✗ CANCEL",
            font=("Segoe UI", 14, "bold"),
            bg=theme["button_bg"], fg=theme["button_fg"],
            activebackground=theme["button_active_bg"],
            relief="raised", bd=2, cursor="hand2",
            padx=20, pady=10,
        )
        cancel_btn.pack(side="left")

        # Pin to top-centre of screen
        dialog.update_idletasks()
        sw   = self.winfo_screenwidth()
        dw   = dialog.winfo_width()
        dialog.geometry(f"+{(sw - dw) // 2}+0")

        self.status_bar.config(
            text=f"Shortcut mode: type what '{prefix}' means, then SAVE"
        )

    def _close_shortcut_dialog(self):
        """Tear down the shortcut dialog and restore normal keyboard input."""
        if self.shortcut_dialog and self.shortcut_dialog.winfo_exists():
            self.shortcut_dialog.destroy()
        self.shortcut_dialog         = None
        self._shortcut_prefix        = None
        self._shortcut_expansion_var = None
        self.status_bar.config(text="Ready")

    def close_popup(self):
        if self.alt_popup:
            self.alt_popup.destroy()
            self.alt_popup = None

    # =========================================================================
    # PREDICTIONS BAR
    # =========================================================================
    def update_predictions(self):
        for w in self.predictive_container.winfo_children():
            w.destroy()
        context     = get_context_words(" ".join(self.output_words), n=2)
        predictions = ngram_model.get_next_word_suggestions(context, max_results=6)
        for word in predictions:
            btn = tk.Button(
                self.predictive_container, text=word,
                command=lambda w=word: self.apply_prediction(w),
                font=("Segoe UI", 14, "bold"), relief="raised", bd=2, cursor="hand2"
            )
            btn.pack(side="left", padx=3, ipadx=20, ipady=12, expand=True, fill="both")

    def apply_prediction(self, word):
        context = get_context_words(" ".join(self.output_words), n=2)
        self.output_words.append(word)
        self.output_cursor = -1
        ngram_model.track_word_usage(word, context)
        self.update_display()
        self.status_bar.config(text=f"Predicted: '{word}'")

    # =========================================================================
    # WORD COMMIT
    # =========================================================================
    def apply_alternative_from_popup(self, word):
        self.close_popup()
        self._commit_word(word)

    def apply_alternative(self, word):
        self.close_popup()
        self._commit_word(word)

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
        # Redirect to shortcut dialog if it is open
        if self.shortcut_dialog and self.shortcut_dialog.winfo_exists():
            self._shortcut_expansion_var.set(
                self._shortcut_expansion_var.get() + char
            )
            return
        self.current_input += char
        self.update_display()
        self.status_bar.config(text=f"Typing: '{self.current_input}'")

    def backspace(self):
        # Redirect to shortcut dialog if it is open
        if self.shortcut_dialog and self.shortcut_dialog.winfo_exists():
            val = self._shortcut_expansion_var.get()
            if val:
                self._shortcut_expansion_var.set(val[:-1])
            return
        print(f"\n⌫ BACKSPACE: input='{self.current_input}' words={self.output_words} cursor={self.output_cursor}")
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
        if not self.current_input:
            self.status_bar.config(text="Nothing to finalize")
            return
        self.close_popup()
        word = self.current_completion if self.current_completion else self.current_input
        print(f"\n🔹 FINALIZE: input='{self.current_input}' → '{word}'")
        self._commit_word(word)
        self.status_bar.config(text=f"Added '{word}'")

    def enter(self):
        if self.current_input:
            self.finalize_word()
        output_text = " ".join(self.output_words)
        if output_text.strip():
            print(f"🔊 TTS: {output_text.strip()}")
        self.output_words  = []
        self.output_cursor = -1
        self.current_input = ""
        self.update_display()
        self.status_bar.config(text="Spoken and cleared")

    def clear_all(self):
        self.output_words            = []
        self.output_cursor           = -1
        self.current_input           = ""
        self.current_completion      = ""
        self.alternative_suggestions = []
        self.update_display()
        self.status_bar.config(text="All cleared")

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
        self.status_bar.config(text=f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'")

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
        word_info = f"Cursor at word {self.output_cursor + 1}: '{self.output_words[self.output_cursor]}'" \
                    if self.output_cursor != -1 else "Cursor at end — ready for new word"
        self.status_bar.config(text=word_info)

    # =========================================================================
    # THEME
    # =========================================================================
    def apply_theme(self):
        theme = self.themes[self.current_theme]
        self.configure(bg=theme["bg"])
        self.output_display.config(bg=theme["output_bg"], fg=theme["text_fg"],
                                   insertbackground=theme["text_fg"])
        self.input_display.config(bg=theme["input_bg"], fg=theme["text_fg"],
                                  insertbackground=theme["text_fg"])
        self.output_display.tag_config("input",      foreground=theme["text_fg"])
        self.output_display.tag_config("suggestion", foreground=theme["suggestion_fg"])
        if hasattr(self, 'keyboard_buttons'):
            for btn in self.keyboard_buttons:
                btn.config(bg=theme["button_bg"], fg=theme["button_fg"],
                           activebackground=theme["button_active_bg"],
                           activeforeground=theme["button_fg"])
        if hasattr(self, 'predictive_container'):
            for widget in self.predictive_container.winfo_children():
                if isinstance(widget, tk.Button):
                    widget.config(bg=theme["button_bg"], fg=theme["button_fg"],
                                  activebackground=theme["button_active_bg"],
                                  activeforeground=theme["button_fg"])

    def change_theme(self, theme, settings_window=None):
        self.current_theme = theme
        self.apply_theme()
        if settings_window:
            settings_window.destroy()
        self.status_bar.config(text=f"Theme changed to {theme.capitalize()} Mode")

    # =========================================================================
    # SETTINGS PANEL
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
                   command=lambda: self.change_theme("dark", win),  width=18).pack(side="left", ipady=8)
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
