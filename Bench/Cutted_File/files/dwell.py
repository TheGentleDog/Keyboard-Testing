# =============================================================================
# dwell.py — Accumulative dwell engine
#
# Trial-based approach:
#   - Each trial lasts DWELL_MIN_MS (e.g. 600 ms)
#   - During the trial, hover time is ACCUMULATED per button — leaving a
#     button does NOT reset its counter
#   - At trial end, the button with the most accumulated time fires
#     (provided it has at least 1 poll tick of dwell)
#   - If any button reaches 100% mid-trial, it fires immediately
#   - After firing (or an empty trial), all counters reset and a new trial begins
# =============================================================================

import tkinter as tk
import config


class DwellMixin:

    def _dwell_init(self):
        self.dwell_enabled   = config.DWELL_ENABLED
        self.dwell_hover_ms  = {}   # btn_id → accumulated hover time this trial
        self.dwell_hovered   = None # currently hovered button widget
        self.dwell_trial_job = None
        self.dwell_overlays  = {}   # btn_id → Canvas progress bar
        self.dwell_btn_meta  = {}   # btn_id → (widget, command)
        self._dwell_trial_elapsed = 0  # ms elapsed in current trial

    # ── Registration ──────────────────────────────────────────────────────────
    def _dwell_register(self, btn, command):
        bid = id(btn)
        self.dwell_btn_meta[bid] = (btn, command)
        self.dwell_hover_ms[bid] = 0
        btn.bind("<Enter>", lambda e, b=btn: self._dwell_enter(b))
        btn.bind("<Leave>", lambda e, b=btn: self._dwell_leave(b))
        btn.bind("<Map>",   lambda e, b=btn: self._dwell_create_overlay(b))

    def _make_dwell_btn(self, parent, command, **kwargs):
        """Create a Label-based button that supports BOTH click and dwell."""
        kwargs.pop('command', None)
        relief = kwargs.pop('relief', 'flat')
        bd     = kwargs.pop('bd', 1)
        lbl = tk.Label(parent, relief=relief, bd=bd, **kwargs)
        lbl.bind('<Button-1>', lambda _e, c=command: c())
        self._dwell_register(lbl, command)
        return lbl

    # ── Overlay (progress bar) ─────────────────────────────────────────────
    def _dwell_create_overlay(self, btn):
        bid = id(btn)
        if bid in self.dwell_overlays:
            return
        try:
            w = btn.winfo_width()
            if w < 2:
                btn.after(100, lambda: self._dwell_create_overlay(btn))
                return
            theme  = self.themes[self.current_theme]
            canvas = tk.Canvas(btn.master, height=6, bd=0,
                               highlightthickness=0, bg=theme["dwell_bg"])
            canvas.place(in_=btn, relx=0, rely=1.0, anchor="sw",
                         relwidth=1.0, height=6)
            canvas.lift()
            canvas.create_rectangle(0, 0, 0, 6,
                                    fill=theme["dwell_bar"],
                                    outline="", tags="bar")
            self.dwell_overlays[bid] = canvas
        except Exception:
            pass

    # ── Enter / Leave — only track hover, NO reset on leave ───────────────────
    def _dwell_enter(self, btn):
        if not self.dwell_enabled:
            return
        self.dwell_hovered = btn
        try:
            btn.config(highlightthickness=3,
                       highlightbackground="#00cc44",
                       highlightcolor="#00cc44")
        except Exception:
            pass

    def _dwell_leave(self, btn):
        if not self.dwell_enabled:
            return
        if self.dwell_hovered is btn:
            self.dwell_hovered = None
        # Remove highlight border only — do NOT reset accumulated time
        try:
            btn.config(highlightthickness=0)
        except Exception:
            pass

    # ── Poll loop ─────────────────────────────────────────────────────────────
    def _dwell_start_trial(self):
        if self.dwell_trial_job:
            self.after_cancel(self.dwell_trial_job)
        self._dwell_reset_all()
        self._dwell_tick()

    def _dwell_reset_all(self):
        self._dwell_trial_elapsed = 0
        for bid in self.dwell_hover_ms:
            self.dwell_hover_ms[bid] = 0
        for canvas in self.dwell_overlays.values():
            try:
                canvas.coords("bar", 0, 0, 0, 6)
            except Exception:
                pass

    def _dwell_tick(self):
        if not self.dwell_enabled:
            self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)
            return

        # Accumulate hover time for the currently hovered button
        if self.dwell_hovered is not None:
            bid = id(self.dwell_hovered)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] += config.DWELL_POLL_MS

                # Update this button's progress bar
                if bid in self.dwell_overlays:
                    canvas = self.dwell_overlays[bid]
                    try:
                        cw = canvas.winfo_width()
                        if cw > 1:
                            progress = min(self.dwell_hover_ms[bid] / config.DWELL_MIN_MS, 1.0)
                            canvas.coords("bar", 0, 0, int(cw * progress), 6)
                    except Exception:
                        pass

                # Early fire — button hit 100% mid-trial
                if self.dwell_hover_ms[bid] >= config.DWELL_MIN_MS:
                    print(f"✓ Dwell early-fire at {self.dwell_hover_ms[bid]}ms")
                    self._dwell_fire(bid)
                    self._dwell_reset_all()
                    self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)
                    return

        self._dwell_trial_elapsed += config.DWELL_POLL_MS

        # Trial window ended — fire the winner
        if self._dwell_trial_elapsed >= config.DWELL_MIN_MS:
            winner_bid = max(self.dwell_hover_ms, key=self.dwell_hover_ms.get, default=None)
            if winner_bid is not None and self.dwell_hover_ms[winner_bid] > 0:
                print(f"✓ Dwell trial winner: {self.dwell_hover_ms[winner_bid]}ms accumulated")
                self._dwell_fire(winner_bid)
            self._dwell_reset_all()

        self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)

    # ── Fire & flash ──────────────────────────────────────────────────────────
    def _dwell_fire(self, bid):
        meta = self.dwell_btn_meta.get(bid)
        if meta:
            btn, command = meta
            self._dwell_flash(btn)
            try:
                command()
            except Exception as e:
                print(f"⚠ Dwell command error: {e}")

    def _dwell_flash(self, btn):
        theme = self.themes[self.current_theme]
        try:
            btn.config(bg="#00cc44", fg="#ffffff")
            btn.after(200, lambda: btn.config(
                bg=theme["button_bg"], fg=theme["button_fg"]))
        except Exception:
            pass

    # ── Settings helpers ──────────────────────────────────────────────────────
    def _toggle_dwell(self, enabled):
        config.DWELL_ENABLED = enabled
        self.dwell_enabled   = enabled
        self.status_bar.config(text=f"Hover dwell: {'ON' if enabled else 'OFF'}")

    def _apply_min_hover(self, ms):
        config.DWELL_MIN_MS = max(50, int(ms))
        self.status_bar.config(text=f"Minimum hover time set to {config.DWELL_MIN_MS} ms")
