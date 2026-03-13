# =============================================================================
# dwell.py — Synchronous dwell hover engine
# (based on Algorithm 2 from Meena & Salvi, 2025)
#
# Trial period  ∆t2 = DWELL_TRIAL_MS  (e.g. 1000 ms)
# Poll interval      = DWELL_POLL_MS   (e.g. 50 ms  → 20 frames/trial)
# Selection          : btn fires when cumulative hover >= DWELL_MIN_MS
# =============================================================================

import tkinter as tk
import config


class DwellMixin:
    """
    Mixin class that adds dwell-hover input to any tkinter Tk/Toplevel window.
    The host class must call _dwell_init() inside __init__ before using
    _make_dwell_btn().
    """

    def _dwell_init(self):
        """Initialise dwell state — call once from host __init__."""
        self.dwell_enabled   = config.DWELL_ENABLED
        self.dwell_hover_ms  = {}   # btn_id → cumulative hover time (ms)
        self.dwell_hovered   = None # currently hovered button widget
        self.dwell_trial_job = None # after() handle for poll loop
        self.dwell_overlays  = {}   # btn_id → Canvas progress bar
        self.dwell_btn_meta  = {}   # btn_id → (widget, command_callable)

    # ── Registration ──────────────────────────────────────────────────────────
    def _dwell_register(self, btn, command):
        """Register a button for dwell hover interaction."""
        bid = id(btn)
        self.dwell_btn_meta[bid] = (btn, command)
        self.dwell_hover_ms[bid] = 0
        btn.bind("<Enter>", lambda e, b=btn: self._dwell_enter(b))
        btn.bind("<Leave>", lambda e, b=btn: self._dwell_leave(b))
        btn.bind("<Map>",   lambda e, b=btn: self._dwell_create_overlay(b))

    def _make_dwell_btn(self, parent, command, **kwargs):
        """Create a Button that supports BOTH click and dwell."""
        btn = tk.Button(parent, command=command, **kwargs)
        self._dwell_register(btn, command)
        return btn

    # ── Overlay (progress bar) ─────────────────────────────────────────────
    def _dwell_create_overlay(self, btn):
        """Create a slim 6-px Canvas progress bar at the bottom of btn."""
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

    # ── Enter / Leave ─────────────────────────────────────────────────────────
    def _dwell_enter(self, btn):
        """Mouse entered — highlight border green."""
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
        """Mouse left — remove highlight and reset that button's counter."""
        if not self.dwell_enabled:
            return
        if self.dwell_hovered is btn:
            self.dwell_hovered = None
        try:
            btn.config(highlightthickness=0)
        except Exception:
            pass
        bid = id(btn)
        self.dwell_hover_ms[bid] = 0
        if bid in self.dwell_overlays:
            try:
                self.dwell_overlays[bid].coords("bar", 0, 0, 0, 6)
            except Exception:
                pass

    # ── Poll loop ─────────────────────────────────────────────────────────────
    def _dwell_start_trial(self):
        """Start the dwell polling loop."""
        if self.dwell_trial_job:
            self.after_cancel(self.dwell_trial_job)
        self._dwell_reset_all()
        self._dwell_tick()

    def _dwell_reset_all(self):
        """Zero all hover times and progress bars."""
        for bid in self.dwell_hover_ms:
            self.dwell_hover_ms[bid] = 0
        for canvas in self.dwell_overlays.values():
            try:
                canvas.coords("bar", 0, 0, 0, 6)
            except Exception:
                pass

    def _dwell_tick(self):
        """Poll every DWELL_POLL_MS — accumulate hover time and fire on threshold."""
        if not self.dwell_enabled:
            self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)
            return

        if self.dwell_hovered is not None:
            bid = id(self.dwell_hovered)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] += config.DWELL_POLL_MS
                elapsed = self.dwell_hover_ms[bid]

                # Update progress bar
                if bid in self.dwell_overlays:
                    canvas = self.dwell_overlays[bid]
                    try:
                        cw = canvas.winfo_width()
                        if cw > 1:
                            progress = min(elapsed / config.DWELL_MIN_MS, 1.0)
                            canvas.coords("bar", 0, 0, int(cw * progress), 6)
                    except Exception:
                        pass

                # Threshold reached → fire
                if elapsed >= config.DWELL_MIN_MS:
                    print(f"✓ Dwell fired after {elapsed}ms")
                    self._dwell_fire(bid)
                    self._dwell_reset_all()

        self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)

    # ── Fire & flash ──────────────────────────────────────────────────────────
    def _dwell_fire(self, bid):
        """Fire the command for the given button id."""
        meta = self.dwell_btn_meta.get(bid)
        if meta:
            btn, command = meta
            self._dwell_flash(btn)
            try:
                command()
            except Exception as e:
                print(f"⚠ Dwell command error: {e}")

    def _dwell_flash(self, btn):
        """Green flash — visual confirmation feedback."""
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
