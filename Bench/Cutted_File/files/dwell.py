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
        self.dwell_enabled        = config.DWELL_ENABLED
        self.dwell_hover_ms       = {}
        self.dwell_hovered        = None
        self.dwell_trial_job      = None
        self.dwell_overlays       = {}
        self.dwell_btn_meta       = {}
        self._dwell_trial_elapsed = 0
        self._zoom_popup          = None
        self._zoom_source_btn     = None
        self._dwell_cooldown_ms   = 0      # remaining cooldown after a fire

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
        return

    # ── Enter / Leave — only track hover, NO reset on leave ───────────────────
    def _dwell_enter(self, btn):
        if not self.dwell_enabled:
            return
        self.dwell_hovered = btn

    def _dwell_leave(self, btn):
        if not self.dwell_enabled:
            return
        if self.dwell_hovered is btn:
            self.dwell_hovered = None
        # Async mode: leaving resets the counter (continuous hold required)
        # BUT don't reset if cursor moved onto the zoom popup (it's our own overlay)
        if config.DWELL_MODE == "async":
            if self._zoom_popup is not None:
                try:
                    mx = self.winfo_pointerx()
                    my = self.winfo_pointery()
                    zx = self._zoom_popup.winfo_rootx()
                    zy = self._zoom_popup.winfo_rooty()
                    zw = self._zoom_popup.winfo_width()
                    zh = self._zoom_popup.winfo_height()
                    if zx <= mx <= zx + zw and zy <= my <= zy + zh:
                        # Cursor went to our zoom popup — keep tracking this button
                        self.dwell_hovered = btn
                        return
                except Exception:
                    pass
            bid = id(btn)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] = 0
            if bid in self.dwell_overlays:
                try:
                    self.dwell_overlays[bid].coords("bar", 0, 0, 0, 6)
                except Exception:
                    pass
            self._zoom_hide()

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

        self._dwell_refresh_hover_target()

        # Cooldown — pause after a key fires
        if self._dwell_cooldown_ms > 0:
            self._dwell_cooldown_ms -= config.DWELL_POLL_MS
            self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)
            return

        if config.DWELL_MODE == "async":
            self._dwell_tick_async()
        else:
            self._dwell_tick_sync()

        self.dwell_trial_job = self.after(config.DWELL_POLL_MS, self._dwell_tick)

    def _dwell_refresh_hover_target(self):
        """Poll the current pointer target so stable gaze still accumulates dwell."""
        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
            target = self._dwell_target_at(px, py)
        except Exception:
            return

        if target is None:
            if config.DWELL_MODE == "async" and self.dwell_hovered is not None:
                self._dwell_leave(self.dwell_hovered)
            else:
                self.dwell_hovered = None
            return

        if target is not self.dwell_hovered:
            if config.DWELL_MODE == "async" and self.dwell_hovered is not None:
                self._dwell_leave(self.dwell_hovered)
            self._dwell_enter(target)

    def _dwell_target_at(self, px, py):
        """Return the registered dwell button at screen point px/py."""
        if self._point_in_zoom_popup(px, py):
            return self._zoom_source_btn

        for btn, _command in self.dwell_btn_meta.values():
            if self._point_in_widget(btn, px, py):
                return btn
        return None

    def _point_in_zoom_popup(self, px, py):
        if self._zoom_popup is None or self._zoom_source_btn is None:
            return False
        return self._point_in_widget(self._zoom_popup, px, py)

    def _point_in_widget(self, widget, px, py):
        try:
            if not widget.winfo_ismapped():
                return False
            wx = widget.winfo_rootx()
            wy = widget.winfo_rooty()
            ww = widget.winfo_width()
            wh = widget.winfo_height()
            return wx <= px <= wx + ww and wy <= py <= wy + wh
        except Exception:
            return False

    def _resolve_dwell_target(self, widget):
        """Map the widget under the pointer to a registered dwell button."""
        current = widget
        while current is not None:
            if id(current) in self.dwell_btn_meta:
                return current
            if self._zoom_popup is not None and current in (self._zoom_popup, getattr(self, "_zoom_lbl", None)):
                return self._zoom_source_btn
            current = getattr(current, "master", None)
        return None

    # ── Async mode: fires as soon as the hovered key hits threshold ───────────
    def _dwell_tick_async(self):
        if self.dwell_hovered is not None:
            bid = id(self.dwell_hovered)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] += config.DWELL_POLL_MS
                progress = min(self.dwell_hover_ms[bid] / config.DWELL_MIN_MS, 1.0)
                self._update_bar(bid, progress)
                self._zoom_show(self.dwell_hovered, progress)
                if self.dwell_hover_ms[bid] >= config.DWELL_MIN_MS:
                    print(f"✓ Async dwell fire at {self.dwell_hover_ms[bid]}ms")
                    self._zoom_hide()
                    self._dwell_fire(bid)
                    self._dwell_reset_all()
        else:
            self._zoom_hide()

    # ── Sync mode: accumulates across trial window, winner fires at end ───────
    def _dwell_tick_sync(self):
        if self.dwell_hovered is not None:
            bid = id(self.dwell_hovered)
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] += config.DWELL_POLL_MS
                progress = min(self.dwell_hover_ms[bid] / config.DWELL_MIN_MS, 1.0)
                self._update_bar(bid, progress)
                # Early fire if one key dominates mid-trial
                if self.dwell_hover_ms[bid] >= config.DWELL_MIN_MS:
                    print(f"✓ Sync early-fire at {self.dwell_hover_ms[bid]}ms")
                    self._dwell_fire(bid)
                    self._dwell_reset_all()
                    return

        self._dwell_trial_elapsed += config.DWELL_POLL_MS
        if self._dwell_trial_elapsed >= config.DWELL_MIN_MS:
            winner_bid = max(self.dwell_hover_ms, key=self.dwell_hover_ms.get, default=None)
            if winner_bid is not None and self.dwell_hover_ms[winner_bid] > 0:
                print(f"✓ Sync trial winner: {self.dwell_hover_ms[winner_bid]}ms")
                self._dwell_fire(winner_bid)
            self._dwell_reset_all()

    def _update_bar(self, bid, progress):
        if bid in self.dwell_overlays:
            canvas = self.dwell_overlays[bid]
            try:
                cw = canvas.winfo_width()
                if cw > 1:
                    canvas.coords("bar", 0, 0, int(cw * progress), 6)
            except Exception:
                pass

    # ── Fire & flash ──────────────────────────────────────────────────────────
    def _dwell_fire(self, bid):
        meta = self.dwell_btn_meta.get(bid)
        if meta:
            btn, command = meta
            self._dwell_flash(btn)
            self._dwell_cooldown_ms = config.DWELL_COOLDOWN_MS
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

    # ── Zoom overlay ──────────────────────────────────────────────────────────
    def _zoom_show(self, btn, progress: float):
        """
        Show/update a full enlarged copy of btn centered on it.
        The overlay grows from 1× to 2× the button's actual pixel size.
        Its Enter/Leave bindings feed the same button's dwell counter,
        so the wider hit area is the real dwell target.
        """
        if not config.DWELL_ZOOM:
            return
        # Don't start zooming until gaze has settled for DWELL_ZOOM_DELAY_MS
        bid = id(btn)
        elapsed = self.dwell_hover_ms.get(bid, 0)
        if elapsed < config.DWELL_ZOOM_DELAY_MS:
            self._zoom_hide()
            return
        try:
            theme = self.themes[self.current_theme]
            text  = btn.cget("text")
            bg    = btn.cget("bg")
            fg    = btn.cget("fg")

            bx = btn.winfo_rootx()
            by = btn.winfo_rooty()
            bw = btn.winfo_width()
            bh = btn.winfo_height()

            # Scale: 1.0× at 0% → 2.0× at 100%
            scale = 1.0 + 1.0 * progress
            pw    = int(bw * scale)
            ph    = int(bh * scale)
            cx    = bx + bw // 2
            cy    = by + bh // 2
            x     = max(0, min(cx - pw // 2, self.winfo_screenwidth()  - pw))
            y     = max(0, min(cy - ph // 2, self.winfo_screenheight() - ph))

            # Font size scales with the button
            base_sz = 22
            try:
                font_cfg = btn.cget("font")
                if isinstance(font_cfg, (list, tuple)):
                    base_sz = int(font_cfg[1])
                elif isinstance(font_cfg, str):
                    parts = font_cfg.split()
                    base_sz = int(parts[1]) if len(parts) > 1 else 22
            except Exception:
                pass
            font_sz = max(18, int(base_sz * scale))

            if self._zoom_popup is None:
                self._zoom_popup      = tk.Toplevel(self)
                self._zoom_source_btn = btn
                self._zoom_popup.overrideredirect(True)
                self._zoom_popup.attributes('-topmost', True)
                self._zoom_popup.configure(
                    cursor=getattr(self, "pointer_cursor", "arrow"),
                    bg=theme["bg"],
                    highlightbackground=theme.get("button_active_bg", theme["button_bg"]),
                    highlightthickness=2,
                )
                self._zoom_lbl = tk.Label(
                    self._zoom_popup,
                    text=text,
                    font=("Segoe UI", font_sz, "bold"),
                    bg=bg, fg=fg,
                    relief="flat", bd=0,
                    justify="center",
                    cursor=getattr(self, "pointer_cursor", "arrow"),
                    padx=10, pady=10,
                )
                self._zoom_lbl.pack(fill="both", expand=True)
                # Bind Enter/Leave on the zoom popup itself
                for w in (self._zoom_popup, self._zoom_lbl):
                    w.bind("<Enter>", lambda e, b=btn: self._dwell_enter(b))
                    w.bind("<Leave>", lambda e, b=btn: self._zoom_popup_leave(b))
            else:
                self._zoom_source_btn = btn
                self._zoom_lbl.config(
                    text=text,
                    font=("Segoe UI", font_sz, "bold"),
                    bg=bg, fg=fg,
                    cursor=getattr(self, "pointer_cursor", "arrow"),
                )
                self._zoom_popup.configure(
                    cursor=getattr(self, "pointer_cursor", "arrow"),
                    bg=theme["bg"],
                    highlightbackground=theme.get("button_active_bg", theme["button_bg"]),
                )

            self._zoom_popup.geometry(f"{pw}x{ph}+{x}+{y}")
        except Exception:
            pass

    def _zoom_hide(self):
        if self._zoom_popup:
            try:
                self._zoom_popup.destroy()
            except Exception:
                pass
            self._zoom_popup = None
            self._zoom_source_btn = None

    def _zoom_popup_leave(self, btn):
        """Called when gaze leaves the zoom popup — check if it went back to the original button."""
        try:
            mx = self.winfo_pointerx()
            my = self.winfo_pointery()
            bx = btn.winfo_rootx()
            by = btn.winfo_rooty()
            bw = btn.winfo_width()
            bh = btn.winfo_height()
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                # Went back to original button — keep tracking
                return
        except Exception:
            pass
        # Truly left — reset this button's counter
        if config.DWELL_MODE == "async":
            bid = id(btn)
            self.dwell_hovered = None
            if bid in self.dwell_hover_ms:
                self.dwell_hover_ms[bid] = 0
            if bid in self.dwell_overlays:
                try:
                    self.dwell_overlays[bid].coords("bar", 0, 0, 0, 6)
                except Exception:
                    pass
            self._zoom_hide()

    # ── Settings helpers ──────────────────────────────────────────────────────
    def _toggle_dwell(self, enabled):
        config.DWELL_ENABLED = enabled
        self.dwell_enabled   = enabled
        self.status_bar.config(text=f"Hover dwell: {'ON' if enabled else 'OFF'}")

    def _apply_min_hover(self, ms):
        config.DWELL_MIN_MS = max(50, int(ms))
        self.status_bar.config(text=f"Minimum hover time set to {config.DWELL_MIN_MS} ms")
