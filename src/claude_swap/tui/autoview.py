"""Live auto-switch screen: the real engine, visualized.

Runs :class:`AutoSwitchEngine` in a thread worker and renders its typed
events. Opens in **dry-run** — opening a view must never start switching
accounts on its own; going live is an explicit, confirmed action. The
engine's own state file semantics (shared cooldown, quarantine list, state
lock) make it safe to run alongside an external ``cswap auto``.

The active account's full card sits on top (same widget as the dashboard's
panel, with the threshold tick); this screen adds the engine badge, the
ranked switch candidates, and the decision log. While it is up, the app's
snapshot poller runs store-only: the engine is the only fetcher.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from claude_swap.autoswitch import (
    AutoSwitchEngine,
    AutoSwitchEvent,
    _seven_day_reset_ts,
    binding_pct,
    pct_label,
)
from claude_swap.models import AccountsSnapshot
from claude_swap.settings import SETTING_SPECS, load_settings, parse_model_names
from claude_swap.tui import data
from claude_swap.tui.modals import ConfirmModal
from claude_swap.tui.theme import Palette
from claude_swap.tui.widgets import AccountsPanel

if TYPE_CHECKING:
    from claude_swap.tui.app import CswapApp

_EVENT_ROLES = {
    "switch": "accent",
    "error": "sev_warn",
    "account-quarantined": "sev_warn",
    "all-exhausted": "sev_crit",
}
_QUIET_KINDS = {"poll", "no-switch", "sleep", "account-unquarantined"}


def event_text(event: AutoSwitchEvent, *, palette: Palette = Palette.DARK) -> Text:
    """Log line for one engine event, styled like the CLI's human renderer."""
    role = _EVENT_ROLES.get(event.kind)
    if role is not None:
        style = getattr(palette, role)
    else:
        style = palette.muted if event.kind in _QUIET_KINDS else palette.foreground
    text = Text()
    text.append(f"{data.clock_stamp()}  ", style=palette.muted)
    text.append(event.human(), style=style)
    return text


class AutoScreen(Screen):
    BINDINGS = [
        Binding("l", "toggle_live", "Go live / dry-run"),
        Binding("t", "adjust_threshold", "Threshold"),
        Binding("left", "threshold_step(-1)", "-1%"),
        Binding("right", "threshold_step(1)", "+1%"),
        Binding("enter", "adjust_done", "Done"),
        Binding("escape,q", "back", "Back"),
    ]

    app: "CswapApp"

    def __init__(self) -> None:
        super().__init__()
        self._engine: AutoSwitchEngine | None = None
        self._settings = None
        # Session-only threshold adjustment (t, then arrows). Never written
        # to settings.json — same memory-only precedent as the dry-run
        # toggle. ``_configured_threshold`` is the mount-time file value the
        # screen reverts to on exit; ``_entry_threshold`` is the value when
        # adjust mode was entered (wake/log only on a net change).
        self._adjusting = False
        self._configured_threshold: float | None = None
        self._entry_threshold: float | None = None

    def compose(self) -> ComposeResult:
        yield AccountsPanel(show_minis=False, id="auto-active-panel")
        with Vertical(id="auto-top"):
            with Horizontal(id="auto-title-row"):
                yield Static(" DRY-RUN ", id="mode-badge", classes="dry")
                yield Static("", id="auto-summary")
            yield Static("", id="candidates")
        yield RichLog(id="event-log", highlight=False, markup=False, wrap=True)
        yield Footer()

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self.app.set_store_only(True)
        self._settings = load_settings(self.app.switcher.backup_dir)
        # The bar tick everywhere reads app.threshold_pct, loaded once at app
        # startup — sync it to the fresh file value so bars and engine agree,
        # and remember that value: unmount restores it (only the session
        # adjustment reverts, not this correction).
        self._configured_threshold = self._settings.threshold
        self.app.threshold_pct = self._settings.threshold
        self._update_summary()
        self.watch(self.app, "snapshot", self._on_snapshot)
        self.watch(self.app, "theme", self._on_theme_change)
        self._start_engine(dry_run=True)

    def on_unmount(self) -> None:
        if self._engine is not None:
            self._engine.stop()
        # A session threshold must not outlive the engine it steered: unpin
        # the poll planner and put the bar tick back on the file value.
        self.app.switcher.clear_poll_policy_inputs()
        if self._configured_threshold is not None:
            self.app.threshold_pct = self._configured_threshold
        self.app.set_store_only(False)

    def _on_theme_change(self, _theme: str) -> None:
        self._update_summary()
        self._update_badge()
        snap = self.app.snapshot
        if snap is not None:
            self._on_snapshot(snap)

    def action_back(self) -> None:
        if self._adjusting:
            self._end_adjust()
            return
        self.app.pop_screen()

    # -- threshold adjust mode ------------------------------------------------

    @property
    def _chain(self) -> tuple[str, ...]:
        """The wave's chain, or empty. Tolerates a screen with no engine yet."""
        engine = getattr(self, "_engine", None)
        return tuple(getattr(engine, "sequence", None) or ())

    @property
    def _chain_position(self) -> int:
        engine = getattr(self, "_engine", None)
        return getattr(engine, "sequence_position", -1)

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action in ("adjust_threshold", "threshold_step", "adjust_done") and (
            self._chain
        ):
            # A wave leaves on the same thresholds, but the knob offers a
            # single number while the decision now reads one bar per window —
            # and the wave's own driver is the chain. Offering the slider here
            # would suggest it steers something it does not.
            return False
        if action in ("threshold_step", "adjust_done") and not self._adjusting:
            return False  # hidden and inert until adjust mode is armed
        return True

    def action_adjust_threshold(self) -> None:
        if self._adjusting:
            self._end_adjust()
            return
        self._adjusting = True
        self._entry_threshold = self._settings.threshold
        self._update_summary()
        self.refresh_bindings()

    def action_adjust_done(self) -> None:
        if self._adjusting:
            self._end_adjust()

    def action_threshold_step(self, delta: float) -> None:
        if not self._adjusting:
            return
        spec = SETTING_SPECS["autoswitch.threshold"]
        value = min(spec.hi, max(spec.lo, self._settings.threshold + delta))
        self._set_threshold(value)

    def _end_adjust(self) -> None:
        self._adjusting = False
        self._update_summary()
        self.refresh_bindings()
        if self._settings.threshold == self._entry_threshold:
            return  # no net change: nothing to announce, no tick to force
        if self._engine is not None:
            self._engine.wake()  # show a decision at the new value now
        self.query_one("#event-log", RichLog).write(
            Text(
                f"— threshold set to {pct_label(self._settings.threshold)}% "
                "for this session —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _set_threshold(self, value: float) -> None:
        if value == self._settings.threshold:
            return
        self._settings = replace(self._settings, threshold=value)
        if self._engine is not None:
            self._engine.apply_threshold(value)
        self.app.threshold_pct = value
        self.query_one("#auto-active-panel", AccountsPanel).refresh()
        self._update_summary()

    def _update_summary(self) -> None:
        palette = Palette.from_theme(self.app.current_theme)
        text = Text()
        text.append("auto-switch · ")
        chain = self._chain
        if chain:
            position = self._chain_position
            text.append(
                f"chain {' → '.join(chain)}"
                + (f" · step {position + 1}/{len(chain)}" if position >= 0 else "")
            )
        else:
            text.append(
                f"threshold {pct_label(self._settings.threshold)}%",
                style=palette.accent if self._adjusting else "",
            )
            if self._settings.threshold != self._configured_threshold:
                text.append(" (session)", style=palette.muted)
        text.append(f" · poll every {self._settings.interval_seconds:.0f}s")
        if self._adjusting:
            text.append("   ← → adjust · enter done", style=palette.muted)
        self.query_one("#auto-summary", Static).update(text)

    # -- engine -------------------------------------------------------------

    def _start_engine(self, *, dry_run: bool) -> None:
        engine = AutoSwitchEngine(
            self.app.switcher,
            self._settings,
            self._emit_from_thread,
            dry_run=dry_run,
        )
        self._engine = engine
        # The summary was drawn on mount, before there was an engine to
        # describe — redraw it now that one exists, or a wave would be
        # reported as a threshold loop for as long as the screen is open.
        self._update_summary()
        self.run_worker(
            engine.run_loop,
            thread=True,
            group="engine",
            exit_on_error=False,
            name=f"auto-engine-{'dry' if dry_run else 'live'}",
        )
        self._update_badge()
        log = self.query_one("#event-log", RichLog)
        mode = "DRY-RUN (watching only)" if dry_run else "LIVE (will switch accounts)"
        log.write(
            Text(
                f"— engine started: {mode} —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _emit_from_thread(self, event: AutoSwitchEvent) -> None:
        """Engine ``on_event`` callback — runs on the worker thread."""
        try:
            self.app.call_from_thread(self._on_engine_event, event)
        except Exception:
            # App/screen tearing down mid-tick; the event has nowhere to go.
            pass

    def _on_engine_event(self, event: AutoSwitchEvent) -> None:
        if not self.is_attached:
            return
        palette = Palette.from_theme(self.app.current_theme)
        self.query_one("#event-log", RichLog).write(event_text(event, palette=palette))
        if event.kind == "switch":
            self.app.request_refresh()

    def action_toggle_live(self) -> None:
        if self._engine is None:
            return
        if self._engine.dry_run:
            self.app.push_screen(
                ConfirmModal(
                    "Go live? claude-swap will switch your active account "
                    "automatically when the threshold is reached.\n\n"
                    "(Same behavior as running `cswap auto` in a terminal.)",
                    title="Go live",
                    yes_label="Go live",
                ),
                self._on_live_confirm,
            )
        else:
            self._restart_engine(dry_run=True)

    def _on_live_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self._restart_engine(dry_run=False)

    def _restart_engine(self, *, dry_run: bool) -> None:
        if self._engine is not None:
            self._engine.stop()
        self._start_engine(dry_run=dry_run)

    def _update_badge(self) -> None:
        badge = self.query_one("#mode-badge", Static)
        if self._engine is not None and not self._engine.dry_run:
            badge.update(" LIVE ")
            badge.set_classes("live")
        else:
            badge.update(" DRY-RUN ")
            badge.set_classes("dry")

    # -- candidates -----------------------------------------------------------

    def _on_snapshot(self, snap: AccountsSnapshot | None) -> None:
        if snap is None:
            return
        self.query_one("#candidates", Static).update(
            self._candidates_text(snap, active_number=snap.active_number)
        )

    def _sequence_text(
        self, snap: AccountsSnapshot, active_number: str | None
    ) -> Text:
        """The declared chain, in its own order, with the wave's position.

        A ranking would be a lie here: the user wrote the order, and the only
        question the panel can usefully answer is how far along it is. Spent
        elements are marked because the wave skips them rather than waiting,
        so seeing one greyed out explains a jump that would otherwise look
        like the order being ignored.
        """
        palette = Palette.from_theme(self.app.current_theme)
        models = parse_model_names(self._settings.model) if self._settings else ()
        chain = self._chain
        position = self._chain_position
        by_number = {acc.number: acc for acc in snap.accounts}

        text = Text()
        text.append("Chain", style=palette.muted)
        for index, number in enumerate(chain):
            acc = by_number.get(number)
            marker = "▸" if index == position else " "
            entry = Text()
            entry.append(f"\n {marker} {number:>2}  ", style=palette.foreground)
            if acc is None:
                entry.append("not in this install", style=palette.muted)
                text.append(entry)
                continue
            entry.append(acc.email, style=palette.foreground)
            pct = binding_pct(acc.usage.last_good, models)
            if acc.usage.sentinel is not None:
                entry.append(
                    "  " + data.sentinel_label(
                        acc.usage.sentinel, acc.usage.foreign_owner
                    ),
                    style=palette.muted,
                )
            elif pct is None:
                entry.append("  usage unknown", style=palette.muted)
            else:
                entry.append(f"  {pct:3.0f}% used", style=palette.severity(pct))
            if index < position:
                entry.append("  done", style=palette.muted)
            elif number == active_number:
                entry.append("  here", style=palette.muted)
            text.append(entry)
        return text

    def _candidates_text(
        self, snap: AccountsSnapshot, active_number: str | None
    ) -> Text:
        """What the engine will actually do next.

        This panel used to re-derive a headroom ranking and label it "Next
        best" no matter what the engine had been told to do — already wrong
        under ``consume-first``, which ranks by weekly reset, and wrong in a
        different way under a scripted wave, where there is no ranking at all
        and the order was declared by hand. Each mode now renders itself.
        """
        if self._chain:
            return self._sequence_text(snap, active_number)
        # Same window set as the engine (autoswitch.model included), so the
        # displayed ranking can never disagree with the account it picks.
        palette = Palette.from_theme(self.app.current_theme)
        models = parse_model_names(self._settings.model) if self._settings else ()
        consume_first = (
            self._settings is not None
            and self._settings.strategy == "consume-first"
        )
        ranked: list[tuple[float, str]] = []  # (sort key: pct used, number)
        lines: dict[str, Text] = {}
        for acc in snap.accounts:
            if acc.number == active_number or not acc.switchable:
                continue
            pct = binding_pct(acc.usage.last_good, models)
            entry = Text()
            entry.append(f"\n  {acc.number:>2}  ", style=palette.foreground)
            entry.append(acc.email, style=palette.foreground)
            if acc.usage.sentinel is not None:
                entry.append(
                    "  " + data.sentinel_label(
                        acc.usage.sentinel, acc.usage.foreign_owner
                    ),
                    style=palette.muted,
                )
                ranked.append((998.0, acc.number))
            elif pct is None:
                entry.append("  usage unknown", style=palette.muted)
                ranked.append((999.0, acc.number))
            else:
                entry.append(f"  {pct:3.0f}% used", style=palette.severity(pct))
                if consume_first:
                    # The strategy spends the most perishable quota first, so
                    # the soonest weekly reset leads and unknown resets sort
                    # last — the engine's own key, not a second opinion.
                    reset_ts = _seven_day_reset_ts(acc.usage.last_good, time.time())
                    ranked.append(
                        (reset_ts if reset_ts is not None else float("inf"),
                         acc.number)
                    )
                else:
                    ranked.append((pct, acc.number))
            lines[acc.number] = entry

        text = Text()
        text.append(
            "Next by weekly reset" if consume_first else "Next best",
            style=palette.muted,
        )
        if not ranked:
            text.append("\n  no other switchable accounts", style=palette.muted)
            return text
        for _pct, number in sorted(ranked):
            text.append(lines[number])
        return text
