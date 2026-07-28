from __future__ import annotations

from datetime import date

import pallas_plugin_draw.draw_usage_store as mod


class _FakeTimer:
    def __init__(self, interval, fn):
        self.interval = interval
        self.fn = fn
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def test_usage_map_ignores_obsolete_total_entries(monkeypatch):
    warnings: list[str] = []

    def _warn(msg, *args, **kwargs):
        text = str(msg)
        if args:
            try:
                text = text.format(*args)
            except Exception:
                text = f"{text} {args}"
        warnings.append(text)

    monkeypatch.setattr(mod.logger, "warning", _warn)
    raw = {
        "version": 2,
        "entries": {},
        "total_entries": {
            "3997431382": {"day": "2026-07-24", "count": 1},
        },
    }
    out = mod._usage_map_from_payload(raw)
    assert out == {}
    assert any("total_entries" in w for w in warnings)


def test_bump_usage_does_not_persist_immediately(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(mod, "_pallas_draw_usage", {})
    monkeypatch.setattr(mod, "_flush_timer", None)
    monkeypatch.setattr(mod, "_usage_dirty", False)
    monkeypatch.setattr(mod, "_loaded", True)
    monkeypatch.setattr(mod, "_persist", lambda: calls.append("persist"))
    monkeypatch.setattr(mod.threading, "Timer", _FakeTimer)

    mod.bump_pallas_draw_usage((1, 2), True)

    assert calls == []
    assert mod._usage_dirty is True
    assert isinstance(mod._flush_timer, _FakeTimer)
    assert mod._flush_timer.started is True


def test_flush_pending_usage_persists_once(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(mod, "_pallas_draw_usage", {(1, 2): (date.today(), 1)})
    monkeypatch.setattr(mod, "_flush_timer", None)
    monkeypatch.setattr(mod, "_usage_dirty", True)
    monkeypatch.setattr(mod, "_loaded", True)
    monkeypatch.setattr(mod, "_persist", lambda: calls.append("persist"))

    mod.flush_pending_draw_usage_sync()

    assert calls == ["persist"]
    assert mod._usage_dirty is False
