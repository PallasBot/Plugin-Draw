from __future__ import annotations

import json

from pallas_plugin_draw.draw_stats_store import (
    classify_draw_gateway,
    draw_stats_snapshot,
    merge_draw_snapshots,
    record_draw_stats,
    reset_draw_stats_for_tests,
    resolve_draw_stats_cost,
)


def test_classify_draw_gateway_provider_vs_manual() -> None:
    assert classify_draw_gateway(provider_id="openai-main") == ("provider", "openai-main")
    assert classify_draw_gateway(name="主站", base_url="https://img.example/v1") == (
        "manual",
        "主站",
    )
    assert classify_draw_gateway(base_url="https://img.example/v1") == ("manual", "img.example")


def test_record_draw_stats_ok_and_fail(tmp_path, monkeypatch) -> None:
    reset_draw_stats_for_tests()
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.stats_file_path",
        lambda: tmp_path / "draw_stats_daily.json",
    )
    record_draw_stats(
        ok=True,
        gateway="provider",
        provider="p1",
        model="gpt-image-1",
        images=1,
        cost_amount=0.12,
        cost_currency="USD",
    )
    record_draw_stats(ok=False, gateway="manual", provider="host.example", model="m2")
    snap = draw_stats_snapshot()
    assert snap["ok_count"] == 1
    assert snap["fail_count"] == 1
    assert snap["image_count"] == 1
    assert snap["cost_total"] == 0.12
    assert snap["cost_currency"] == "USD"
    assert snap["by_gateway"]["provider"]["ok_count"] == 1
    assert snap["by_gateway"]["manual"]["fail_count"] == 1
    assert snap["by_model"]["gpt-image-1"]["ok_count"] == 1


def test_resolve_draw_stats_cost_from_gateway_unit_price(monkeypatch) -> None:
    class _Backend:
        cost_per_image = 0.04
        provider_id = "p1"
        name = "主站"

    class _Cfg:
        pallas_image_stats_cost_currency = "CNY"
        pallas_image_cost_per_image = 0.01

    monkeypatch.setattr(
        "pallas_plugin_draw.config.get_draw_config",
        lambda: _Cfg(),
    )
    amount, currency = resolve_draw_stats_cost(backend=_Backend(), images=2)
    assert amount == 0.08
    assert currency == "CNY"


def test_resolve_draw_stats_cost_prefers_response_usage() -> None:
    amount, currency = resolve_draw_stats_cost(
        images=1,
        response_body='{"usage":{"cost":0.25,"currency":"USD"},"data":[]}',
    )
    assert amount == 0.25
    assert currency == "USD"


def test_merge_draw_snapshots_sums_totals_and_buckets() -> None:
    merged = merge_draw_snapshots([
        {
            "day_key": "2026-07-27",
            "ok_count": 2,
            "fail_count": 1,
            "image_count": 2,
            "cost_total": 0.1,
            "by_gateway": {"provider": {"ok_count": 2, "fail_count": 0, "image_count": 2, "cost_total": 0.1}},
            "by_provider": {"p1": {"ok_count": 2, "fail_count": 0, "image_count": 2, "cost_total": 0.1}},
            "by_model": {"m1": {"ok_count": 2, "fail_count": 0, "image_count": 2, "cost_total": 0.1}},
        },
        {
            "day_key": "2026-07-27",
            "ok_count": 1,
            "fail_count": 0,
            "image_count": 3,
            "cost_total": 0.2,
            "by_gateway": {"provider": {"ok_count": 1, "fail_count": 0, "image_count": 3, "cost_total": 0.2}},
            "by_provider": {"p1": {"ok_count": 1, "fail_count": 0, "image_count": 3, "cost_total": 0.2}},
            "by_model": {"m1": {"ok_count": 1, "fail_count": 0, "image_count": 3, "cost_total": 0.2}},
        },
    ])
    assert merged["source"] == "draw_cluster"
    assert merged["ok_count"] == 3
    assert merged["fail_count"] == 1
    assert merged["image_count"] == 5
    assert abs(merged["cost_total"] - 0.3) < 1e-9
    assert merged["by_gateway"]["provider"]["ok_count"] == 3
    assert merged["by_model"]["m1"]["image_count"] == 5


def test_merge_draw_snapshots_dedupes_identical_rows() -> None:
    row = {
        "day_key": "2026-07-27",
        "ok_count": 48,
        "fail_count": 5,
        "image_count": 48,
        "cost_total": 0.98,
        "by_gateway": {"provider": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
        "by_provider": {"AK": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
        "by_model": {"gpt-image-2": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
    }
    merged = merge_draw_snapshots([row, dict(row), dict(row)])
    assert merged["ok_count"] == 48
    assert merged["fail_count"] == 5
    assert merged["image_count"] == 48
    assert abs(merged["cost_total"] - 0.98) < 1e-9


def test_worker_skips_shared_file_hydrate(tmp_path, monkeypatch) -> None:
    reset_draw_stats_for_tests()
    path = tmp_path / "draw_stats_daily.json"
    path.write_text(
        json.dumps({
            "v": 1,
            "day_key": "2026-07-27",
            "ok_count": 48,
            "fail_count": 5,
            "image_count": 48,
            "cost_total": 0.98,
            "by_gateway": {},
            "by_provider": {"AK": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
            "by_model": {},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.stats_file_path",
        lambda: path,
    )
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.today_key",
        lambda: "2026-07-27",
    )
    monkeypatch.setattr("pallas.core.platform.shard.context.sharding_active", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.is_worker", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.shard_id", lambda: 1)
    monkeypatch.setattr(
        "pallas.core.platform.shard.console_stats.read_worker_stats_file",
        lambda shard_id: {},
    )
    import pallas_plugin_draw.draw_stats_store as ds

    with ds._lock:
        ds._day_key = "2026-07-27"
        ds._hydrated = False
        ds._ok_count = 0
        ds._fail_count = 0
        ds._image_count = 0
        ds._cost_total = 0.0
        ds._by_gateway.clear()
        ds._by_provider.clear()
        ds._by_model.clear()

    snap = draw_stats_snapshot(include_persisted=True)
    assert snap["ok_count"] == 0
    assert snap["fail_count"] == 0


def test_worker_scrubs_memory_matching_shared_file(tmp_path, monkeypatch) -> None:
    reset_draw_stats_for_tests()
    path = tmp_path / "draw_stats_daily.json"
    payload = {
        "v": 1,
        "day_key": "2026-07-27",
        "ok_count": 48,
        "fail_count": 5,
        "image_count": 48,
        "cost_total": 0.98,
        "cost_currency": "CNY",
        "by_gateway": {"provider": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
        "by_provider": {"AK": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
        "by_model": {"m": {"ok_count": 48, "fail_count": 0, "image_count": 48, "cost_total": 0.98}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.stats_file_path",
        lambda: path,
    )
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.today_key",
        lambda: "2026-07-27",
    )
    monkeypatch.setattr("pallas.core.platform.shard.context.sharding_active", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.is_worker", lambda: True)
    import pallas_plugin_draw.draw_stats_store as ds

    with ds._lock:
        ds._day_key = "2026-07-27"
        ds._hydrated = True
        ds._ok_count = 48
        ds._fail_count = 5
        ds._image_count = 48
        ds._cost_total = 0.98
        ds._cost_currency = "CNY"
        ds._by_gateway.clear()
        ds._by_provider.clear()
        ds._by_model.clear()
        ds._copy_breakdown(ds._by_gateway, payload["by_gateway"])
        ds._copy_breakdown(ds._by_provider, payload["by_provider"])
        ds._copy_breakdown(ds._by_model, payload["by_model"])

    snap = draw_stats_snapshot(include_persisted=False)
    assert snap["ok_count"] == 0
    assert snap["fail_count"] == 0
    assert snap["image_count"] == 0


def test_worker_skips_shared_file_persist(tmp_path, monkeypatch) -> None:
    reset_draw_stats_for_tests()
    path = tmp_path / "draw_stats_daily.json"
    monkeypatch.setattr(
        "pallas_plugin_draw.draw_stats_store.stats_file_path",
        lambda: path,
    )
    monkeypatch.setattr("pallas.core.platform.shard.context.sharding_active", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.is_worker", lambda: True)
    record_draw_stats(ok=True, gateway="provider", provider="p1", model="m1", images=1)
    assert not path.is_file()
    snap = draw_stats_snapshot(include_persisted=False)
    assert snap["ok_count"] == 1


def test_stale_file_salvaged_and_worker_rehydrates(tmp_path, monkeypatch) -> None:
    import pallas_plugin_draw.draw_stats_store as ds

    reset_draw_stats_for_tests()
    path = tmp_path / "draw_stats_daily.json"
    monkeypatch.setattr(ds, "stats_file_path", lambda: path)
    monkeypatch.setattr(ds, "today_key", lambda: "2026-07-27")
    written: list[tuple] = []

    def fake_write(day: str, side: str, snapshot: dict) -> None:
        written.append((day, side, snapshot))

    monkeypatch.setattr("pallas.product.llm.llm_daily_stats_store.write_day_side", fake_write)

    path.write_text(
        json.dumps({
            "v": 1,
            "day_key": "2026-07-25",
            "ok_count": 9,
            "fail_count": 1,
            "image_count": 9,
            "cost_total": 1.0,
            "by_gateway": {},
            "by_provider": {"old": {"ok_count": 9, "fail_count": 1, "image_count": 9, "cost_total": 1.0}},
            "by_model": {},
        }),
        encoding="utf-8",
    )

    with ds._lock:
        ds._day_key = "2026-07-27"
        ds._hydrated = False
        ds._ok_count = 0
        ds._fail_count = 0
        ds._image_count = 0
        ds._cost_total = 0.0
        ds._by_gateway.clear()
        ds._by_provider.clear()
        ds._by_model.clear()

    snap = draw_stats_snapshot(include_persisted=True)
    assert snap["ok_count"] == 0
    assert written
    assert written[0][0] == "2026-07-25"
    assert written[0][2]["images"]["ok_count"] == 9

    reset_draw_stats_for_tests()
    monkeypatch.setattr("pallas.core.platform.shard.context.sharding_active", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.is_worker", lambda: True)
    monkeypatch.setattr("pallas.core.platform.shard.context.is_hub", lambda: False)
    monkeypatch.setattr("pallas.core.platform.shard.context.shard_id", lambda: 2)
    monkeypatch.setattr(
        "pallas.core.platform.shard.console_stats.read_worker_stats_file",
        lambda shard_id: {
            "llm_draw": {
                "day_key": "2026-07-27",
                "ok_count": 4,
                "fail_count": 0,
                "image_count": 4,
                "cost_total": 0.4,
                "by_gateway": {"provider": {"ok_count": 4, "fail_count": 0, "image_count": 4, "cost_total": 0.4}},
                "by_provider": {"p1": {"ok_count": 4, "fail_count": 0, "image_count": 4, "cost_total": 0.4}},
                "by_model": {"m1": {"ok_count": 4, "fail_count": 0, "image_count": 4, "cost_total": 0.4}},
            }
        },
    )
    with ds._lock:
        ds._day_key = "2026-07-27"
        ds._hydrated = False
        ds._ok_count = 0
        ds._fail_count = 0
        ds._image_count = 0
        ds._cost_total = 0.0
        ds._by_gateway.clear()
        ds._by_provider.clear()
        ds._by_model.clear()

    restored = draw_stats_snapshot(include_persisted=True)
    assert restored["ok_count"] == 4
    assert restored["image_count"] == 4
    assert restored["by_model"]["m1"]["ok_count"] == 4
