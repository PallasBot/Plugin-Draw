from __future__ import annotations

from pallas_plugin_draw.draw_stats_store import (
    classify_draw_gateway,
    draw_stats_snapshot,
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
