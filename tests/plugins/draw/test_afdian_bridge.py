"""画画可选爱发电软接入。"""

from __future__ import annotations

from pallas_plugin_draw.afdian_bridge import (
    DrawQuotaDecision,
    free_quota_exhausted_message,
    resolve_draw_quota,
)


def test_free_exhausted_without_afdian_is_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        "pallas_plugin_draw.afdian_bridge._load_afdian_api", lambda: None
    )
    d = resolve_draw_quota(
        count_usage=True,
        limit_n=3,
        usage_today=3,
        group_id=1,
        user_id=2,
    )
    assert d == DrawQuotaDecision(
        count_usage=True,
        paid_credit=False,
        block_message=free_quota_exhausted_message(3),
    )
    assert "爱发电" not in (d.block_message or "")
    assert "牛牛爱发电" not in (d.block_message or "")
    assert "赞助" not in (d.block_message or "")


def test_within_free_quota_no_block() -> None:
    d = resolve_draw_quota(
        count_usage=True,
        limit_n=3,
        usage_today=1,
        group_id=1,
        user_id=2,
    )
    assert d.block_message is None
    assert d.paid_credit is False


def test_unlimited_user_skips_quota() -> None:
    d = resolve_draw_quota(
        count_usage=False,
        limit_n=3,
        usage_today=99,
        group_id=1,
        user_id=2,
    )
    assert d == DrawQuotaDecision(count_usage=False, paid_credit=False)
