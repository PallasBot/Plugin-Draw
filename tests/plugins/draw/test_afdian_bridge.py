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


def test_zero_free_limit_requires_afdian_or_blocks(monkeypatch) -> None:
    """limit=0：无免费次数；未接入爱发电则拒画。"""
    monkeypatch.setattr(
        "pallas_plugin_draw.afdian_bridge._load_afdian_api", lambda: None
    )
    d = resolve_draw_quota(
        count_usage=True,
        limit_n=0,
        usage_today=0,
        group_id=1,
        user_id=2,
    )
    assert d.block_message is not None
    assert d.paid_credit is False


def test_zero_free_limit_with_credits_uses_paid(monkeypatch) -> None:
    class _Api:
        def is_ready(self) -> bool:
            return True

        def resolve_billing_user_id(self, group_id: int, user_id: int) -> int:
            return user_id

        def group_billing_is_shared(self, group_id: int, user_id: int) -> bool:
            return False

        def ever_had_credits(self, billing_user: int) -> bool:
            return True

        def group_billing_owner(self, group_id: int):
            return None

        def credit_balance(self, billing_user: int) -> int:
            return 2

    monkeypatch.setattr(
        "pallas_plugin_draw.afdian_bridge._load_afdian_api", lambda: _Api()
    )
    d = resolve_draw_quota(
        count_usage=True,
        limit_n=0,
        usage_today=0,
        group_id=1,
        user_id=2,
    )
    assert d == DrawQuotaDecision(count_usage=True, paid_credit=True)
