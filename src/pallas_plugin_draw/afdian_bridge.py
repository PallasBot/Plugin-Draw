"""可选接入社区插件 afdian：未安装/未启用时画画仅走免费日限。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class DrawQuotaDecision:
    """count_usage: 是否计入免费日限；paid_credit: 成功后是否扣额外额度。"""

    count_usage: bool
    paid_credit: bool
    block_message: str | None = None


def _looks_like_afdian_api(mod: ModuleType) -> bool:
    return all(hasattr(mod, name) for name in ("is_ready", "debit_one", "credit_balance"))


def _load_afdian_api() -> ModuleType | None:
    # pip / 短名、以及 local/plugins 路径加载后的 local.plugins.afdian
    for name in ("afdian.api", "local.plugins.afdian.api"):
        try:
            mod = import_module(name)
        except ImportError:
            continue
        if _looks_like_afdian_api(mod):
            return mod
    for key, mod in list(sys.modules.items()):
        if not key.endswith(".afdian.api") and key != "afdian.api":
            continue
        if isinstance(mod, ModuleType) and _looks_like_afdian_api(mod):
            return mod
    return None


def afdian_ready() -> bool:
    api = _load_afdian_api()
    if api is None:
        return False
    try:
        return bool(api.is_ready())
    except Exception:
        return False


def free_quota_exhausted_message(limit_n: int) -> str:
    if limit_n > 0:
        return f"今日次数已用完（{limit_n} 次），请明天再来"
    return "今日画画次数已用完，请明天再来"


def resolve_draw_quota(
    *,
    count_usage: bool,
    limit_n: int,
    usage_today: int,
    group_id: int,
    user_id: int,
) -> DrawQuotaDecision:
    """免费 → 可选额外额度；未接入爱发电时绝不进入付费语义。"""
    if not count_usage:
        return DrawQuotaDecision(count_usage=False, paid_credit=False)
    if limit_n > 0 and usage_today < limit_n:
        return DrawQuotaDecision(count_usage=True, paid_credit=False)

    api = _load_afdian_api()
    if api is None or not api.is_ready():
        return DrawQuotaDecision(
            count_usage=True,
            paid_credit=False,
            block_message=free_quota_exhausted_message(limit_n),
        )

    billing_user = int(api.resolve_billing_user_id(group_id, user_id))
    shared_pool = bool(api.group_billing_is_shared(group_id, user_id))
    ever_had = bool(api.ever_had_credits(billing_user))
    owner = api.group_billing_owner(group_id) if hasattr(api, "group_billing_owner") else None
    has_owner = owner is not None

    if int(api.credit_balance(billing_user)) < 1:
        msg = api.quota_exhausted_message(
            limit_n=limit_n,
            shared_pool=shared_pool or has_owner,
            ever_had=ever_had,
            has_owner=has_owner,
        )
        return DrawQuotaDecision(count_usage=True, paid_credit=False, block_message=str(msg))
    return DrawQuotaDecision(count_usage=True, paid_credit=True)


async def apply_usage_after_success(
    *,
    usage_key: tuple[int, int],
    user_id: int,
    count_usage: bool,
    paid_credit: bool,
    bump_free,
) -> None:
    """成功出图后扣次：免费与额外额度互斥。"""
    group_id = usage_key[0]
    if paid_credit:
        api = _load_afdian_api()
        if api is None or not api.is_ready():
            return
        billing_user = int(api.resolve_billing_user_id(group_id, user_id))
        await api.debit_one(billing_user)
        return
    bump_free(usage_key, count_usage)
