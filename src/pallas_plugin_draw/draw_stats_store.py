"""画画运维向用量统计（次数 + 可选费用；与用户日限配额库分离）。"""

from __future__ import annotations

import json
import threading
import time
from typing import Any
from urllib.parse import urlparse

from pallas.api.paths import plugin_data_dir

_STORE_VER = 1
_STATS_FILE = "draw_stats_daily.json"

_lock = threading.Lock()
_day_key = ""
_hydrated = False
_ok_count = 0
_fail_count = 0
_image_count = 0
_cost_total = 0.0
_cost_currency = ""
_by_gateway: dict[str, dict[str, float | int]] = {}
_by_provider: dict[str, dict[str, float | int]] = {}
_by_model: dict[str, dict[str, float | int]] = {}

_EMPTY_ROW: dict[str, float | int] = {
    "ok_count": 0,
    "fail_count": 0,
    "image_count": 0,
    "cost_total": 0.0,
}


def today_key() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def stats_file_path():
    return plugin_data_dir("draw") / _STATS_FILE


def classify_draw_gateway(
    *,
    provider_id: str | None = None,
    name: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """返回 (gateway, provider_key)。有 provider_id → provider；否则 manual。"""
    pid = str(provider_id or "").strip()
    if pid:
        return "provider", pid
    label = str(name or "").strip()
    if label:
        return "manual", label
    host = ""
    raw = str(base_url or "").strip()
    if raw:
        try:
            host = (urlparse(raw if "://" in raw else f"https://{raw}").hostname or "").strip()
        except Exception:
            host = ""
    return "manual", host or "manual"


def classify_backend(backend: Any) -> tuple[str, str, str]:
    """从 ImageApiBackend 提取 gateway / provider_key / model。"""
    gateway, provider_key = classify_draw_gateway(
        provider_id=getattr(backend, "provider_id", None),
        name=getattr(backend, "name", None),
        base_url=getattr(backend, "base_url", None),
    )
    model = str(getattr(backend, "model", "") or "").strip()
    return gateway, provider_key, model


def _bump_row(
    row: dict[str, float | int],
    *,
    ok: bool,
    images: int,
    cost: float,
) -> None:
    if ok:
        row["ok_count"] = int(row.get("ok_count") or 0) + 1
        row["image_count"] = int(row.get("image_count") or 0) + max(0, images)
    else:
        row["fail_count"] = int(row.get("fail_count") or 0) + 1
    if cost:
        row["cost_total"] = float(row.get("cost_total") or 0) + cost


def _copy_breakdown(dst: dict[str, dict[str, float | int]], src: Any) -> None:
    if not isinstance(src, dict):
        return
    for key, metrics in src.items():
        if not isinstance(metrics, dict):
            continue
        name = str(key or "").strip()
        if not name:
            continue
        dst[name] = {
            "ok_count": int(metrics.get("ok_count") or 0),
            "fail_count": int(metrics.get("fail_count") or 0),
            "image_count": int(metrics.get("image_count") or 0),
            "cost_total": float(metrics.get("cost_total") or 0),
        }


def _apply_raw_locked(raw: dict[str, Any]) -> None:
    global _ok_count, _fail_count, _image_count, _cost_total, _cost_currency
    if _ok_count or _fail_count or _image_count or _cost_total or _by_gateway or _by_provider or _by_model:
        return
    _ok_count = int(raw.get("ok_count") or 0)
    _fail_count = int(raw.get("fail_count") or 0)
    _image_count = int(raw.get("image_count") or 0)
    _cost_total = float(raw.get("cost_total") or 0)
    _cost_currency = str(raw.get("cost_currency") or "").strip()
    _copy_breakdown(_by_gateway, raw.get("by_gateway"))
    _copy_breakdown(_by_provider, raw.get("by_provider"))
    _copy_breakdown(_by_model, raw.get("by_model"))


def _salvage_day_to_daily(day: str, raw: dict[str, Any]) -> None:
    try:
        from pallas.product.llm.llm_daily_stats_store import write_day_side

        write_day_side(
            day,
            "ai",
            {
                "day_key": day,
                "source": "bot",
                "images": {**raw, "day_key": day, "source": "draw_plugin"},
            },
        )
    except Exception:
        pass


def _persist_shared_allowed() -> bool:
    try:
        from pallas.core.platform.shard import context as shard_ctx

        if shard_ctx.sharding_active() and shard_ctx.is_worker():
            return False
    except Exception:
        pass
    return True


def _hydrate_from_disk_locked() -> None:
    global _hydrated  # noqa: PLW0603
    if _hydrated:
        return
    _hydrated = True
    today = str(_day_key or today_key()).strip()[:10]

    try:
        from pallas.product.llm.shard_metric_hydrate import load_worker_day_metric

        worker_raw = load_worker_day_metric(metric_key="llm_draw", day_key=today)
        if isinstance(worker_raw, dict):
            _apply_raw_locked(worker_raw)
            return
    except Exception:
        pass

    path = stats_file_path()
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict) or not raw.get("day_key"):
        return
    file_day = str(raw.get("day_key") or "").strip()[:10]
    if file_day and file_day != today:
        _salvage_day_to_daily(file_day, raw)
        return
    _apply_raw_locked(raw)


def _rollover_if_needed_locked() -> None:
    global _day_key, _hydrated, _ok_count, _fail_count, _image_count, _cost_total, _cost_currency
    today = today_key()
    if _day_key == today:
        return
    if _day_key:
        try:
            snap = _snapshot_locked(day_override=_day_key)
            if _persist_shared_allowed():
                _persist_locked(day_override=_day_key)
            _salvage_day_to_daily(_day_key, snap)
        except Exception:
            pass
        _ok_count = 0
        _fail_count = 0
        _image_count = 0
        _cost_total = 0.0
        _cost_currency = ""
        _by_gateway.clear()
        _by_provider.clear()
        _by_model.clear()
        _day_key = today
        _hydrated = True
        return
    _day_key = today
    _hydrated = False


def _snapshot_locked(*, day_override: str | None = None) -> dict[str, Any]:
    return {
        "source": "draw_plugin",
        "day_key": day_override or _day_key or today_key(),
        "updated_at": time.time(),
        "ok_count": _ok_count,
        "fail_count": _fail_count,
        "image_count": _image_count,
        "cost_total": _cost_total,
        "cost_currency": _cost_currency,
        "by_gateway": {k: dict(v) for k, v in _by_gateway.items()},
        "by_provider": {k: dict(v) for k, v in _by_provider.items()},
        "by_model": {k: dict(v) for k, v in _by_model.items()},
    }


def _persist_locked(*, day_override: str | None = None) -> None:
    snap = _snapshot_locked(day_override=day_override)
    path = stats_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v": _STORE_VER, **snap}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cost_from_response(response_body: Any) -> tuple[float, str]:
    """从上游响应提取明确费用字段；无则 (0, \"\")。不臆造 token×单价。"""
    payload: Any = response_body
    if isinstance(response_body, (bytes, bytearray)):
        try:
            response_body = response_body.decode("utf-8", errors="ignore")
        except Exception:
            return 0.0, ""
    if isinstance(response_body, str):
        text = response_body.strip()
        if not text:
            return 0.0, ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return 0.0, ""
    if not isinstance(payload, dict):
        return 0.0, ""

    def pick_amount(node: dict[str, Any]) -> float:
        for key in (
            "cost",
            "total_cost",
            "cost_amount",
            "cost_usd",
            "cost_cny",
            "price",
            "total_price",
        ):
            raw = node.get(key)
            if raw is None or isinstance(raw, bool):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0.0

    def pick_currency(node: dict[str, Any]) -> str:
        for key in ("currency", "cost_currency", "price_currency"):
            cur = str(node.get(key) or "").strip().upper()
            if cur:
                return cur
        if node.get("cost_usd") is not None:
            return "USD"
        if node.get("cost_cny") is not None:
            return "CNY"
        return ""

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    for node in (usage, payload):
        if not isinstance(node, dict):
            continue
        amount = pick_amount(node)
        if amount > 0:
            return amount, pick_currency(node)
    return 0.0, ""


def _unit_price_from_backend(backend: Any) -> float:
    if backend is None:
        return 0.0
    try:
        return max(0.0, float(getattr(backend, "cost_per_image", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _lookup_gateway_unit_price(
    *,
    provider_id: str | None = None,
    backend_name: str | None = None,
) -> float:
    """按 provider_id / 显示名从已配置网关解析单价；找不到则用主线单价。"""
    try:
        from .config import ImageGenSettings, get_draw_config

        cfg = get_draw_config()
        backends = ImageGenSettings(cfg).api_backends()
        pid = str(provider_id or "").strip()
        name = str(backend_name or "").strip()
        if pid:
            for row in backends:
                if (row.provider_id or "").strip() == pid:
                    return _unit_price_from_backend(row)
        if name:
            for row in backends:
                if (row.name or "").strip() == name or (row.label or "").strip() == name:
                    return _unit_price_from_backend(row)
        return max(0.0, float(getattr(cfg, "pallas_image_cost_per_image", 0) or 0))
    except Exception:
        return 0.0


def resolve_draw_stats_cost(
    *,
    model: str | None = None,
    images: int = 1,
    response_body: Any = None,
    unit_price: float | None = None,
    backend: Any = None,
    provider_id: str | None = None,
    backend_name: str | None = None,
) -> tuple[float, str]:
    """解析本次成功画画应记费用：(amount, currency)。失败调用方勿传正费用。"""
    del model  # 单价按网关，不再按模型全局表
    amount, currency = _cost_from_response(response_body)
    if amount > 0:
        return amount, currency
    unit = 0.0
    if unit_price is not None:
        try:
            unit = max(0.0, float(unit_price))
        except (TypeError, ValueError):
            unit = 0.0
    if unit <= 0:
        unit = _unit_price_from_backend(backend)
    if unit <= 0:
        unit = _lookup_gateway_unit_price(
            provider_id=provider_id or getattr(backend, "provider_id", None),
            backend_name=backend_name or getattr(backend, "name", None) or getattr(backend, "label", None),
        )
    if unit <= 0:
        return 0.0, ""
    try:
        from .config import get_draw_config

        cur = str(getattr(get_draw_config(), "pallas_image_stats_cost_currency", "") or "").strip().upper()
    except Exception:
        cur = ""
    imgs = max(1, int(images or 1))
    return unit * imgs, cur


def record_draw_stats(
    *,
    ok: bool,
    gateway: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    images: int = 1,
    cost_amount: float = 0.0,
    cost_currency: str | None = None,
    source: str = "plugin",
) -> None:
    """记一笔画画结果。失败时 images 计 0。"""
    del source  # 预留字段，快照统一 source=draw_plugin
    try:
        with _lock:
            _rollover_if_needed_locked()
            _hydrate_from_disk_locked()
            global _ok_count, _fail_count, _image_count, _cost_total, _cost_currency
            cost = max(0.0, float(cost_amount or 0))
            imgs = max(0, int(images)) if ok else 0
            if ok:
                _ok_count += 1
                _image_count += imgs
            else:
                _fail_count += 1
            if cost:
                _cost_total += cost
                cur = str(cost_currency or "").strip().upper()
                if cur and not _cost_currency:
                    _cost_currency = cur
            gw = str(gateway or "").strip().lower() or "manual"
            prow = _by_gateway.setdefault(gw, dict(_EMPTY_ROW))
            _bump_row(prow, ok=ok, images=imgs, cost=cost)
            provider_key = str(provider or "").strip() or gw
            brow = _by_provider.setdefault(provider_key, dict(_EMPTY_ROW))
            _bump_row(brow, ok=ok, images=imgs, cost=cost)
            model_key = str(model or "").strip()
            if model_key:
                mrow = _by_model.setdefault(model_key, dict(_EMPTY_ROW))
                _bump_row(mrow, ok=ok, images=imgs, cost=cost)
            if _persist_shared_allowed():
                _persist_locked()
    except Exception:
        pass


def record_draw_stats_for_backend(
    backend: Any,
    *,
    ok: bool,
    images: int = 1,
    cost_amount: float = 0.0,
    cost_currency: str | None = None,
    source: str = "plugin",
) -> None:
    gateway, provider_key, model = classify_backend(backend)
    record_draw_stats(
        ok=ok,
        gateway=gateway,
        provider=provider_key,
        model=model,
        images=images,
        cost_amount=cost_amount,
        cost_currency=cost_currency,
        source=source,
    )


def _merge_breakdown_sum(
    dst: dict[str, dict[str, float | int]],
    src: Any,
) -> None:
    if not isinstance(src, dict):
        return
    for key, metrics in src.items():
        if not isinstance(metrics, dict):
            continue
        name = str(key or "").strip()
        if not name:
            continue
        row = dst.setdefault(name, dict(_EMPTY_ROW))
        row["ok_count"] = int(row.get("ok_count") or 0) + int(metrics.get("ok_count") or 0)
        row["fail_count"] = int(row.get("fail_count") or 0) + int(metrics.get("fail_count") or 0)
        row["image_count"] = int(row.get("image_count") or 0) + int(metrics.get("image_count") or 0)
        row["cost_total"] = float(row.get("cost_total") or 0) + float(metrics.get("cost_total") or 0)


def merge_draw_snapshots(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_count = 0
    fail_count = 0
    image_count = 0
    cost_total = 0.0
    cost_currency = ""
    by_gateway: dict[str, dict[str, float | int]] = {}
    by_provider: dict[str, dict[str, float | int]] = {}
    by_model: dict[str, dict[str, float | int]] = {}
    day_key = ""
    updated_at = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        day_key = str(row.get("day_key") or day_key)
        try:
            updated_at = max(updated_at, float(row.get("updated_at") or 0))
        except (TypeError, ValueError):
            pass
        ok_count += int(row.get("ok_count") or 0)
        fail_count += int(row.get("fail_count") or 0)
        image_count += int(row.get("image_count") or 0)
        cost_total += float(row.get("cost_total") or 0)
        if not cost_currency:
            cost_currency = str(row.get("cost_currency") or "").strip()
        _merge_breakdown_sum(by_gateway, row.get("by_gateway"))
        _merge_breakdown_sum(by_provider, row.get("by_provider"))
        _merge_breakdown_sum(by_model, row.get("by_model"))
    return {
        "source": "draw_cluster",
        "day_key": day_key or today_key(),
        "updated_at": updated_at or time.time(),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "image_count": image_count,
        "cost_total": cost_total,
        "cost_currency": cost_currency,
        "by_gateway": by_gateway,
        "by_provider": by_provider,
        "by_model": by_model,
    }


def draw_stats_snapshot(*, include_persisted: bool = True) -> dict[str, Any]:
    with _lock:
        _rollover_if_needed_locked()
        if include_persisted:
            _hydrate_from_disk_locked()
        return _snapshot_locked()


def cluster_draw_stats_snapshot(*, max_stale_sec: float = 300.0) -> dict[str, Any]:
    """分片 hub：合并本进程与各 worker stats 中的 llm_draw 快照。"""
    rows = [draw_stats_snapshot(include_persisted=True)]
    try:
        from pallas.core.platform.shard import context as shard_ctx

        if shard_ctx.sharding_active() and shard_ctx.is_hub():
            from pallas.core.platform.shard.console_stats import iter_worker_shard_ids, read_worker_stats_file

            for shard_id in iter_worker_shard_ids(max_stale_sec=max_stale_sec):
                blob = read_worker_stats_file(shard_id)
                draw = blob.get("llm_draw")
                if not isinstance(draw, dict):
                    continue
                if (
                    int(draw.get("ok_count") or 0) <= 0
                    and int(draw.get("fail_count") or 0) <= 0
                    and not draw.get("by_model")
                    and not draw.get("by_provider")
                ):
                    continue
                rows.append(draw)
    except Exception:
        pass
    if len(rows) <= 1:
        out = rows[0]
        if isinstance(out, dict):
            return {**out, "source": out.get("source") or "draw_plugin"}
        return out
    return merge_draw_snapshots(rows)


def flush_draw_stats_sync() -> None:
    try:
        from pallas.core.platform.shard import context as shard_ctx

        if shard_ctx.sharding_active() and shard_ctx.is_worker():
            return
    except Exception:
        pass
    with _lock:
        _rollover_if_needed_locked()
        _hydrate_from_disk_locked()
        try:
            _persist_locked()
        except Exception:
            pass
    try:
        from pallas.product.llm.llm_daily_stats_store import write_day_side

        snap = draw_stats_snapshot(include_persisted=True)
        if int(snap.get("ok_count") or 0) <= 0 and int(snap.get("fail_count") or 0) <= 0:
            return
        write_day_side(
            str(snap.get("day_key") or today_key()),
            "ai",
            {"images": snap, "day_key": snap.get("day_key"), "source": "bot"},
        )
    except Exception:
        pass


def reset_draw_stats_for_tests() -> None:
    """仅测试用。"""
    global _day_key, _hydrated, _ok_count, _fail_count, _image_count, _cost_total, _cost_currency
    with _lock:
        _day_key = today_key()
        _hydrated = True
        _ok_count = 0
        _fail_count = 0
        _image_count = 0
        _cost_total = 0.0
        _cost_currency = ""
        _by_gateway.clear()
        _by_provider.clear()
        _by_model.clear()
