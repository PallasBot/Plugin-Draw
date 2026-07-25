"""draw 媒体任务 AI callback 收尾。"""

from __future__ import annotations

from typing import Any

from pallas.api.platform import import_plugin_submodule


def on_draw_media_task_failed(task: dict[str, Any]) -> None:
    runtime_state = import_plugin_submodule("draw", "runtime_state")
    stats_store = import_plugin_submodule("draw", "draw_stats_store")
    runtime_state.record_ai_runtime_failure("draw_callback_failed")
    gateway, provider_key = stats_store.classify_draw_gateway(
        provider_id=task.get("provider_id"),
        name=task.get("backend_id") or task.get("model"),
    )
    stats_store.record_draw_stats(
        ok=False,
        gateway=gateway,
        provider=provider_key,
        model=str(task.get("model") or "").strip(),
        source="ai_runtime",
    )


def on_draw_media_task_success(task: dict[str, Any], image_bytes: bytes, group_id: int) -> None:
    runtime_state = import_plugin_submodule("draw", "runtime_state")
    usage_store = import_plugin_submodule("draw", "draw_usage_store")
    stats_store = import_plugin_submodule("draw", "draw_stats_store")
    image_api = import_plugin_submodule("draw", "image_api")

    runtime_state.record_ai_runtime_success()
    persist_user = task.get("user_id")
    if persist_user is not None:
        image_api.schedule_persist_generated_draw(image_bytes, int(group_id), int(persist_user))
    if task.get("count_usage"):
        usage_user = task.get("user_id")
        if usage_user is not None:
            usage_store.bump_pallas_draw_usage((int(group_id), int(usage_user)), True)
    gateway, provider_key = stats_store.classify_draw_gateway(
        provider_id=task.get("provider_id"),
        name=task.get("backend_id") or task.get("model"),
    )
    model = str(task.get("model") or "").strip()
    cost_amount, cost_currency = stats_store.resolve_draw_stats_cost(
        model=model,
        images=1,
        provider_id=task.get("provider_id"),
        backend_name=task.get("backend_id"),
    )
    stats_store.record_draw_stats(
        ok=True,
        gateway=gateway,
        provider=provider_key,
        model=model,
        source="ai_runtime",
        cost_amount=cost_amount,
        cost_currency=cost_currency or None,
    )
