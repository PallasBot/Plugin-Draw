"""AI 服务画图执行（bundled draw 主路径）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nonebot import logger

if TYPE_CHECKING:
    import httpx

    from .config import ImageGenSettings
    from .draw_attempts import DrawDeadline

from .ai_runtime_client import (
    gateway_payload_from_backends,
    generate_image_via_ai_service,
)
from .draw_stats_store import (
    classify_draw_gateway,
    record_draw_stats,
    resolve_draw_stats_cost,
)
from .draw_usage_store import bump_pallas_draw_usage
from .image_api import (
    message_at_user,
    reply_generated_image_bytes,
    request_timeout_for_deadline,
)
from .replies import DRAW_VAGUE_REPLY
from .runtime_state import (
    ai_runtime_circuit_is_open,
    ai_runtime_circuit_status,
    record_ai_runtime_failure,
    record_ai_runtime_success,
)


@dataclass(frozen=True, slots=True)
class AiDrawRunResult:
    handled: bool
    image_sent: bool = False
    fallback_plugin: bool = False


async def run_ai_service_draw(
    matcher,
    http_client: httpx.AsyncClient,
    *,
    cfg: ImageGenSettings,
    bot_id: int,
    group_id: int,
    user_id: int,
    usage_key: tuple[int, int],
    count_usage: bool,
    gen_prompt: str,
    ref_urls: list[str],
    deadline: DrawDeadline,
) -> AiDrawRunResult:
    if ai_runtime_circuit_is_open():
        circuit = ai_runtime_circuit_status()
        logger.warning(
            f"bot [{bot_id}] draw ai runtime circuit open in group [{group_id}] "
            f"failures={circuit.consecutive_failures} "
            f"reason={circuit.recent_failure_reason or 'unknown'}",
        )
        if cfg.ai_runtime_fallback_to_plugin:
            return AiDrawRunResult(handled=False, fallback_plugin=True)
        await matcher.finish(message_at_user(user_id, DRAW_VAGUE_REPLY))
        return AiDrawRunResult(handled=True, image_sent=False)

    ai_result = await generate_image_via_ai_service(
        http_client,
        bot_id=bot_id,
        group_id=group_id,
        user_id=user_id,
        prompt=gen_prompt,
        ref_urls=ref_urls,
        timeout_sec=request_timeout_for_deadline(deadline.remaining_seconds()),
        count_usage=count_usage,
        gateway=gateway_payload_from_backends(cfg.api_backends()),
    )
    if ai_result.pending_callback:
        return AiDrawRunResult(handled=True, image_sent=True)
    if ai_result.ok and ai_result.image_bytes is not None:
        record_ai_runtime_success()
        await reply_generated_image_bytes(
            matcher,
            ai_result.image_bytes,
            at_user_id=user_id,
            persist_draw=(usage_key[0], usage_key[1]),
        )
        bump_pallas_draw_usage(usage_key, count_usage)
        gw, provider_key = classify_draw_gateway(
            provider_id=ai_result.provider_id,
            name=ai_result.backend_id,
        )
        model = (cfg.model or "").strip()
        cost_amount, cost_currency = resolve_draw_stats_cost(
            images=1,
            provider_id=ai_result.provider_id,
            backend_name=ai_result.backend_id,
        )
        record_draw_stats(
            ok=True,
            gateway=gw,
            provider=provider_key,
            model=model,
            source="ai_runtime",
            cost_amount=cost_amount,
            cost_currency=cost_currency or None,
        )
        return AiDrawRunResult(handled=True, image_sent=True)

    record_ai_runtime_failure(ai_result.reply_text or "ai_runtime_failed")
    logger.warning(
        f"bot [{bot_id}] draw ai runtime failed in group [{group_id}] "
        f"provider={ai_result.provider_id or '-'} "
        f"backend={ai_result.backend_id or '-'} "
        f"reply={ai_result.reply_text[:120]!r}",
    )
    if not cfg.ai_runtime_fallback_to_plugin:
        gw, provider_key = classify_draw_gateway(
            provider_id=ai_result.provider_id,
            name=ai_result.backend_id,
        )
        record_draw_stats(
            ok=False,
            gateway=gw,
            provider=provider_key,
            model=(cfg.model or "").strip(),
            source="ai_runtime",
        )
        await matcher.finish(message_at_user(user_id, ai_result.reply_text or DRAW_VAGUE_REPLY))
        return AiDrawRunResult(handled=True, image_sent=False)
    logger.info(f"bot [{bot_id}] draw fallback to plugin runtime in group [{group_id}]")
    return AiDrawRunResult(handled=False, fallback_plugin=True)
