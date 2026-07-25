"""牛牛画画上游请求重试与超时控制。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from nonebot import logger
from nonebot.exception import FinishedException
from pallas.api.messages import (
    http_body_rejects_response_format,
    http_status_should_skip_backend,
    http_status_should_try_next_param,
    upstream_error_should_skip_backend,
    upstream_error_visible_to_user,
    user_failure_reply,
)

from .afdian_bridge import apply_usage_after_success
from .config import ImageApiBackend
from .draw_stats_store import (
    record_draw_stats,
    record_draw_stats_for_backend,
    resolve_draw_stats_cost,
)
from .draw_usage_store import bump_pallas_draw_usage
from .image_api import (
    CffiRequestsError,
    image_api_body_issue_label,
    optional_message_at_user,
    reply_from_image_api_json,
    request_timeout_for_backend_attempt,
)
from .image_request_options import ImageGenRequestOptions, capped_param_attempts
from .replies import DRAW_VAGUE_REPLY
from .runtime_state import image_gen_semaphore

_AT_USER_UNSET: Any = object()


class DrawDeadline:
    def __init__(self, total_seconds: float) -> None:
        self.end = time.monotonic() + max(1.0, total_seconds)

    def expired(self) -> bool:
        return time.monotonic() >= self.end

    def remaining_seconds(self) -> float:
        return max(0.0, self.end - time.monotonic())


class DrawTotalTimeoutError(Exception):
    pass


def log_image_backend_unusable(
    bot_id: int,
    op: str,
    backend_label: str,
    group_id: int,
    body_text: str,
    *,
    has_more: bool,
) -> None:
    issue = image_api_body_issue_label(body_text) or "image_send_failed"
    snippet = body_text[:200]
    if has_more:
        logger.info(
            f"bot [{bot_id}] draw {op} backend={backend_label} unusable "
            f"in group [{group_id}] issue={issue} body={snippet!r}, trying next",
        )
    else:
        logger.warning(
            f"bot [{bot_id}] draw {op} backend={backend_label} unusable "
            f"in group [{group_id}] issue={issue} body={snippet!r}",
        )


def http_status_edits_unsupported(status: int) -> bool:
    return status in (404, 405, 501)


def format_transport_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return f"{type(exc).__name__}: {text}"
    return f"{type(exc).__name__}: {exc!r}"


def resolve_reply_at_user_id(user_id: int, at_user_id: Any) -> int | None:
    """未显式传入时默认 @ user_id；显式 None 表示不 @（如私聊）。"""
    if at_user_id is _AT_USER_UNSET:
        return user_id
    return at_user_id


PostRequestFn = Callable[
    [ImageApiBackend, ImageGenRequestOptions, float],
    Awaitable[tuple[int, str]],
]


async def run_backend_param_attempts(
    matcher,
    http_client: httpx.AsyncClient,
    bot_id: int,
    group_id: int,
    user_id: int,
    usage_key: tuple[int, int],
    count_usage: bool,
    deadline: DrawDeadline,
    op: str,
    backends: list[ImageApiBackend],
    *,
    with_ref_urls: bool,
    post_request: PostRequestFn,
    last_body_holder: list[str],
    last_status_holder: list[int],
    edits_abort_holder: list[bool] | None = None,
    paid_afdian: bool = False,
    at_user_id: int | None = _AT_USER_UNSET,
) -> bool:
    """按 backend × 参数组合请求；成功发图返回 True。"""
    reply_at = resolve_reply_at_user_id(user_id, at_user_id)
    for idx, backend in enumerate(backends):
        if deadline.expired():
            raise DrawTotalTimeoutError
        has_more_backend = idx < len(backends) - 1
        skip_backend = False
        param_attempts = capped_param_attempts(
            with_ref_urls=with_ref_urls,
            omit_response_format=backend.omit_response_format,
        )
        for opt_idx, req_opts in enumerate(param_attempts):
            if deadline.expired():
                raise DrawTotalTimeoutError
            has_more_opts = opt_idx < len(param_attempts) - 1
            still_retrying = has_more_backend or has_more_opts
            if opt_idx > 0:
                logger.info(
                    f"bot [{bot_id}] draw {op} retry params "
                    f"({req_opts.log_label()}) backend={backend.label} group=[{group_id}]",
                )
            req_timeout_cap = request_timeout_for_backend_attempt(
                deadline.remaining_seconds(),
                backends_remaining=len(backends) - idx,
                param_attempts_remaining=len(param_attempts) - opt_idx,
                primary_backend=(idx == 0),
            )
            logger.info(
                f"bot [{bot_id}] draw {op} request in group [{group_id}] "
                f"backend={backend.label} params=({req_opts.log_label()}) "
                f"timeout_cap={req_timeout_cap:.0f}s",
            )
            req_started = time.perf_counter()
            try:
                async with image_gen_semaphore:
                    status, body_text = await post_request(backend, req_opts, req_timeout_cap)
            except DrawTotalTimeoutError:
                raise
            except (
                FinishedException,
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.HTTPError,
                CffiRequestsError,
                RuntimeError,
            ) as e:
                err_text = format_transport_error(e)
                # 传输层失败（超时/断连等）优先切备线，避免主网关参数重试耗尽总超时。
                if has_more_backend:
                    logger.info(
                        f"bot [{bot_id}] draw {op} transport error "
                        f"backend={backend.label} group=[{group_id}]: {err_text}, trying next backend",
                    )
                    break
                if has_more_opts:
                    logger.info(
                        f"bot [{bot_id}] draw {op} transport error "
                        f"backend={backend.label} group=[{group_id}]: {err_text}, trying next params",
                    )
                    continue
                logger.warning(
                    f"bot [{bot_id}] draw {op} transport error backend={backend.label} group=[{group_id}]: {err_text}",
                )
                break
            last_body_holder[:] = [body_text]
            last_status_holder[:] = [status]
            logger.info(
                f"bot [{bot_id}] draw {op} response in group [{group_id}]: "
                f"backend={backend.label} status={status} "
                f"elapsed_ms={(time.perf_counter() - req_started) * 1000:.0f} "
                f"body_len={len(body_text)}",
            )
            if status == 200:
                if await reply_from_image_api_json(
                    matcher,
                    http_client,
                    body_text,
                    at_user_id=reply_at,
                    persist_draw=(usage_key[0], usage_key[1]),
                    finish_on_error=not still_retrying,
                ):
                    await apply_usage_after_success(
                        usage_key=usage_key,
                        user_id=user_id,
                        count_usage=count_usage,
                        paid_credit=paid_afdian,
                        bump_free=bump_pallas_draw_usage,
                    )
                    cost_amount, cost_currency = resolve_draw_stats_cost(
                        backend=backend,
                        images=1,
                        response_body=body_text,
                    )
                    record_draw_stats_for_backend(
                        backend,
                        ok=True,
                        source="plugin",
                        cost_amount=cost_amount,
                        cost_currency=cost_currency or None,
                    )
                    return True
                issue = image_api_body_issue_label(body_text) or "image_send_failed"
                log_image_backend_unusable(
                    bot_id,
                    op,
                    backend.label,
                    group_id,
                    body_text,
                    has_more=still_retrying,
                )
                if issue == "upstream_error":
                    if upstream_error_visible_to_user(body_text):
                        await matcher.finish(
                            optional_message_at_user(
                                reply_at,
                                user_failure_reply(body_text, vague_reply=DRAW_VAGUE_REPLY),
                            )
                        )
                        return True
                    if upstream_error_should_skip_backend(body_text):
                        skip_backend = True
                        break
                    if has_more_opts:
                        continue
                    break
                if has_more_opts:
                    continue
                break
            if http_status_should_skip_backend(status):
                skip_backend = True
                if op == "edits" and http_status_edits_unsupported(status) and edits_abort_holder is not None:
                    edits_abort_holder[0] = True
                if still_retrying:
                    logger.info(
                        f"bot [{bot_id}] draw {op} backend={backend.label} "
                        f"status={status} in group [{group_id}], trying next backend",
                    )
                else:
                    logger.warning(
                        f"bot [{bot_id}] draw {op} failed in group [{group_id}]: "
                        f"backend={backend.label} status={status} body={body_text[:500]}",
                    )
                break
            if status == 400 and http_body_rejects_response_format(body_text):
                skip_backend = True
                if still_retrying:
                    logger.info(
                        f"bot [{bot_id}] draw {op} backend={backend.label} "
                        f"rejects response_format in group [{group_id}], trying next backend",
                    )
                break
            if http_status_should_try_next_param(status) and has_more_opts:
                logger.info(
                    f"bot [{bot_id}] draw {op} backend={backend.label} "
                    f"status={status} in group [{group_id}], trying next params",
                )
                continue
            if still_retrying:
                logger.info(
                    f"bot [{bot_id}] draw {op} backend={backend.label} "
                    f"status={status} in group [{group_id}], trying next",
                )
            else:
                logger.warning(
                    f"bot [{bot_id}] draw {op} failed in group [{group_id}]: "
                    f"backend={backend.label} status={status} body={body_text[:500]}",
                )
            break
        if skip_backend:
            continue
        if edits_abort_holder is not None and edits_abort_holder[0]:
            break
    return False


async def finish_draw_failure(
    matcher,
    user_id: int,
    last_body: str,
    *,
    at_user_id: int | None = _AT_USER_UNSET,
    last_status: int | None = None,
) -> None:
    record_draw_stats(ok=False, gateway="manual", provider="exhausted", source="plugin")
    reply_at = resolve_reply_at_user_id(user_id, at_user_id)
    # last_status：与本地扩展签名对齐；社区版失败文案不暴露 HTTP 细节。
    if last_status not in (None, 0, 200):
        logger.info(f"draw exhausted last_status={last_status}")
    await matcher.finish(
        optional_message_at_user(
            reply_at,
            user_failure_reply(last_body, vague_reply=DRAW_VAGUE_REPLY),
        )
    )
