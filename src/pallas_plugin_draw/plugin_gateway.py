"""插件直连图像网关（默认 plugin_runtime）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nonebot import logger

if TYPE_CHECKING:
    import httpx

    from .config import ImageApiBackend, ImageGenSettings
    from .draw_attempts import DrawDeadline
    from .image_request_options import ImageGenRequestOptions

from .draw_attempts import (
    finish_draw_failure,
    http_status_edits_unsupported,
    run_backend_param_attempts,
)
from .image_api import (
    download_reference_images,
    generations_payload,
    image_edits_endpoint,
    image_gen_auth_headers_json,
    image_gen_endpoint,
    post_edits_with_transport,
    post_generations_with_transport,
    resolve_reference_urls_for_upstream,
)


def image_backends_with_endpoint(
    backends: list[ImageApiBackend],
    endpoint_fn,
) -> list[ImageApiBackend]:
    return [b for b in backends if endpoint_fn(b)]


async def run_plugin_gateway_draw(
    matcher,
    http_client: httpx.AsyncClient,
    *,
    cfg: ImageGenSettings,
    bot_id: int,
    group_id: int,
    user_id: int,
    usage_key: tuple[int, int],
    count_usage: bool,
    paid_credit: bool = False,
    text: str,
    ref_urls: list[str],
    gen_prompt: str,
    deadline: DrawDeadline,
) -> bool:
    backends = cfg.api_backends()
    default_prompt = (cfg.default_edit_prompt or "生成图像").strip()
    last_body: list[str] = [""]
    last_status: list[int] = [0]

    ref_dl_timeout = min(
        cfg.ref_download_timeout,
        max(1.0, deadline.remaining_seconds()),
    )
    ref_result = await resolve_reference_urls_for_upstream(
        http_client,
        ref_urls,
        download_timeout=ref_dl_timeout,
    )
    ref_urls = ref_result.inline_urls
    if ref_result.failed_tokens:
        logger.warning(
            f"bot [{bot_id}] draw ref resolve partial in group [{group_id}] failed={len(ref_result.failed_tokens)}",
        )

    if ref_urls and cfg.use_edits_for_reference_images:
        ref_dl_timeout = min(
            cfg.ref_download_timeout,
            max(1.0, deadline.remaining_seconds()),
        )
        blobs = await download_reference_images(
            http_client,
            ref_urls,
            download_timeout=ref_dl_timeout,
        )
        if blobs:
            if len(ref_urls) > len(blobs):
                logger.warning(
                    f"bot [{bot_id}] draw ref download partial in group [{group_id}]: "
                    f"requested {len(ref_urls)} refs, got {len(blobs)} blobs",
                )
            edit_prompt = text.strip() or default_prompt
            edit_backends = image_backends_with_endpoint(backends, image_edits_endpoint)
            edits_abort = [False]

            async def post_edits(
                backend: ImageApiBackend,
                req_opts: ImageGenRequestOptions,
                req_timeout_cap: float,
            ) -> tuple[int, str]:
                return await post_edits_with_transport(
                    http_client,
                    blobs,
                    edit_prompt,
                    backend,
                    options=req_opts,
                    req_timeout_cap=req_timeout_cap,
                )

            if await run_backend_param_attempts(
                matcher,
                http_client,
                bot_id,
                group_id,
                user_id,
                usage_key,
                count_usage,
                deadline,
                "edits",
                edit_backends,
                with_ref_urls=False,
                post_request=post_edits,
                last_body_holder=last_body,
                last_status_holder=last_status,
                edits_abort_holder=edits_abort,
                paid_afdian=paid_credit,
            ):
                return True
            if edits_abort[0] or http_status_edits_unsupported(last_status[0]):
                logger.info(
                    f"bot [{bot_id}] draw edits unsupported status={last_status[0]} "
                    f"in group [{group_id}], fallback to generations",
                )
            else:
                logger.warning(
                    f"bot [{bot_id}] draw edits exhausted in group [{group_id}], fallback to generations",
                )

    payload_model = backends[0].model if backends else cfg.model
    gen_backends = image_backends_with_endpoint(backends, image_gen_endpoint)

    async def post_generations(
        backend: ImageApiBackend,
        req_opts: ImageGenRequestOptions,
        req_timeout_cap: float,
    ) -> tuple[int, str]:
        gen_ep = image_gen_endpoint(backend)
        headers = image_gen_auth_headers_json(backend)
        payload = generations_payload(
            gen_prompt,
            ref_urls,
            model=backend.model or payload_model,
            backend=backend,
            options=req_opts,
        )
        return await post_generations_with_transport(
            http_client,
            gen_ep,
            headers,
            payload,
            req_timeout_cap=req_timeout_cap,
        )

    if await run_backend_param_attempts(
        matcher,
        http_client,
        bot_id,
        group_id,
        user_id,
        usage_key,
        count_usage,
        deadline,
        "generations",
        gen_backends,
        with_ref_urls=bool(ref_urls),
        post_request=post_generations,
        last_body_holder=last_body,
        last_status_holder=last_status,
        paid_afdian=paid_credit,
    ):
        return True

    logger.error(
        f"bot [{bot_id}] draw generations exhausted backends in group [{group_id}]",
    )
    await finish_draw_failure(matcher, user_id, last_body[0])
    return False
