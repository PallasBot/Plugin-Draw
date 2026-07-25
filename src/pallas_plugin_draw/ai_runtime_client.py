from __future__ import annotations

import asyncio
import base64
import time
import uuid
from dataclasses import dataclass

import httpx
from pallas.api.platform import get_llm_config, llm_server_base_url

from .replies import DRAW_VAGUE_REPLY

_IMAGE_GENERATE_ENDPOINT = "/api/images/generate"
_MEDIA_TASK_ENDPOINT = "/api/media/tasks"


@dataclass(frozen=True)
class AiImageResult:
    ok: bool
    image_bytes: bytes | None = None
    reply_text: str = DRAW_VAGUE_REPLY
    provider_id: str | None = None
    backend_id: str | None = None
    pending_callback: bool = False


def image_generate_endpoint() -> str:
    cfg = get_llm_config()
    return llm_server_base_url(cfg) + _IMAGE_GENERATE_ENDPOINT


def media_task_endpoint() -> str:
    cfg = get_llm_config()
    return llm_server_base_url(cfg) + _MEDIA_TASK_ENDPOINT


def media_task_status_endpoint(task_id: str) -> str:
    cfg = get_llm_config()
    return llm_server_base_url(cfg) + f"{_MEDIA_TASK_ENDPOINT}/{task_id}"


def should_use_task_mode(*, ref_urls: list[str], timeout_sec: float) -> bool:
    if ref_urls:
        return True
    return timeout_sec >= 90.0


def reference_urls_for_payload(ref_urls: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in ref_urls:
        token = (url or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def gateway_payload_from_backends(backends: list | None) -> dict | None:
    """将 Bot ImageApiBackend 列表转为 AI payload.gateway（不含日志密钥）。"""
    if not backends:
        return None
    rows: list[dict] = []
    for item in backends:
        base_url = str(getattr(item, "base_url", "") or "").strip()
        api_key = str(getattr(item, "api_key", "") or "").strip()
        if not base_url or not api_key:
            continue
        rows.append({
            "base_url": base_url,
            "api_key": api_key,
            "model": str(getattr(item, "model", "") or "").strip(),
            "omit_response_format": bool(getattr(item, "omit_response_format", False)),
            "name": str(getattr(item, "name", "") or getattr(item, "label", "") or "").strip(),
        })
    if not rows:
        return None
    return {"backends": rows}


def build_image_request_payload(
    *,
    request_id: str,
    bot_id: int,
    group_id: int,
    user_id: int,
    prompt: str,
    ref_urls: list[str],
    timeout_sec: float,
    force_task_mode: bool,
    gateway: dict | None = None,
) -> dict:
    payload_body: dict = {
        "prompt": prompt,
        "reference_urls": reference_urls_for_payload(ref_urls),
    }
    if gateway:
        payload_body["gateway"] = gateway
    return {
        "request_id": request_id,
        "capability": "image.generate",
        "caller": {
            "source": "bot",
            "bot_id": bot_id,
            "plugin": "draw",
        },
        "context": {
            "group_id": group_id,
            "user_id": user_id,
            "metadata": {
                "submitted_at": int(time.time()),
            },
        },
        "policy": {
            "mode": "default",
            "timeout_sec": timeout_sec,
            "allow_fallback": True,
            "prefer_local": False,
            "force_task_mode": force_task_mode,
            "deliver_mode": "callback" if force_task_mode else "poll",
        },
        "payload": payload_body,
    }


def image_result_from_body(body: dict) -> AiImageResult:
    result_state = str(body.get("result_state") or "").strip().lower()
    if result_state == "accepted":
        task_id = str(body.get("task_id") or "").strip()
        if not task_id:
            return AiImageResult(ok=False, reply_text="AI 任务已受理但缺少 task_id")
        return AiImageResult(ok=False, reply_text=f"__task__:{task_id}")
    if result_state != "success":
        return AiImageResult(
            ok=False,
            reply_text=str(
                (((body.get("error") or {}) if isinstance(body, dict) else {}).get("message")) or DRAW_VAGUE_REPLY
            ),
            provider_id=str(body.get("provider_id") or "") or None,
            backend_id=str(body.get("backend_id") or "") or None,
        )
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        return AiImageResult(ok=False)
    raw_b64 = str(data.get("b64_data") or "").strip()
    if not raw_b64:
        return AiImageResult(ok=False)
    try:
        blob = base64.b64decode(raw_b64, validate=True)
    except (ValueError, TypeError):
        return AiImageResult(ok=False)
    return AiImageResult(
        ok=True,
        image_bytes=blob,
        provider_id=str(body.get("provider_id") or "") or None,
        backend_id=str(body.get("backend_id") or "") or None,
    )


async def poll_image_task_result(
    client: httpx.AsyncClient,
    *,
    task_id: str,
    timeout_sec: float,
) -> AiImageResult:
    deadline = time.monotonic() + max(5.0, timeout_sec)
    poll_interval = 1.0
    while time.monotonic() < deadline:
        try:
            response = await client.get(
                media_task_status_endpoint(task_id),
                timeout=httpx.Timeout(min(15.0, timeout_sec), connect=min(10.0, timeout_sec)),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return AiImageResult(ok=False, reply_text=str(exc)[:200] or DRAW_VAGUE_REPLY)

        body = response.json()
        state = str(body.get("state") or "").strip().lower()
        if state == "succeeded":
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, dict):
                return AiImageResult(ok=False)
            raw_b64 = str(data.get("b64_data") or "").strip()
            if not raw_b64:
                return AiImageResult(ok=False)
            try:
                blob = base64.b64decode(raw_b64, validate=True)
            except (ValueError, TypeError):
                return AiImageResult(ok=False)
            return AiImageResult(
                ok=True,
                image_bytes=blob,
                provider_id=str(body.get("provider_id") or "") or None,
                backend_id=str(body.get("backend_id") or "") or None,
            )
        if state in {"failed", "cancelled"}:
            err = body.get("error") if isinstance(body, dict) else None
            message = DRAW_VAGUE_REPLY
            if isinstance(err, dict):
                message = str(err.get("message") or message)
            return AiImageResult(ok=False, reply_text=message)
        await asyncio.sleep(poll_interval)
    return AiImageResult(ok=False, reply_text="AI 画图任务等待超时")


async def generate_image_via_ai_service(
    client: httpx.AsyncClient,
    *,
    bot_id: int,
    group_id: int,
    user_id: int,
    prompt: str,
    ref_urls: list[str],
    timeout_sec: float,
    count_usage: bool = False,
    gateway: dict | None = None,
) -> AiImageResult:
    from pallas.api.config import TaskManager

    request_id = f"draw-{bot_id}-{group_id}-{user_id}-{uuid.uuid4().hex[:8]}"
    use_task_mode = should_use_task_mode(ref_urls=ref_urls, timeout_sec=timeout_sec)
    payload = build_image_request_payload(
        request_id=request_id,
        bot_id=bot_id,
        group_id=group_id,
        user_id=user_id,
        prompt=prompt,
        ref_urls=ref_urls,
        timeout_sec=timeout_sec,
        force_task_mode=use_task_mode,
        gateway=gateway,
    )
    endpoint = media_task_endpoint() if use_task_mode else image_generate_endpoint()
    if use_task_mode:
        await TaskManager.add_task(
            request_id,
            {
                "bot_id": bot_id,
                "group_id": group_id,
                "user_id": user_id,
                "task_type": "draw",
                "start_time": time.time(),
                "count_usage": count_usage,
            },
        )
    try:
        response = await client.post(
            endpoint,
            json=payload,
            timeout=httpx.Timeout(timeout_sec, connect=min(15.0, timeout_sec)),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if use_task_mode:
            await TaskManager.remove_task(request_id)
        detail = exc.response.text[:200] if exc.response is not None else str(exc)
        return AiImageResult(ok=False, reply_text=detail or DRAW_VAGUE_REPLY)
    except httpx.HTTPError as exc:
        if use_task_mode:
            await TaskManager.remove_task(request_id)
        return AiImageResult(ok=False, reply_text=str(exc)[:200] or DRAW_VAGUE_REPLY)

    body = response.json()
    if use_task_mode:
        result_state = str(body.get("result_state") or "").strip().lower()
        if result_state == "accepted" and str(body.get("task_id") or "").strip():
            return AiImageResult(
                ok=False,
                pending_callback=True,
                provider_id=str(body.get("provider_id") or "") or None,
                backend_id=str(body.get("backend_id") or "") or None,
            )
        await TaskManager.remove_task(request_id)
        err = body.get("error") if isinstance(body, dict) else None
        message = DRAW_VAGUE_REPLY
        if isinstance(err, dict):
            message = str(err.get("message") or message)
        return AiImageResult(
            ok=False,
            reply_text=message,
            provider_id=str(body.get("provider_id") or "") or None,
            backend_id=str(body.get("backend_id") or "") or None,
        )

    result = image_result_from_body(body)
    if result.ok:
        return result
    if result.reply_text.startswith("__task__:"):
        task_id = result.reply_text.removeprefix("__task__:")
        return await poll_image_task_result(client, task_id=task_id, timeout_sec=timeout_sec)
    return result
