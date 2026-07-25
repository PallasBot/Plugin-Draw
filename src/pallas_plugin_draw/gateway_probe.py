from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from curl_cffi.requests import AsyncSession as CffiAsyncSession
from curl_cffi.requests import RequestsError as CffiRequestsError
from nonebot import logger
from pallas.api.probe import ServiceProbeResult, format_probe_lines

from .config import Config, ImageApiBackend, ImageGenSettings, get_draw_config
from .image_api import (
    cffi_error_is_timeout,
    image_api_base,
    image_gen_auth_headers_json,
    image_gen_config,
)


def transport_mode_for_settings(settings: ImageGenSettings) -> str:
    raw = (settings.http_transport or "auto").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    if raw in ("httpx", "curl", "cffi"):
        return raw
    return "auto"


IMAGE_PROBE_CATEGORY = "牛牛画画"


def backend_site_name(index: int) -> str:
    return f"备线{index}"


def backend_display_site(backend: ImageApiBackend, index: int) -> str:
    name = (backend.name or "").strip()
    if name:
        return name
    if backend.label == "primary":
        return "主网关"
    return backend_site_name(index)


def models_probe_urls(backend: ImageApiBackend) -> list[str]:
    base = image_api_base(backend)
    if not base:
        return []
    root = base.rstrip("/")
    parts = ["models"]
    if not root.endswith("/v1"):
        parts.insert(0, "v1/models")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        url = urljoin(base, part)
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def format_gateway_status_lines(results: list[ServiceProbeResult]) -> list[str]:
    return format_probe_lines(results)


def format_gateway_status_text(results: list[ServiceProbeResult]) -> str:
    return "\n".join(format_gateway_status_lines(results))


def image_gen_settings_from_draft(draft: dict[str, Any] | None) -> ImageGenSettings:
    if not draft:
        return image_gen_config
    from pallas.api.config import normalize_patch_value

    current = get_draw_config().model_dump(mode="python")
    merged = dict(current)
    for key, value in draft.items():
        if key not in Config.model_fields:
            continue
        merged[key] = normalize_patch_value(Config.model_fields[key], value)
    from .config import migrate_legacy_gateway_config

    return ImageGenSettings(migrate_legacy_gateway_config(Config.model_validate(merged)))


def probe_timeout_sec(settings: ImageGenSettings) -> float:
    return min(30.0, max(5.0, settings.request_timeout))


async def httpx_probe_get(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    timeout_sec: float,
) -> tuple[int, str]:
    r = await client.get(
        url,
        headers=headers,
        timeout=httpx.Timeout(timeout_sec, connect=min(15.0, timeout_sec)),
    )
    return r.status_code, r.text[:400]


async def cffi_probe_get(
    url: str,
    headers: dict[str, str],
    settings: ImageGenSettings,
    timeout_sec: float,
) -> tuple[int, str]:
    impersonate = (settings.tls_impersonate or "").strip()
    if not impersonate:
        raise ValueError("tls_impersonate 为空")
    async with CffiAsyncSession() as session:
        r = await session.get(url, headers=headers, impersonate=impersonate, timeout=timeout_sec)
        return r.status_code, (r.text or "")[:400]


async def probe_url(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    settings: ImageGenSettings,
) -> tuple[int, str]:
    timeout_sec = probe_timeout_sec(settings)
    mode = transport_mode_for_settings(settings)
    if mode == "cffi":
        return await cffi_probe_get(url, headers, settings, timeout_sec)
    if mode == "curl":
        raise RuntimeError("curl 传输未用于网关探测，请改用 httpx 或 auto")
    if mode == "httpx":
        return await httpx_probe_get(client, url, headers, timeout_sec)
    if (settings.tls_impersonate or "").strip():
        try:
            return await cffi_probe_get(url, headers, settings, timeout_sec)
        except (CffiRequestsError, OSError, ValueError) as e:
            if not cffi_error_is_timeout(e):
                logger.warning(f"draw gateway probe cffi failed, fallback httpx: {e}")
    return await httpx_probe_get(client, url, headers, timeout_sec)


async def probe_single_backend(
    client: httpx.AsyncClient,
    settings: ImageGenSettings,
    backend: ImageApiBackend,
    site_index: int,
) -> ServiceProbeResult:
    site = backend_display_site(backend, site_index)
    urls = models_probe_urls(backend)
    if not urls:
        return ServiceProbeResult(
            category=IMAGE_PROBE_CATEGORY,
            site=site,
            ok=False,
            latency_ms=None,
            status_code=None,
            error="未配置",
        )
    headers = image_gen_auth_headers_json(backend)
    started = time.perf_counter()
    last_status: int | None = None
    last_error: str | None = None
    for url in urls:
        try:
            status, _ = await probe_url(client, url, headers, settings)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            last_status = status
            if 200 <= status < 300:
                return ServiceProbeResult(
                    category=IMAGE_PROBE_CATEGORY,
                    site=site,
                    ok=True,
                    latency_ms=elapsed_ms,
                    status_code=status,
                    error=None,
                )
        except httpx.TimeoutException:
            return ServiceProbeResult(
                category=IMAGE_PROBE_CATEGORY,
                site=site,
                ok=False,
                latency_ms=None,
                status_code=None,
                error="超时",
            )
        except httpx.ConnectError:
            last_error = "连接失败"
        except RuntimeError as e:
            last_error = str(e)[:120]
        except Exception as e:  # noqa: BLE001
            last_error = str(e)[:120]
            logger.debug(f"draw gateway probe {backend.label} {url}: {e}")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if last_status is not None:
        return ServiceProbeResult(
            category=IMAGE_PROBE_CATEGORY,
            site=site,
            ok=False,
            latency_ms=elapsed_ms,
            status_code=last_status,
            error=None,
        )
    return ServiceProbeResult(
        category=IMAGE_PROBE_CATEGORY,
        site=site,
        ok=False,
        latency_ms=None,
        status_code=None,
        error=last_error or "不可用",
    )


async def probe_all_backends(
    settings: ImageGenSettings | None = None,
) -> list[ServiceProbeResult]:
    cfg = settings or image_gen_config
    backends = cfg.api_backends()
    if not backends:
        return []
    fallback_idx = 0
    site_indexes: list[int] = []
    for backend in backends:
        if backend.label == "primary":
            site_indexes.append(0)
        else:
            fallback_idx += 1
            site_indexes.append(fallback_idx)
    async with httpx.AsyncClient() as client:
        tasks = [
            probe_single_backend(client, cfg, backend, site_index)
            for backend, site_index in zip(backends, site_indexes, strict=True)
        ]
        return list(await asyncio.gather(*tasks))
