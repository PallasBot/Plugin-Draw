import asyncio
import time
from dataclasses import dataclass

from .config import image_gen_config

image_gen_semaphore = asyncio.Semaphore(image_gen_config.max_concurrency)
_semaphore_limit = image_gen_config.max_concurrency
_draw_pending = 0
_draw_pending_lock = asyncio.Lock()


@dataclass(slots=True)
class AiRuntimeCircuitState:
    consecutive_failures: int = 0
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    circuit_open_until: float = 0.0
    recent_failure_reason: str = ""


async def acquire_draw_pending_slot() -> bool:
    """进程内画画任务占位；draw_max_pending<=0 时不限制。"""
    global _draw_pending
    limit = image_gen_config.draw_max_pending
    if limit <= 0:
        return True
    async with _draw_pending_lock:
        if _draw_pending >= limit:
            return False
        _draw_pending += 1
        return True


async def release_draw_pending_slot() -> None:
    global _draw_pending
    limit = image_gen_config.draw_max_pending
    if limit <= 0:
        return
    async with _draw_pending_lock:
        if _draw_pending > 0:
            _draw_pending -= 1


def sync_image_gen_semaphore(max_concurrency: int) -> None:
    global image_gen_semaphore, _semaphore_limit
    limit = max(1, min(int(max_concurrency), 32))
    if limit == _semaphore_limit:
        return
    image_gen_semaphore = asyncio.Semaphore(limit)
    _semaphore_limit = limit


def ai_runtime_circuit_is_open(now: float | None = None) -> bool:
    _ = now
    try:
        from pallas.api.ai_runtime_health import image_runtime_circuit_is_open
    except ImportError:
        return False
    return image_runtime_circuit_is_open()


def record_ai_runtime_success() -> None:
    return


def record_ai_runtime_failure(reason: str) -> None:
    _ = reason


def ai_runtime_circuit_status() -> AiRuntimeCircuitState:
    try:
        from pallas.api.ai_runtime_health import image_runtime_circuit_snapshot
    except ImportError:
        return AiRuntimeCircuitState()
    snap = image_runtime_circuit_snapshot()
    open_now = ai_runtime_circuit_is_open()
    return AiRuntimeCircuitState(
        consecutive_failures=snap.consecutive_failures,
        last_failure_at=0.0,
        last_success_at=0.0,
        circuit_open_until=time.time() + 60.0 if open_now else 0.0,
        recent_failure_reason=snap.recent_failure_reason,
    )
