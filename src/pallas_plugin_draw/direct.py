from __future__ import annotations

from nonebot.adapters.onebot.v11 import Message, MessageSegment
from nonebot.exception import FinishedException
from pallas.api.runtime import (
    DirectCommandContext,
    DirectCommandResult,
    completion_effect,
    matcher_fallback,
    register_prefix_command_handler,
)

from .draw import pallas_draw_handle

DRAW_PREFIX = "牛牛画画"


def draw_command_args(message: Message) -> Message:
    args = Message()
    prefix_removed = False
    for segment in message:
        copied = MessageSegment(segment.type, dict(segment.data))
        if not prefix_removed and copied.type == "text":
            text = str(copied.data.get("text") or "")
            if text.lstrip().startswith(DRAW_PREFIX):
                leading = len(text) - len(text.lstrip())
                copied.data["text"] = text[leading + len(DRAW_PREFIX) :].lstrip()
                prefix_removed = True
        args.append(copied)
    return args


async def draw(context: DirectCommandContext) -> DirectCommandResult:
    command_text = context.command_text.strip()
    if command_text != DRAW_PREFIX and not command_text.startswith(f"{DRAW_PREFIX} "):
        return matcher_fallback("invalid_command_shape")
    args = draw_command_args(context.event.get_message())

    async def run() -> None:
        try:
            await pallas_draw_handle(context.bot, context.event, args)
        except FinishedException:
            pass

    return DirectCommandResult(effects=(completion_effect("draw.execute", run),))


DRAW_DECLARATION = register_prefix_command_handler(
    handler_id="draw.draw.direct",
    module="draw",
    prefixes=(DRAW_PREFIX,),
    command_id="draw.draw",
    execute=draw,
)
