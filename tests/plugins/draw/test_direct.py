from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message, MessageSegment
from nonebot.exception import FinishedException
from pallas.api.runtime import DirectCommandContext


def make_context(message: Message) -> DirectCommandContext:
    event = GroupMessageEvent.model_construct(
        time=1,
        self_id=1001,
        post_type="message",
        message_type="group",
        sub_type="normal",
        user_id=2002,
        group_id=3003,
        message_id=4004,
        message=message,
        raw_message=str(message),
        reply=None,
    )
    return DirectCommandContext(
        bot=SimpleNamespace(self_id="1001"),
        event=event,
        bot_id=1001,
        group_id=3003,
        message_id=4004,
        command_text=event.get_plaintext().strip(),
    )


def test_draw_command_args_remove_prefix_and_keep_image_segments() -> None:
    from pallas_plugin_draw.direct import draw_command_args

    image = MessageSegment.image("https://example.com/ref.png")
    args = draw_command_args(Message(MessageSegment.text("牛牛画画 小羊") + image))

    assert args.extract_plain_text().strip() == "小羊"
    assert [segment.type for segment in args] == ["text", "image"]


@pytest.mark.asyncio
async def test_direct_draw_runs_existing_handler_as_a_completion_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    from pallas_plugin_draw import direct

    handler = AsyncMock(side_effect=FinishedException)
    monkeypatch.setattr(direct, "pallas_draw_handle", handler)
    context = make_context(Message("牛牛画画 小羊"))

    result = await direct.draw(context)

    assert result.work_jobs == ()
    assert len(result.effects) == 1
    await result.effects[0].run()
    handler.assert_awaited_once()
    assert handler.await_args.args[2].extract_plain_text().strip() == "小羊"


@pytest.mark.asyncio
async def test_direct_draw_runs_handler_inside_matcher_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from nonebot.internal.matcher import current_bot, current_event

    from pallas_plugin_draw import direct

    seen: dict[str, object] = {}

    async def spy(bot, event, args) -> None:
        seen["bot"] = current_bot.get()
        seen["event"] = current_event.get()

    monkeypatch.setattr(direct, "pallas_draw_handle", spy)
    context = make_context(Message("牛牛画画 小羊"))

    result = await direct.draw(context)

    assert len(result.effects) == 1
    await result.effects[0].run()
    assert seen["bot"] is context.bot
    assert seen["event"] is context.event


@pytest.mark.asyncio
async def test_direct_draw_falls_back_for_a_glued_prefix() -> None:
    from pallas_plugin_draw.direct import draw

    result = await draw(make_context(Message("牛牛画画家")))

    assert result.fallback_to_matcher is True
    assert result.fallback_reason == "invalid_command_shape"
