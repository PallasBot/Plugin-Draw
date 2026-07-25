from nonebot.plugin import PluginMetadata
from pallas.api.metadata import (
    PLUGIN_EXTRA_VERSION,
    PLUGIN_HOMEPAGE,
    PLUGIN_MENU_TEMPLATE,
    SCENE_GROUP,
    join_usage,
    usage_line,
)
from pallas.api.platform import llm_command_tool_row
from pallas.api.storage import plugin_storage_list, plugin_storage_row
from pallas.product.llm.knowledge.declare import knowledge_source_row

__plugin_meta__ = PluginMetadata(
    name="牛牛画画",
    description="群内按文字描述生图，支持参考图改图。",
    usage=join_usage(
        usage_line("牛牛画画 …", "按描述生图"),
        usage_line("牛牛画画 + 附图 / 回复图片", "以参考图改图，可多图"),
    ),
    type="application",
    homepage=PLUGIN_HOMEPAGE,
    supported_adapters={"~onebot.v11"},
    extra={
        "version": PLUGIN_EXTRA_VERSION,
        "menu_template": PLUGIN_MENU_TEMPLATE,
        "ingress_route": {"lane": "remote"},
        "command_prefixes": ["牛牛画画"],
        "command_permissions": [
            {"id": "draw.draw", "label": "牛牛画画", "default": "everyone"},
        ],
        "command_limits": [
            {"id": "draw.draw", "cd_sec": 3},
        ],
        "llm_tools": [
            llm_command_tool_row(
                name="draw.image",
                command_id="draw.draw",
                description="根据文字描述生成或修改图片。用户想画画、生图、画图、改图时使用。",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "画面描述，尽量保留用户原话",
                        },
                    },
                    "required": ["prompt"],
                },
                command_template="牛牛画画 {prompt}",
            ),
        ],
        "plugin_storage": plugin_storage_list(
            plugin_storage_row("daily_usage", scope="deploy", label="群内日用量"),
        ),
        "menu_data": [
            {
                "func": "牛牛画画",
                "trigger_method": "on_cmd",
                "trigger_scene": SCENE_GROUP,
                "trigger_condition": "牛牛画画 …",
                "command_permission": "draw.draw",
                "brief_des": "生图或改图",
                "detail_des": "可纯文字，也可附图或回复图片作参考；次数用尽时会提示。",
            },
        ],
        "knowledge_sources": [
            knowledge_source_row(
                source_id="draw.faq",
                title="牛牛画画说明",
                description="群内文字生图与参考图改图",
                chunks=[
                    {
                        "title": "如何生图",
                        "content": (
                            "在群内发送「牛牛画画」加画面描述即可按文字生图；描述尽量具体，保留用户原话效果更好。"
                        ),
                        "keywords": "画画,生图,画图,怎么画,牛牛画画",
                    },
                    {
                        "title": "参考图改图",
                        "content": ("可在「牛牛画画」消息中附图，或回复某张图片并发送改图描述；支持多张参考图。"),
                        "keywords": "改图,参考图,附图,回复图片,参考",
                    },
                    {
                        "title": "与 draw.image 工具的分工",
                        "content": (
                            "闲聊中若用户想画画，可调用 draw.image 工具并带上 prompt；"
                            "口令「牛牛画画 …」与工具效果一致，不要编造其它生图入口。"
                        ),
                        "keywords": "工具,draw.image,口令,生图",
                    },
                ],
            ),
        ],
    },
)

from . import draw as _pallas_draw  # noqa: F401
from . import startup as _draw_startup  # noqa: F401
