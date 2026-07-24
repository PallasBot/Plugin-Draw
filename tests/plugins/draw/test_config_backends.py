from unittest.mock import patch

from pallas_plugin_draw.config import (
    Config,
    ImageGenSettings,
    migrate_legacy_gateway_config,
    normalize_image_base_url,
)


def test_migrate_legacy_gateway_promotes_first_backend() -> None:
    cfg = Config(
        pallas_image_base_url="",
        pallas_image_api_key="",
        pallas_image_model="gpt-image-2",
        pallas_image_api_backends=[
            {
                "base_url": "https://router.example/api/",
                "api_key": "sk-first",
                "model": "openai/gpt-image-2",
                "name": "主站",
            },
            {"base_url": "https://b.example/", "api_key": "sk-b"},
        ],
    )
    out = migrate_legacy_gateway_config(cfg)
    assert out.pallas_image_base_url == "https://router.example/api/"
    assert out.pallas_image_api_key == "sk-first"
    assert out.pallas_image_model == "openai/gpt-image-2"
    assert out.pallas_image_primary_name == "主站"
    assert len(out.pallas_image_api_backends) == 1
    assert out.pallas_image_api_backends[0].base_url == "https://b.example/"

    backends = ImageGenSettings(out).api_backends()
    assert len(backends) == 2
    assert backends[0].label == "primary"
    assert backends[0].name == "主站"


def test_api_backends_keeps_provider_id() -> None:
    def find(pid: str):
        return {
            "id": pid,
            "base_url": f"https://{pid}.example/v1",
            "api_key": f"sk-{pid}",
            "default_model": "m",
        }

    cfg = Config(
        pallas_image_provider_id="prov-a",
        pallas_image_base_url="",
        pallas_image_api_key="",
        pallas_image_model="m1",
        pallas_image_api_backends=[
            {"provider_id": "prov-b", "base_url": "", "api_key": "", "model": "m2"},
        ],
    )
    with (
        patch("pallas_plugin_draw.config.find_provider", side_effect=find),
        patch(
            "pallas_plugin_draw.config.resolve_provider_base_url",
            side_effect=lambda row: row["base_url"],
        ),
        patch(
            "pallas_plugin_draw.config.resolve_provider_api_key",
            side_effect=lambda row: row["api_key"],
        ),
    ):
        backends = ImageGenSettings(cfg).api_backends()
    assert backends[0].provider_id == "prov-a"
    assert backends[1].provider_id == "prov-b"


def test_api_backends_keeps_cost_per_image() -> None:
    cfg = Config(
        pallas_image_base_url="https://primary.example/",
        pallas_image_api_key="sk-p",
        pallas_image_model="m1",
        pallas_image_cost_per_image=0.05,
        pallas_image_api_backends=[
            {
                "base_url": "https://b.example/",
                "api_key": "sk-b",
                "model": "m2",
                "cost_per_image": 0.12,
            },
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert backends[0].cost_per_image == 0.05
    assert backends[1].cost_per_image == 0.12


def test_migrate_legacy_gateway_skips_when_primary_set() -> None:
    cfg = Config(
        pallas_image_base_url="https://primary.example/",
        pallas_image_api_key="sk-p",
        pallas_image_api_backends=[{"base_url": "https://b.example/", "api_key": "sk-b"}],
    )
    out = migrate_legacy_gateway_config(cfg)
    assert out.pallas_image_base_url == "https://primary.example/"
    assert len(out.pallas_image_api_backends) == 1


def test_migrate_legacy_gateway_skips_when_provider_id_set() -> None:
    cfg = Config(
        pallas_image_provider_id="openai",
        pallas_image_base_url="",
        pallas_image_api_key="",
        pallas_image_api_backends=[
            {"base_url": "https://b.example/", "api_key": "sk-b"},
        ],
    )
    out = migrate_legacy_gateway_config(cfg)
    assert out.pallas_image_provider_id == "openai"
    assert out.pallas_image_base_url == ""
    assert len(out.pallas_image_api_backends) == 1


def test_normalize_image_base_url_strips_v1() -> None:
    assert normalize_image_base_url("https://api.openai.com/v1/") == "https://api.openai.com"
    assert normalize_image_base_url("https://api.openai.com/v1") == "https://api.openai.com"
    assert normalize_image_base_url("https://gw.example/api/") == "https://gw.example/api"


def test_api_backends_inherits_primary_provider() -> None:
    row = {
        "id": "openai",
        "enabled": True,
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-from-provider",
        "default_model": "gpt-image-2",
    }
    with (
        patch("pallas_plugin_draw.config.find_provider", return_value=row),
        patch("pallas_plugin_draw.config.resolve_provider_base_url", return_value=row["base_url"]),
        patch("pallas_plugin_draw.config.resolve_provider_api_key", return_value=row["api_key"]),
    ):
        cfg = Config(
            pallas_image_provider_id="openai",
            pallas_image_base_url="",
            pallas_image_api_key="",
            pallas_image_model="",
            pallas_image_primary_name="OpenAI",
        )
        backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 1
    assert backends[0].label == "primary"
    assert backends[0].base_url == "https://api.openai.com"
    assert backends[0].api_key == "sk-from-provider"
    assert backends[0].model == "gpt-image-2"
    assert backends[0].name == "OpenAI"


def test_api_backends_provider_model_override() -> None:
    row = {
        "id": "openai",
        "enabled": True,
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-from-provider",
        "default_model": "gpt-image-2",
    }
    with (
        patch("pallas_plugin_draw.config.find_provider", return_value=row),
        patch("pallas_plugin_draw.config.resolve_provider_base_url", return_value=row["base_url"]),
        patch("pallas_plugin_draw.config.resolve_provider_api_key", return_value=row["api_key"]),
    ):
        cfg = Config(
            pallas_image_provider_id="openai",
            pallas_image_model="custom-image",
        )
        backends = ImageGenSettings(cfg).api_backends()
    assert backends[0].model == "custom-image"


def test_api_backends_skips_missing_provider() -> None:
    with patch("pallas_plugin_draw.config.find_provider", return_value=None):
        cfg = Config(
            pallas_image_provider_id="missing",
            pallas_image_base_url="",
            pallas_image_api_key="",
        )
        backends = ImageGenSettings(cfg).api_backends()
    assert backends == []


def test_api_backends_fallback_provider_id() -> None:
    row = {
        "id": "backup",
        "enabled": True,
        "base_url": "https://backup.example/v1/",
        "api_key": "sk-backup",
        "default_model": "img-b",
    }

    def find(pid: str, **_kwargs):
        return row if pid == "backup" else None

    with (
        patch("pallas_plugin_draw.config.find_provider", side_effect=find),
        patch(
            "pallas_plugin_draw.config.resolve_provider_base_url",
            side_effect=lambda r: str(r.get("base_url") or ""),
        ),
        patch(
            "pallas_plugin_draw.config.resolve_provider_api_key",
            side_effect=lambda r: str(r.get("api_key") or ""),
        ),
    ):
        cfg = Config(
            pallas_image_base_url="https://primary.example/",
            pallas_image_api_key="sk-p",
            pallas_image_model="m1",
            pallas_image_api_backends=[{"provider_id": "backup", "name": "备"}],
        )
        backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 2
    assert backends[1].base_url == "https://backup.example"
    assert backends[1].api_key == "sk-backup"
    assert backends[1].model == "m1"
    assert backends[1].name == "备"


def test_api_backends_dedupe_same_url_key_model() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_api_backends=[
            {"base_url": "https://api.example.com/", "api_key": "sk-a", "model": "m1"},
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 1
    assert backends[0].label == "primary"


def test_api_backends_propagate_omit_response_format() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_api_backends=[
            {
                "base_url": "https://gateway.example.net/api/",
                "api_key": "sk-b",
                "model": "m2",
                "omit_response_format": True,
            },
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert len(backends) == 2
    assert backends[0].omit_response_format is False
    assert backends[1].omit_response_format is True


def test_api_backends_propagate_display_name() -> None:
    cfg = Config(
        pallas_image_base_url="https://api.example.com/",
        pallas_image_api_key="sk-a",
        pallas_image_model="m1",
        pallas_image_primary_name="主站",
        pallas_image_api_backends=[
            {"name": "备线 A", "base_url": "https://b.example/", "api_key": "sk-b"},
        ],
    )
    backends = ImageGenSettings(cfg).api_backends()
    assert backends[0].name == "主站"
    assert backends[1].name == "备线 A"
