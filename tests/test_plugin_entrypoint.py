from pathlib import Path
from unittest.mock import MagicMock
import pytest

import neural_context
from neural_context.models import DomainClassification, RouteResult, TriageAnalysis


def test_plugin_registration_and_schemas():
    mock_ctx = MagicMock()
    mock_ctx.cwd = "/tmp"
    mock_ctx.get_config.side_effect = lambda key, default=None: default

    import __init__ as plugin_entry

    plugin_entry.register(mock_ctx)

    # Verify hooks registered
    registered_hooks = [call.args[0] for call in mock_ctx.register_hook.call_args_list]
    assert "pre_llm_call" in registered_hooks
    assert "post_llm_call" in registered_hooks

    # Verify tools registered with unwrapped schema
    registered_tools = {}
    for call in mock_ctx.register_tool.call_args_list:
        name = call.kwargs.get("name")
        schema = call.kwargs.get("schema")
        registered_tools[name] = schema

    assert "focus_status" in registered_tools
    assert "focus_route" in registered_tools

    # Crucial check: schema is directly {"name": ..., "parameters": ...}, NOT wrapped in {"type": "function", "function": ...}
    status_schema = registered_tools["focus_status"]
    assert "parameters" in status_schema
    assert "function" not in status_schema

    route_schema = registered_tools["focus_route"]
    assert "parameters" in route_schema
    assert "function" not in route_schema
    assert "query" in route_schema["parameters"]["properties"]


def test_session_isolated_hooks(tmp_path):
    import __init__ as plugin_entry

    mock_ctx = MagicMock()
    mock_ctx.cwd = str(tmp_path)
    mock_ctx.get_config.side_effect = lambda key, default=None: default

    # Session 1 Turn 1
    res1 = plugin_entry.on_pre_llm_call(
        user_message="Hola amigo",
        _ctx=mock_ctx,
        session_id="session_alpha",
        turn_id="turn_1"
    )
    # Casual prompt -> returns None (skipped)
    assert res1 is None

    # Check store has session_alpha:turn_1
    key1 = "session_alpha:turn_1"
    assert key1 in plugin_entry._session_turns

    # Post hook should pop and record
    plugin_entry.on_post_llm_call(
        response="Hola",
        _ctx=mock_ctx,
        session_id="session_alpha",
        turn_id="turn_1"
    )
    assert key1 not in plugin_entry._session_turns


def test_plugin_disabled_fast_path(tmp_path):
    import __init__ as plugin_entry

    mock_ctx = MagicMock()
    mock_ctx.cwd = str(tmp_path)
    # Simulate config where plugin is disabled
    mock_ctx.get_config.side_effect = lambda key, default=None: False if key == "enabled" else default

    # Reset any cached config
    plugin_entry._config = None

    res = plugin_entry.on_pre_llm_call(
        user_message="Need to run complex audit",
        _ctx=mock_ctx,
        session_id="session_disabled",
        turn_id="turn_disabled"
    )
    # Must immediately return None without creating turn state
    assert res is None
    assert "session_disabled:turn_disabled" not in plugin_entry._session_turns


def test_services_refresh_native_skill_inventory_without_cwd_suppression(monkeypatch):
    from unittest.mock import Mock
    from neural_context.config import PluginConfig

    import __init__ as plugin_entry

    config = PluginConfig(software_inspection_enabled=False)
    sidecar = Mock()
    inventory = ([], [], {}, {}, {0, 1, 2, 3})
    discover = Mock(return_value=inventory)
    monkeypatch.setattr(plugin_entry, "_config", config)
    monkeypatch.setattr(plugin_entry, "_sidecar", sidecar)
    monkeypatch.setattr(plugin_entry, "_triage", Mock())
    monkeypatch.setattr(plugin_entry, "_coordinator", Mock())
    monkeypatch.setattr(plugin_entry, "_receipt_emitter", Mock())
    monkeypatch.setattr(plugin_entry, "discover_skill_inventory", discover)

    plugin_entry._get_services()
    plugin_entry._get_services()

    assert discover.call_count == 2
    discover.assert_called_with(include_project=False)
    assert sidecar.sync_skills.call_count == 2


def test_multichannel_message_extraction():
    import __init__ as plugin_entry

    # 1. Plain string (CLI)
    assert plugin_entry.extract_message_text("Hello from CLI") == "Hello from CLI"

    # 2. List of text content objects (Discord / Telegram / WhatsApp bots)
    discord_payload = [
        {"type": "text", "text": "Can you check the database status?"}
    ]
    assert plugin_entry.extract_message_text(discord_payload) == "Can you check the database status?"

    # 3. Multimodal list with text and image
    whatsapp_payload = [
        {"type": "text", "text": "Here is the error screenshot:"},
        {"type": "image_url", "image_url": "https://example.com/err.png"}
    ]
    assert plugin_entry.extract_message_text(whatsapp_payload) == "Here is the error screenshot:"

    # 4. Dictionary format (Desktop / Electron IPC)
    desktop_payload = {"content": "Inspect Docker container"}
    assert plugin_entry.extract_message_text(desktop_payload) == "Inspect Docker container"


def test_multichannel_turn_receipt(tmp_path):
    import __init__ as plugin_entry

    mock_ctx = MagicMock()
    mock_ctx.cwd = str(tmp_path)
    mock_ctx.get_config.side_effect = lambda key, default=None: default
    plugin_entry._config = None

    # Pre-call from Telegram bot on Qwen model
    plugin_entry.on_pre_llm_call(
        user_message=[{"text": "Hola bot"}],
        _ctx=mock_ctx,
        session_id="tg_session",
        turn_id="tg_turn_1",
        platform="telegram",
        model="qwen-2.5-coder"
    )

    turn_key = "tg_session:tg_turn_1"
    assert turn_key in plugin_entry._session_turns
    analysis, result, platform, model = plugin_entry._session_turns[turn_key]
    assert platform == "telegram"
    assert model == "qwen-2.5-coder"

    # Post-call pops and cleans up
    plugin_entry.on_post_llm_call(
        response="Hola humano",
        _ctx=mock_ctx,
        session_id="tg_session",
        turn_id="tg_turn_1"
    )
    assert turn_key not in plugin_entry._session_turns
