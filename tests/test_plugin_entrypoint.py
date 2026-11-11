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

    assert "neural_context_status" in registered_tools
    assert "neural_context_route" in registered_tools

    # Crucial check: schema is directly {"name": ..., "parameters": ...}, NOT wrapped in {"type": "function", "function": ...}
    status_schema = registered_tools["neural_context_status"]
    assert "parameters" in status_schema
    assert "function" not in status_schema

    route_schema = registered_tools["neural_context_route"]
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
