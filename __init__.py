"""
Neural Context Plugin — Hermes Agent Plugin Entrypoint.

Intercepts turns before LLM execution, extracts true intent and entities,
prioritizes repository governance, and injects winning skills in a single step
while preserving KV-cache stability on the user message turn.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure neural_context package is importable when loaded by path
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from neural_context.cli import handle_cli, register_cli
from neural_context.config import PluginConfig, get_hermes_home
from neural_context.models import RouteResult, TriageAnalysis
from neural_context.receipt import ReceiptEmitter
from neural_context.router import ContextCoordinator
from neural_context.sidecar import SidecarIndex
from neural_context.triage import CognitiveTriage

PLUGIN_NAME = "focus"
PLUGIN_VERSION = "0.1.0"

logger = logging.getLogger(PLUGIN_NAME)

# Shared singleton instances per process
_config: Optional[PluginConfig] = None
_sidecar: Optional[SidecarIndex] = None
_triage: Optional[CognitiveTriage] = None
_coordinator: Optional[ContextCoordinator] = None
_receipt_emitter: Optional[ReceiptEmitter] = None

# Thread-safe / session-isolated store for turn state: key -> (TriageAnalysis, RouteResult, platform, model)
_session_turns: Dict[str, Tuple[TriageAnalysis, RouteResult, str, str]] = {}


def extract_message_text(user_message: Any) -> str:
    """
    Extract clean textual prompt from diverse channel formats:
    plain text (CLI), list of content blocks (Discord/Telegram/WhatsApp/Desktop),
    or multimodal payload dicts.
    """
    if isinstance(user_message, str):
        return user_message.strip()
    if isinstance(user_message, list):
        parts = []
        for item in user_message:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                val = item.get("text") or item.get("content")
                if isinstance(val, str):
                    parts.append(val)
        return " ".join(parts).strip()
    if isinstance(user_message, dict):
        val = user_message.get("text") or user_message.get("content")
        if isinstance(val, str):
            return val.strip()
    return str(user_message or "").strip()


def _get_services(ctx: Any = None):
    global _config, _sidecar, _triage, _coordinator, _receipt_emitter
    if _config is None:
        _config = PluginConfig.load_from_ctx(ctx)
    if _sidecar is None:
        _sidecar = SidecarIndex(db_path=_config.sidecar_db_path)
        # Initial scan if index is empty
        if _sidecar.n_docs == 0:
            skills_dir = get_hermes_home() / "skills"
            if skills_dir.is_dir():
                _sidecar.sync_skills(list(skills_dir.rglob("SKILL.md")))
    if _triage is None:
        _triage = CognitiveTriage(ctx=ctx)
    if _coordinator is None:
        _coordinator = ContextCoordinator(config=_config, sidecar=_sidecar, ctx=ctx)
    if _receipt_emitter is None:
        _receipt_emitter = ReceiptEmitter(receipts_path=_config.receipts_path)

    return _config, _sidecar, _triage, _coordinator, _receipt_emitter


def on_pre_llm_call(user_message: Any = None, **kwargs: Any) -> Optional[Dict[str, str]]:
    """
    Hook body called before LLM inference across any platform (CLI, WhatsApp, Telegram, Discord, Desktop).
    Appends ephemeral context to the USER message turn.
    Returns None if nothing fits or on any error (fail-closed).
    """
    global _session_turns, _config
    ctx = kwargs.get("_ctx")
    if _config is None:
        _config = PluginConfig.load_from_ctx(ctx)

    if not _config.enabled or not _config.pre_hook_enabled:
        return None

    text = extract_message_text(user_message)
    if not text:
        return None

    turn_id = str(kwargs.get("turn_id", ""))
    session_id = str(kwargs.get("session_id", ""))
    turn_key = f"{session_id}:{turn_id}"

    platform = str(kwargs.get("platform") or getattr(getattr(ctx, "agent", None), "platform", None) or "cli").lower()
    model_name = str(kwargs.get("model") or getattr(getattr(ctx, "agent", None), "model", None) or "").strip()

    try:
        config, sidecar, triage, coordinator, _ = _get_services(ctx)
        cwd = kwargs.get("cwd") or getattr(ctx, "cwd", None)
        analysis = triage.analyze(text, cwd=cwd)
        result = coordinator.execute(analysis, user_message=text, cwd=cwd)

        # Store session/turn state isolated from concurrent sessions
        _session_turns[turn_key] = (analysis, result, platform, model_name)

        if result.combined_context_block:
            return {"context": result.combined_context_block}
    except Exception as exc:
        logger.warning("focus pre_llm_call error (fail-closed): %s", exc)

    return None


def on_post_llm_call(response: Any = None, **kwargs: Any) -> None:
    """
    Hook called after LLM inference.
    Writes verifiable execution record to receipts.jsonl.
    """
    global _session_turns
    turn_id = str(kwargs.get("turn_id", ""))
    session_id = str(kwargs.get("session_id", ""))
    turn_key = f"{session_id}:{turn_id}"

    # Retrieve and clean up turn state for this exact turn (no unsafe cross-turn fallback)
    turn_data = _session_turns.pop(turn_key, None)
    if turn_data is None:
        return

    analysis, route_result, platform, model_name = turn_data
    if analysis is None or route_result is None:
        return

    try:
        ctx = kwargs.get("_ctx")
        _, _, _, _, receipt_emitter = _get_services(ctx)
        receipt_emitter.record_turn(
            turn_id=turn_id,
            session_id=session_id,
            analysis=analysis,
            route_result=route_result,
            platform=platform,
            model=model_name
        )
    except Exception as exc:
        logger.warning("focus post_llm_call receipt error: %s", exc)


def register(ctx: Any) -> None:
    """
    Wires hooks, CLI commands, and inspection tools into Hermes Agent runtime.
    Guaranteed fail-safe execution and strictly lazy initialization.
    """
    # 1. Register pre_llm_call hook
    def pre_hook_handler(**kwargs: Any) -> Optional[Dict[str, str]]:
        kwargs["_ctx"] = ctx
        return on_pre_llm_call(**kwargs)

    ctx.register_hook("pre_llm_call", pre_hook_handler)

    # 2. Register post_llm_call hook
    def post_llm_handler(**kwargs: Any) -> None:
        kwargs["_ctx"] = ctx
        on_post_llm_call(**kwargs)

    ctx.register_hook("post_llm_call", post_llm_handler)

    # 3. Register inspection tools
    if hasattr(ctx, "register_tool"):
        def focus_status_tool(args: dict, **kw: Any) -> str:
            cfg, sidecar, _, _, receipt_emitter = _get_services(ctx)
            stats = receipt_emitter.get_summary_stats()
            return (
                f"Focus Context Engine Active | Indexed Skills: {sidecar.n_docs} | "
                f"Backend: {cfg.router_backend} | Total Saved Tokens: {stats.get('total_tokens_saved', 0)}"
            )

        status_schema = {
            "name": "focus_status",
            "description": "Inspect the operational status of the Focus context engine, sidecar index, and token savings.",
            "parameters": {"type": "object", "properties": {}}
        }

        ctx.register_tool(
            name="focus_status",
            toolset="focus",
            handler=focus_status_tool,
            description="Inspect the operational status of the Focus context engine, sidecar index, and token savings.",
            schema=status_schema,
        )

        def focus_route_tool(args: dict, **kw: Any) -> str:
            query = args.get("query", "")
            if not query:
                return "Error: query is required."
            _, _, triage, coordinator, _ = _get_services(ctx)
            analysis = triage.analyze(query)
            result = coordinator.execute(analysis, user_message=query)
            selected = result.selected_skill.name if result.selected_skill else "None"
            return f"Route Result -> Winner: {selected} | Latency: {result.triage_latency_ms:.1f}ms | Status: {result.status}"

        route_schema = {
            "name": "focus_route",
            "description": "Test or manually inspect cognitive triage and skill routing for a given user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user prompt or task description to evaluate."}
                },
                "required": ["query"]
            }
        }

        ctx.register_tool(
            name="focus_route",
            toolset="focus",
            handler=focus_route_tool,
            description="Test or manually inspect cognitive triage and skill routing for a given user query.",
            schema=route_schema,
        )

    # 4. Register CLI command
    if hasattr(ctx, "register_cli_command"):
        def cli_setup_fn(cmd_parser: Any) -> None:
            register_cli(cmd_parser)

        def cli_handler_fn(args: Any) -> int:
            cfg, sidecar, _, _, receipt_emitter = _get_services(ctx)
            return handle_cli(args, cfg, sidecar, receipt_emitter)

        ctx.register_cli_command(
            name="focus",
            help="Focus CLI: manage sidecar index, configure settings, verify receipts, and run diagnostics.",
            setup_fn=cli_setup_fn,
            handler_fn=cli_handler_fn,
            description="Focus: manage cognitive triage, configure budgets, inspect receipts, and optimize skill index."
        )

    logger.info("focus v%s registered successfully.", PLUGIN_VERSION)
