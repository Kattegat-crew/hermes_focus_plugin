"""
Neural Context Plugin — Hermes Agent Plugin Entrypoint.

Intercepts turns before LLM execution, extracts true intent and entities,
prioritizes repository governance, and injects winning skills in a single step
while preserving KV-cache stability on the user message turn.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure neural_context package is importable when loaded by path
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from neural_context.cli import handle_cli, register_cli
from neural_context.config import PluginConfig
from neural_context.models import RouteResult, TriageAnalysis
from neural_context.receipt import ReceiptEmitter
from neural_context.router import ContextCoordinator
from neural_context.sidecar import SidecarIndex, discover_skill_inventory
from neural_context.triage import CognitiveTriage

PLUGIN_NAME = "focus"
PLUGIN_VERSION = "0.1.1"

logger = logging.getLogger(PLUGIN_NAME)


def _ctx_cwd(ctx: Any) -> Optional[str]:
    """Resolve the session working directory from a Hermes PluginContext.

    Hermes exposes ``PluginContext.session_cwd`` as a *method* (``ctx.session_cwd()``); a plain
    ``getattr`` therefore returns a bound method, and passing that into ``Path()`` raised
    ``TypeError`` — which made every ``pre_llm_call`` hook fail closed, so nothing was ever
    injected when running inside a real Hermes runtime.
    """
    for attr in ("session_cwd", "cwd"):
        value = getattr(ctx, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, (str, Path)) and str(value).strip():
            return str(value)
    return None


# Shared singleton instances per process
_config: Optional[PluginConfig] = None
_sidecar: Optional[SidecarIndex] = None
_triage: Optional[CognitiveTriage] = None
_coordinator: Optional[ContextCoordinator] = None
_receipt_emitter: Optional[ReceiptEmitter] = None

# Thread-safe / session-isolated store for turn state: key -> (TriageAnalysis, RouteResult, platform, model)
_session_turns: Dict[str, Tuple[TriageAnalysis, RouteResult, str, str]] = {}
_session_turns_lock = threading.Lock()


def extract_message_text(user_message: Any) -> str:
    """
    Extract clean textual prompt from diverse channel formats:
    plain text (CLI), list of content blocks (Discord/Telegram/WhatsApp/Desktop),
    or multimodal payload dicts.
    """
    if isinstance(user_message, str):
        return user_message.strip()
    if isinstance(user_message, (list, tuple)):
        parts = []
        for item in user_message:
            chunk = extract_message_text(item)
            if chunk:
                parts.append(chunk)
        return " ".join(parts).strip()
    if isinstance(user_message, dict):
        # Extract human text/caption while discarding binary/base64 media payloads (never str(dict))
        for key in ("text", "input_text", "caption"):
            val = user_message.get(key)
            if isinstance(val, str):
                stripped = val.strip()
                if stripped and not stripped.startswith("data:"):
                    return stripped
        for container_key in ("content", "parts"):
            if container_key in user_message and user_message[container_key] is not None:
                nested = extract_message_text(user_message[container_key])
                if nested:
                    return nested
        return ""
    if user_message is None:
        return ""
    return str(user_message).strip()


def discover_all_skill_paths(hermes_home: Path, cwd: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:
    """Compatibility wrapper around Hermes' trusted, active-profile inventory."""
    paths, project_paths, _, _, _ = discover_skill_inventory()
    project_set = set(project_paths)
    return [p for p in paths if p not in project_set], project_paths


def _get_receipt_emitter(ctx: Any = None) -> ReceiptEmitter:
    """Resolve ReceiptEmitter without triggering sidecar skill discovery or sync."""
    global _config, _receipt_emitter
    if ctx is not None:
        cfg = PluginConfig.load_from_ctx(ctx)
        _config = cfg
    elif _config is not None:
        cfg = _config
    else:
        cfg = PluginConfig.load_from_ctx(None)
        _config = cfg

    if _receipt_emitter is None or getattr(_receipt_emitter, "receipts_path", None) != cfg.receipts_path:
        _receipt_emitter = ReceiptEmitter(receipts_path=cfg.receipts_path)
    return _receipt_emitter


def _get_services(ctx: Any = None):
    global _config, _sidecar, _triage, _coordinator, _receipt_emitter
    prev_config = _config
    if ctx is not None:
        new_config = PluginConfig.load_from_ctx(ctx)
        _config = new_config
    else:
        if _config is None:
            _config = PluginConfig.load_from_ctx(None)
        new_config = _config

    sidecar_recreated = False
    if _sidecar is None or (prev_config is not None and new_config.sidecar_db_path != prev_config.sidecar_db_path):
        _sidecar = SidecarIndex(db_path=new_config.sidecar_db_path)
        sidecar_recreated = True

    paths, project_paths, source_info, names, tiers = discover_skill_inventory(
        include_project=new_config.software_inspection_enabled
    )
    _sidecar.sync_skills(
        paths,
        project_local_paths=project_paths,
        source_info=source_info,
        skill_names=names,
        replace_tiers=tiers,
    )

    triage_changed = prev_config is not None and new_config.timeout_ms != prev_config.timeout_ms
    if _triage is None or triage_changed:
        _triage = CognitiveTriage(ctx=ctx, timeout_ms=new_config.timeout_ms)
    elif ctx is not None and hasattr(_triage, "ctx"):
        _triage.ctx = ctx

    coordinator_changed = prev_config is not None and new_config != prev_config
    if _coordinator is None or sidecar_recreated or coordinator_changed:
        _coordinator = ContextCoordinator(config=new_config, sidecar=_sidecar, ctx=ctx)
    elif ctx is not None and hasattr(_coordinator, "ctx"):
        _coordinator.ctx = ctx

    if _receipt_emitter is None or getattr(_receipt_emitter, "receipts_path", None) != new_config.receipts_path:
        _receipt_emitter = ReceiptEmitter(receipts_path=new_config.receipts_path)

    return _config, _sidecar, _triage, _coordinator, _receipt_emitter


def on_pre_llm_call(user_message: Any = None, **kwargs: Any) -> Optional[Dict[str, str]]:
    """
    Hook body called before LLM inference across any platform (CLI, WhatsApp, Telegram, Discord, Desktop).
    Appends ephemeral context to the USER message turn.
    Returns None if nothing fits or on any error (fail-closed).
    """
    global _session_turns, _config
    ctx = kwargs.get("_ctx")
    turn_config = PluginConfig.load_from_ctx(ctx) if ctx is not None else (_config or PluginConfig.load_from_ctx(None))

    if not turn_config.enabled or not turn_config.pre_hook_enabled:
        if ctx is not None:
            _config = turn_config
        return None

    text = extract_message_text(user_message)
    if not text or text.lstrip().startswith("/"):
        return None

    turn_id = str(kwargs.get("turn_id", ""))
    session_id = str(kwargs.get("session_id", ""))
    turn_key = f"{session_id}:{turn_id}"

    platform = str(kwargs.get("platform") or "cli").lower()
    model_name = str(kwargs.get("model") or "").strip()

    try:
        config, sidecar, triage, coordinator, _ = _get_services(ctx)
        cwd = kwargs.get("cwd") or _ctx_cwd(ctx)
        analyze_kwargs: Dict[str, Any] = {
            "software_inspection_enabled": config.software_inspection_enabled
        }
        try:
            import inspect
            sig = inspect.signature(triage.analyze)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if "project_scan_paths" in sig.parameters or has_varkw:
                analyze_kwargs["project_scan_paths"] = config.project_scan_paths
            if "conversation_history" in kwargs and ("conversation_history" in sig.parameters or has_varkw):
                analyze_kwargs["conversation_history"] = kwargs.get("conversation_history")
        except Exception:
            pass

        analysis = triage.analyze(text, cwd=cwd, **analyze_kwargs)
        effective_cwd = (
            analysis.domain.resolved_workspace_path
            if (analysis.domain and analysis.domain.resolved_workspace_path)
            else cwd
        )
        result = coordinator.execute(
            analysis,
            user_message=text,
            cwd=effective_cwd,
            session_id=session_id or None,
            memory_db_uri=config.memory_db_uri,
            extra_memory_dirs=config.extra_memory_dirs,
            excluded_session_sources=config.excluded_session_sources,
            excluded_message_prefixes=config.excluded_message_prefixes,
        )

        # Store session/turn state isolated from concurrent sessions (bounded to 256 entries)
        with _session_turns_lock:
            if len(_session_turns) >= 256:
                oldest_key = next(iter(_session_turns))
                _session_turns.pop(oldest_key, None)
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
    with _session_turns_lock:
        turn_data = _session_turns.pop(turn_key, None)
    if turn_data is None:
        return

    analysis, route_result, platform, model_name = turn_data
    if analysis is None or route_result is None:
        return

    try:
        ctx = kwargs.get("_ctx")
        receipt_emitter = _get_receipt_emitter(ctx)
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
            query = extract_message_text(args.get("query", ""))
            if not query:
                return "Error: query is required."
            turn_config = PluginConfig.load_from_ctx(ctx) if ctx is not None else (_config or PluginConfig.load_from_ctx(None))
            if not turn_config.enabled:
                return "Focus Context Engine Disabled (enabled=false)."

            try:
                cfg, _, triage, coordinator, _ = _get_services(ctx)
                cwd = kw.get("cwd") or _ctx_cwd(ctx)
                tool_kwargs: Dict[str, Any] = {
                    "software_inspection_enabled": cfg.software_inspection_enabled
                }
                history = args.get("conversation_history") if isinstance(args, dict) else None
                if history is None:
                    history = kw.get("conversation_history")
                try:
                    import inspect
                    sig = inspect.signature(triage.analyze)
                    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                    if "project_scan_paths" in sig.parameters or has_varkw:
                        tool_kwargs["project_scan_paths"] = cfg.project_scan_paths
                    if history is not None and ("conversation_history" in sig.parameters or has_varkw):
                        tool_kwargs["conversation_history"] = history
                except Exception:
                    pass

                analysis = triage.analyze(query, cwd=cwd, **tool_kwargs)
                effective_cwd = (
                    analysis.domain.resolved_workspace_path
                    if (analysis.domain and analysis.domain.resolved_workspace_path)
                    else cwd
                )
                result = coordinator.execute(
                    analysis,
                    user_message=query,
                    cwd=effective_cwd,
                    session_id=(args.get("session_id") if isinstance(args, dict) else None) or kw.get("session_id"),
                    memory_db_uri=cfg.memory_db_uri,
                    extra_memory_dirs=cfg.extra_memory_dirs,
                    excluded_session_sources=cfg.excluded_session_sources,
                    excluded_message_prefixes=cfg.excluded_message_prefixes,
                )
                selected = (
                    ", ".join(s.name for s in result.selected_skills)
                    if result.selected_skills
                    else (result.selected_skill.name if result.selected_skill else "None")
                )
                return f"Route Result -> Winner: {selected} | Latency: {result.triage_latency_ms:.1f}ms | Status: {result.status}"
            except Exception as exc:
                logger.warning("focus_route tool error (fail-closed): %s", exc)
                return f"Route Result -> Winner: None | Status: error ({type(exc).__name__})"

        route_schema = {
            "name": "focus_route",
            "description": "Test or manually inspect cognitive triage and skill routing for a given user query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user prompt or task description to evaluate."},
                    "session_id": {"type": "string", "description": "Optional verified session identity for memory scoping."}
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
            return handle_cli(args, cfg, sidecar, receipt_emitter, ctx=ctx)

        ctx.register_cli_command(
            name="focus",
            help="Focus CLI: manage sidecar index, configure settings, verify receipts, and run diagnostics.",
            setup_fn=cli_setup_fn,
            handler_fn=cli_handler_fn,
            description="Focus: manage cognitive triage, configure budgets, inspect receipts, and optimize skill index."
        )

    logger.info("focus v%s registered successfully.", PLUGIN_VERSION)
