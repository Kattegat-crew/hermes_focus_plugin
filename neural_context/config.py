from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


def get_hermes_home() -> Path:
    """Resolve the active Hermes home directory."""
    try:
        from hermes_constants import get_hermes_home as _ghh
        return Path(_ghh())
    except Exception:
        env = os.environ.get("HERMES_HOME")
        return Path(env) if env else (Path.home() / ".hermes")


def get_plugin_data_dir() -> Path:
    """Resolve the canonical Hermes plugin data directory: <hermes_home>/plugin-data/<name>/."""
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir("focus")
    except Exception:
        home = get_hermes_home()
        d = home / "plugin-data" / "focus"
        d.mkdir(parents=True, exist_ok=True)
        return d


def get_hermes_language() -> str:
    """Resolve the active Hermes language setting (secret_scope HERMES_LANGUAGE or display.language)."""
    env_lang: Optional[str] = None
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            env_lang = get_secret("HERMES_LANGUAGE")
        except UnscopedSecretError:
            env_lang = os.environ.get("HERMES_LANGUAGE")
    except ImportError:
        env_lang = os.environ.get("HERMES_LANGUAGE")
    if env_lang and env_lang.strip():
        return env_lang.strip()
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        return str((cfg.get("display") or {}).get("language", "")).strip()
    except Exception:
        return ""


class PluginConfig(BaseModel):
    enabled: bool = True
    pre_hook_enabled: bool = True
    router_backend: str = "graph"
    max_context_tokens: int = 0  # 0 = unlimited by default, preserves full procedural depth
    selective_memory_enabled: bool = True
    memory_token_budget: int = 0  # 0 = unlimited by default, preserves all relevant facts
    memory_db_uri: str = ""
    extra_memory_dirs: List[str] = Field(default_factory=list)
    excluded_session_sources: List[str] = Field(
        default_factory=lambda: ["kanban", "subagent", "tool"]
    )
    excluded_message_prefixes: List[str] = Field(
        default_factory=lambda: ["[CONTEXT COMPACTION", "[CONTEXT SUMMARY]:"]
    )
    software_inspection_enabled: bool = True
    project_scan_paths: List[str] = Field(default_factory=list)
    sidecar_db_path: str = ""
    receipts_path: str = ""
    timeout_ms: int = 0  # 0 = no plugin-imposed deadline; provider limits still apply

    @classmethod
    def load_from_ctx(cls, ctx: Any = None) -> PluginConfig:
        default_inst = cls()
        config_data: Dict[str, Any] = {}

        # Hermes PluginContext.get_config(key: str, default: Any = None)
        if ctx is not None and hasattr(ctx, "get_config"):
            fields = [
                "enabled", "pre_hook_enabled", "router_backend", "max_context_tokens",
                "selective_memory_enabled", "memory_token_budget",
                "memory_db_uri", "extra_memory_dirs", "excluded_session_sources", "excluded_message_prefixes",
                "software_inspection_enabled", "project_scan_paths", "sidecar_db_path", "receipts_path", "timeout_ms",
            ]
            for f in fields:
                default_val = getattr(default_inst, f)
                try:
                    val = ctx.get_config(f, default_val)
                    if val is not None and type(val).__name__ != "MagicMock":
                        config_data[f] = val
                    else:
                        config_data[f] = default_val
                except Exception:
                    config_data[f] = default_val

        # Normalize extra_memory_dirs
        raw_extra_dirs = config_data.get("extra_memory_dirs")
        if isinstance(raw_extra_dirs, str):
            config_data["extra_memory_dirs"] = [
                d.strip() for d in raw_extra_dirs.split(",") if d.strip()
            ]
        elif isinstance(raw_extra_dirs, (list, tuple)):
            config_data["extra_memory_dirs"] = [
                str(d).strip() for d in raw_extra_dirs if str(d).strip()
            ]
        elif raw_extra_dirs is None:
            config_data["extra_memory_dirs"] = []

        # Normalize excluded_session_sources
        raw_sources = config_data.get("excluded_session_sources")
        if isinstance(raw_sources, str):
            config_data["excluded_session_sources"] = [
                s.strip().lower() for s in raw_sources.split(",") if s.strip()
            ]
        elif isinstance(raw_sources, (list, tuple, set)):
            config_data["excluded_session_sources"] = [
                str(s).strip().lower() for s in raw_sources if str(s).strip()
            ]
        elif raw_sources is None:
            config_data["excluded_session_sources"] = list(default_inst.excluded_session_sources)

        # Normalize excluded_message_prefixes (preserve case for prefix matching)
        raw_prefixes = config_data.get("excluded_message_prefixes")
        if isinstance(raw_prefixes, str):
            config_data["excluded_message_prefixes"] = [
                p.strip() for p in raw_prefixes.split(",") if p.strip()
            ]
        elif isinstance(raw_prefixes, (list, tuple, set)):
            config_data["excluded_message_prefixes"] = [
                str(p).strip() for p in raw_prefixes if str(p).strip()
            ]
        elif raw_prefixes is None:
            config_data["excluded_message_prefixes"] = list(default_inst.excluded_message_prefixes)

        # Normalize project_scan_paths
        raw_scan_paths = config_data.get("project_scan_paths")
        if isinstance(raw_scan_paths, str):
            config_data["project_scan_paths"] = [
                d.strip() for d in raw_scan_paths.split(",") if d.strip()
            ]
        elif isinstance(raw_scan_paths, (list, tuple)):
            config_data["project_scan_paths"] = [
                str(d).strip() for d in raw_scan_paths if str(d).strip()
            ]
        elif raw_scan_paths is None:
            config_data["project_scan_paths"] = []

        data_dir = get_plugin_data_dir()

        if not config_data.get("sidecar_db_path"):
            config_data["sidecar_db_path"] = str(data_dir / "sidecar_index.db")
        if not config_data.get("receipts_path"):
            config_data["receipts_path"] = str(data_dir / "receipts.jsonl")

        return cls(**config_data)
