from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


def get_hermes_home() -> Path:
    """Resolve the active Hermes home directory."""
    try:
        from hermes_constants import get_hermes_home as _ghh
        return Path(_ghh())
    except Exception:
        env = os.environ.get("HERMES_HOME")
        return Path(env) if env else (Path.home() / ".hermes")


class PluginConfig(BaseModel):
    enabled: bool = True
    pre_hook_enabled: bool = True
    router_backend: str = "host_llm"
    max_context_tokens: int = 1200
    selective_memory_enabled: bool = True
    memory_token_budget: int = 400
    software_inspection_enabled: bool = True
    sidecar_db_path: str = ""
    receipts_path: str = ""
    timeout_ms: int = 250
    jev_api_key: Optional[str] = None
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"

    @classmethod
    def load_from_ctx(cls, ctx: Any = None) -> PluginConfig:
        default_inst = cls()
        config_data: Dict[str, Any] = {}

        # Hermes PluginContext.get_config(key: str, default: Any = None)
        if ctx is not None and hasattr(ctx, "get_config"):
            fields = [
                "enabled", "pre_hook_enabled", "router_backend", "max_context_tokens",
                "selective_memory_enabled", "memory_token_budget", "software_inspection_enabled",
                "sidecar_db_path", "receipts_path", "timeout_ms", "jev_api_key",
                "jev_base_url", "jev_model"
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

        # Environment variable overrides
        jev_key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("NEURAL_CONTEXT_JEV_KEY")
        if jev_key:
            config_data["jev_api_key"] = jev_key

        home = get_hermes_home()
        data_dir = home / "plugins" / "neural-context-plugin"
        data_dir.mkdir(parents=True, exist_ok=True)

        if not config_data.get("sidecar_db_path"):
            config_data["sidecar_db_path"] = str(data_dir / "sidecar_index.db")
        if not config_data.get("receipts_path"):
            config_data["receipts_path"] = str(data_dir / "receipts.jsonl")

        return cls(**config_data)
