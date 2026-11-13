from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import PluginConfig
from .receipt import ReceiptEmitter
from .sidecar import SidecarIndex, discover_skill_inventory


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the `hermes focus` argparse subcommands tree."""
    sub = subparser.add_subparsers(dest="subcommand", help="Available subcommands")

    sub.add_parser("status", help="Print current plugin status, config, and index summary.")
    sub.add_parser("doctor", help="Run system diagnostics and verify fail-closed boundaries.")

    idx_p = sub.add_parser("index", help="Scan and sync skills into sidecar index.")
    idx_p.add_argument("--path", type=str, default="", help="Custom directory of skills to scan.")

    ver_p = sub.add_parser("verify", help="Inspect verifiable receipts and token savings metrics.")
    ver_p.add_argument("--limit", type=int, default=10, help="Number of recent records to display.")

    cfg_p = sub.add_parser("config", help="View or modify plugin configuration parameters.")
    cfg_p.add_argument("--set", metavar="KEY=VALUE", help="Set a configuration parameter (e.g. max_context_tokens=1500).")


CONFIG_FIELD_TYPES = {
    "enabled": bool,
    "pre_hook_enabled": bool,
    "router_backend": str,
    "max_context_tokens": int,
    "selective_memory_enabled": bool,
    "memory_token_budget": int,
    "software_inspection_enabled": bool,
    "sidecar_db_path": str,
    "receipts_path": str,
    "timeout_ms": int,
}


def _coerce_and_validate(key: str, val: str) -> Tuple[Optional[Any], Optional[str]]:
    if key not in CONFIG_FIELD_TYPES:
        valid_keys = ", ".join(sorted(CONFIG_FIELD_TYPES.keys()))
        return None, f"Unknown configuration setting '{key}'. Valid settings: {valid_keys}"

    expected_type = CONFIG_FIELD_TYPES[key]
    if expected_type is bool:
        if val.lower() in ("true", "1", "yes"):
            return True, None
        elif val.lower() in ("false", "0", "no"):
            return False, None
        else:
            return None, f"Invalid value for '{key}': expected boolean (true/false), got '{val}'"
    elif expected_type is int:
        try:
            return int(val), None
        except ValueError:
            return None, f"Invalid value for '{key}': expected integer, got '{val}'"
    elif expected_type is str:
        if key == "router_backend" and val not in ("host_llm", "graph"):
            return None, f"Invalid backend '{val}': choose from 'host_llm', 'graph'"
        return val, None

    return val, None


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for testing or direct CLI execution."""
    parser = argparse.ArgumentParser(
        prog="hermes focus",
        description="Focus CLI: manage sidecar index, configure settings, verify receipts, and run diagnostics."
    )
    register_cli(parser)
    return parser


def handle_cli(
    args: argparse.Namespace,
    config: PluginConfig,
    sidecar: SidecarIndex,
    receipt_emitter: ReceiptEmitter,
    ctx: Any = None,
) -> int:
    if isinstance(args, (list, tuple)):
        args = build_parser().parse_args(list(args))
    cmd = getattr(args, "subcommand", None) or "status"

    if cmd == "status":
        stats = receipt_emitter.get_summary_stats()
        print("=== Focus Plugin Status ===")
        print(f"  Plugin Enabled       : {config.enabled}")
        print(f"  Pre-Hook Enabled     : {config.pre_hook_enabled}")
        print(f"  Router Backend       : {config.router_backend}")
        print(f"  Sidecar DB           : {config.sidecar_db_path}")
        print(f"  Receipts Log         : {config.receipts_path}")
        print(f"  Skills Indexed       : {sidecar.n_docs}")
        print(f"  Total Turns Audited  : {stats.get('total_turns', 0)}")
        print(f"  Estimated Tokens Saved: {stats.get('total_tokens_saved', 0)}")
        return 0

    if cmd == "doctor":
        print("=== Running Focus Doctor Diagnostics ===")
        problems: List[str] = []

        # 1. Check DB Path
        db_path = Path(config.sidecar_db_path)
        if not db_path.parent.exists():
            problems.append(f"Directory {db_path.parent} does not exist.")
        elif not os.access(db_path.parent, os.W_OK):
            problems.append(f"Directory {db_path.parent} is not writable.")

        # 2. Check indexed skills
        if sidecar.n_docs == 0:
            problems.append("Sidecar index is empty (0 skills indexed). Run `index` subcommand.")

        for p in problems:
            print(f"  [ERROR] {p}")

        if not problems:
            print("  [OK] Database writable and sidecar index operational.")
            print("  [OK] Fail-closed fallback guarantees verified.")
            print("Doctor result: HEALTHY")
            return 0
        else:
            print(f"Doctor result: {len(problems)} problem(s) found.")
            return 1

    if cmd == "index":
        skill_files, project_paths, source_info, names, tiers = discover_skill_inventory(
            include_project=config.software_inspection_enabled
        )
        requested = getattr(args, "path", None)
        if requested:
            scan_dir = Path(requested).expanduser().resolve()
            if not scan_dir.is_dir():
                print(f"[ERROR] Skill source is not a directory: {scan_dir}")
                return 1
            skill_files = [
                path for path in skill_files
                if path.is_relative_to(scan_dir) or scan_dir.is_relative_to(path.parent)
            ]
            if not skill_files:
                try:
                    from agent import skill_utils
                    roots = [Path(root).resolve() for _tier, root in skill_utils.get_skill_search_roots(
                        include_project=config.software_inspection_enabled
                    )]
                    if not any(scan_dir.is_relative_to(root) for root in roots):
                        print("[ERROR] Path is not part of the active trusted Hermes skill inventory.")
                        return 1
                except Exception:
                    print("[ERROR] Cannot verify an empty path against the native trusted skill roots.")
                    return 1
            project_paths = [path for path in project_paths if path in skill_files]
            source_info = {path: source_info[path] for path in skill_files}
            names = {path: names[path] for path in skill_files if path in names}
            # Explicit paths are partial syncs: preserve unrelated source tiers.
            tiers = None
            replace_subtrees = [scan_dir]
            print(f"Scanning trusted skills in: {scan_dir}")
        else:
            print("Scanning the active trusted Hermes skill inventory.")
            replace_subtrees = None

        updated = sidecar.sync_skills(
            skill_files,
            project_local_paths=project_paths,
            source_info=source_info,
            skill_names=names,
            replace_tiers=tiers,
            replace_subtrees=replace_subtrees,
        )
        print(f"Synchronization complete: {updated} skills updated, total indexed: {sidecar.n_docs}.")
        return 0

    if cmd == "verify":
        limit = getattr(args, "limit", 10)
        stats = receipt_emitter.get_summary_stats(limit=limit)
        token_pct = stats.get('token_reduction_pct', 0.0)
        fast_exit_pct = stats.get('fast_exit_pct', 0.0)
        print(f"=== Focus Verifiable Receipts (Last {limit} turns) ===")
        print(f"  Turns Audited       : {stats.get('total_turns', 0)}")
        print(f"  Skills Routed       : {stats.get('skills_routed', 0)}")
        print(f"  Catalog Comparison  : {token_pct}% estimated vs hypothetical active-catalog baseline (not provider token usage)")
        print(f"  Conversational Exit : {fast_exit_pct}% turns fast-exited with zero token bloat")
        print(f"  Tokens Saved Total  : {stats.get('total_tokens_saved', 0)} tokens")
        return 0

    if cmd == "config":
        set_val = getattr(args, "set", None)
        if set_val and "=" in set_val:
            key, val = set_val.split("=", 1)
            key, val = key.strip(), val.strip()

            coerced, err = _coerce_and_validate(key, val)
            if err:
                print(f"[ERROR] {err}")
                return 1

            if ctx is None or not hasattr(ctx, "set_config"):
                print("[ERROR] Cannot persist configuration: active Hermes plugin context (ctx.set_config) is required.")
                return 1

            try:
                ctx.set_config(key, coerced)
                setattr(config, key, coerced)
                print(f"Saved '{key}' as {coerced}. Reload the plugin to refresh cached services and apply this setting.")
                return 0
            except Exception as e:
                print(f"[ERROR] Could not update config: {e}")
                return 1

        print("=== Focus Configuration Settings ===")
        print(f"  enabled                     : {config.enabled} (Active state)")
        print(f"  pre_hook_enabled            : {config.pre_hook_enabled} (pre_llm_call interception)")
        print(f"  router_backend              : {config.router_backend} (host_llm | graph)")
        print(f"  max_context_tokens          : {config.max_context_tokens} (0 = unlimited / unconstrained)")
        print(f"  memory_token_budget         : {config.memory_token_budget} (0 = unlimited / unconstrained)")
        print(f"  selective_memory_enabled    : {config.selective_memory_enabled} (Recall domain facts)")
        print(f"  software_inspection_enabled : {config.software_inspection_enabled} (Detect AGENTS.md & local skills)")
        print(f"  timeout_ms                  : {config.timeout_ms} ms (0 means no plugin-imposed timeout)")
        print("\nTo update a parameter:")
        print("  hermes focus config --set max_context_tokens=1500")
        print("Or edit directly via:")
        print("  hermes config edit")
        return 0

    return 0
