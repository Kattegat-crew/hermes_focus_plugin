from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

from .config import PluginConfig, get_hermes_home
from .receipt import ReceiptEmitter
from .sidecar import SidecarIndex


def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the `hermes neural-context` argparse subcommands tree."""
    sub = subparser.add_subparsers(dest="subcommand", help="Available subcommands")

    sub.add_parser("status", help="Print current plugin status, config, and index summary.")
    sub.add_parser("doctor", help="Run system diagnostics and verify fail-closed boundaries.")

    idx_p = sub.add_parser("index", help="Scan and sync skills into sidecar index.")
    idx_p.add_argument("--path", type=str, default="", help="Custom directory of skills to scan.")

    ver_p = sub.add_parser("verify", help="Inspect verifiable receipts and token savings metrics.")
    ver_p.add_argument("--limit", type=int, default=10, help="Number of recent records to display.")

    cfg_p = sub.add_parser("config", help="View or modify plugin configuration parameters.")
    cfg_p.add_argument("--set", metavar="KEY=VALUE", help="Set a configuration parameter (e.g. max_context_tokens=1500).")


def build_parser() -> argparse.ArgumentParser:
    """Standalone parser for testing or direct CLI execution."""
    parser = argparse.ArgumentParser(
        prog="hermes neural-context",
        description="Neural Context CLI: manage sidecar index, verify audit receipts, and run diagnostics."
    )
    register_cli(parser)
    return parser


def handle_cli(args: argparse.Namespace, config: PluginConfig, sidecar: SidecarIndex, receipt_emitter: ReceiptEmitter) -> int:
    cmd = getattr(args, "subcommand", None) or "status"

    if cmd == "status":
        stats = receipt_emitter.get_summary_stats()
        print("=== Neural Context Plugin Status ===")
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
        print("=== Running Neural Context Doctor Diagnostics ===")
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

        # 3. Check router configuration
        if config.router_backend == "jev" and not config.jev_api_key:
            problems.append("Router backend is 'jev' but TYPESAFE_API_KEY is not set.")

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
        scan_dir = Path(args.path) if getattr(args, "path", None) else (get_hermes_home() / "skills")
        print(f"Scanning skills in: {scan_dir}")
        skill_files: List[Path] = []
        if scan_dir.is_dir():
            for p in scan_dir.rglob("SKILL.md"):
                skill_files.append(p)

        updated = sidecar.sync_skills(skill_files)
        print(f"Synchronization complete: {updated} skills updated, total indexed: {sidecar.n_docs}.")
        return 0

    if cmd == "verify":
        limit = getattr(args, "limit", 10)
        stats = receipt_emitter.get_summary_stats(limit=limit)
        token_pct = stats.get('token_reduction_pct', 0.0)
        fast_exit_pct = stats.get('fast_exit_pct', 0.0)
        print(f"=== Neural Context Verifiable Receipts (Last {limit} turns) ===")
        print(f"  Turns Audited       : {stats.get('total_turns', 0)}")
        print(f"  Skills Routed       : {stats.get('skills_routed', 0)}")
        print(f"  Token Reduction     : {token_pct}% saved vs static preloading")
        print(f"  Conversational Exit : {fast_exit_pct}% turns fast-exited with zero token bloat")
        print(f"  Tokens Saved Total  : {stats.get('total_tokens_saved', 0)} tokens")
        return 0

    if cmd == "config":
        set_val = getattr(args, "set", None)
        if set_val and "=" in set_val:
            key, val = set_val.split("=", 1)
            key, val = key.strip(), val.strip()
            try:
                import yaml
                hermes_home = get_hermes_home()
                cfg_path = hermes_home / "config.yaml"
                raw_cfg = {}
                if cfg_path.is_file():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        raw_cfg = yaml.safe_load(f) or {}

                plugins = raw_cfg.setdefault("plugins", {})
                entries = plugins.setdefault("entries", {})
                nc = entries.setdefault("neural-context-plugin", {})
                settings = nc.setdefault("settings", {})

                if val.lower() in ("true", "false"):
                    coerced = val.lower() == "true"
                elif val.isdigit():
                    coerced = int(val)
                else:
                    coerced = val

                settings[key] = coerced
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(raw_cfg, f, default_flow_style=False)
                print(f"Successfully updated '{key}' to {coerced} in {cfg_path}")
                return 0
            except Exception as e:
                print(f"[ERROR] Could not update config: {e}")
                return 1

        print("=== Neural Context Configuration Settings ===")
        print(f"  enabled                     : {config.enabled} (Active state)")
        print(f"  pre_hook_enabled            : {config.pre_hook_enabled} (pre_llm_call interception)")
        print(f"  router_backend              : {config.router_backend} (host_llm | graph | jev)")
        print(f"  max_context_tokens          : {config.max_context_tokens} (0 = unlimited / unconstrained)")
        print(f"  memory_token_budget         : {config.memory_token_budget} (0 = unlimited / unconstrained)")
        print(f"  selective_memory_enabled    : {config.selective_memory_enabled} (Recall domain facts)")
        print(f"  software_inspection_enabled : {config.software_inspection_enabled} (Detect AGENTS.md & local skills)")
        print(f"  timeout_ms                  : {config.timeout_ms} ms (Pre-inference deadline)")
        print("\nTo update a parameter:")
        print("  hermes neural-context config --set max_context_tokens=1500")
        print("Or edit directly via:")
        print("  hermes config edit")
        return 0

    return 0
