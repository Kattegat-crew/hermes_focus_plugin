from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import get_hermes_home
from .models import MemoryItem, MemoryProbe
from .sidecar import normalize_text_for_matching, term_information_weight, tokenize


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    # Count dense / East Asian / non-Latin characters where each char is ~1-2 tokens
    wide_or_cjk = sum(1 for c in text if ord(c) > 0x2E80)
    return max(words + (len(text) // 4), words + wide_or_cjk)


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(all\s+)?prev[i]ous\s+instructions|<system>|system:|assistant:|human:|<script|override\s+instructions)",
    re.IGNORECASE
)

_FORBIDDEN_PREFIXES = (
    Path("/etc"),
    Path("/proc"),
    Path("/sys"),
    Path("/dev"),
    Path("/var/run"),
    Path("/root/.ssh"),
)


def is_safe_memory_file(path: Path) -> bool:
    """Validate path safety against symlink traversal and denial of service."""
    try:
        resolved = path.resolve()
        for forb in _FORBIDDEN_PREFIXES:
            try:
                if resolved == forb or forb in resolved.parents:
                    return False
            except Exception:
                pass
        # Text files are read in full, while SQLite stores are queried read-only.
        # A database file size is not the amount of data materialized by a query.
        if (
            resolved.is_file()
            and resolved.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}
            and resolved.stat().st_size > 2_000_000
        ):
            return False
        return resolved.is_file()
    except Exception:
        return False


class _MemoryFactNode:
    __slots__ = (
        "fact",
        "norm_fact",
        "tokens",
        "section_entity",
        "source_file",
        "token_estimate",
        "timestamp_epoch",
    )

    def __init__(
        self,
        fact: str,
        norm_fact: str,
        tokens: Set[str],
        section_entity: Optional[str],
        source_file: str,
        token_estimate: int,
        timestamp_epoch: Optional[float] = None,
    ) -> None:
        self.fact = fact
        self.norm_fact = norm_fact
        self.tokens = tokens
        self.section_entity = section_entity
        self.source_file = source_file
        self.token_estimate = token_estimate
        self.timestamp_epoch = timestamp_epoch


# Per-file in-memory graph node cache: resolved_path -> (mtime_ns, file_size, List[_MemoryFactNode])
_MEMORY_FILE_CACHE: Dict[str, Tuple[int, int, List[_MemoryFactNode]]] = {}


def _load_file_fact_nodes(fpath: Path) -> List[_MemoryFactNode]:
    """
    Parse a Markdown memory file into structured Fact Nodes with section heading inheritance,
    caching the parsed nodes in memory keyed by (mtime_ns, size) to avoid line-by-line disk scans.
    """
    try:
        resolved = str(fpath.resolve())
        stat = fpath.stat()
        mtime_ns = stat.st_mtime_ns
        fsize = stat.st_size
    except Exception:
        return []

    cached = _MEMORY_FILE_CACHE.get(resolved)
    if cached is not None and cached[0] == mtime_ns and cached[1] == fsize:
        return cached[2]

    try:
        content = fpath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    nodes: List[_MemoryFactNode] = []
    current_heading: Optional[str] = None
    current_level: int = 0

    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        if trimmed.startswith("#"):
            hashes = len(trimmed) - len(trimmed.lstrip("#"))
            heading_raw = trimmed.lstrip("#").strip()
            if heading_raw:
                current_heading = heading_raw
                current_level = hashes
            elif current_heading and hashes <= current_level:
                current_heading = None
                current_level = 0
            continue

        if len(trimmed) < 8 or _INJECTION_PATTERNS.search(trimmed):
            continue

        norm_line = normalize_text_for_matching(trimmed)
        line_toks = set(tokenize(trimmed))
        if not line_toks:
            continue

        nodes.append(
            _MemoryFactNode(
                fact=trimmed,
                norm_fact=norm_line,
                tokens=line_toks,
                section_entity=current_heading,
                source_file=fpath.name,
                token_estimate=estimate_tokens(trimmed),
            )
        )

    if len(_MEMORY_FILE_CACHE) >= 128:
        _MEMORY_FILE_CACHE.clear()
    _MEMORY_FILE_CACHE[resolved] = (mtime_ns, fsize, nodes)
    return nodes


_NUMERIC_OR_DATE_RE = re.compile(
    r"^(?:[-+]?\d+(?:\.\d+)?|\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)$"
)
_HEX_OR_UUID_RE = re.compile(
    r"^(?:[0-9a-f]{16,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
_FTS_VIRTUAL_RE = re.compile(r"\bCREATE\s+VIRTUAL\s+TABLE\b.*?\bUSING\s+fts\w*\b", re.IGNORECASE | re.DOTALL)


def _detect_fts_tables_and_shadows(table_sql_rows: List[Tuple[str, Optional[str]]]) -> Tuple[Dict[str, str], Set[str]]:
    """
    Directly inspect sqlite_master DDL statements to identify FTS virtual tables
    and any engine-managed shadow tables prefixed by `<fts_table>_` (zero hardcoded suffix lists).
    Returns `(fts_virtual_lower_map, shadow_table_names_lower)`.
    """
    fts_tables: Dict[str, str] = {}
    for tbl_name, ddl in table_sql_rows:
        if ddl and _FTS_VIRTUAL_RE.search(ddl):
            fts_tables[tbl_name.lower()] = tbl_name

    fts_bases = tuple(f"{base}_" for base in fts_tables)
    shadows: Set[str] = {
        tbl_name.lower()
        for tbl_name, _ in table_sql_rows
        if fts_bases and tbl_name.lower().startswith(fts_bases)
    }
    return fts_tables, shadows


def _parse_epoch_seconds(value: Any) -> Optional[float]:
    """
    Parse numeric Unix epochs (seconds or milliseconds) or ISO-8601 date/datetime strings
    into UTC epoch seconds cleanly, rejecting IDs, booleans, and out-of-range integers.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if 946_684_800.0 <= num <= 4_102_444_800.0:
            return num
        if 946_684_800_000.0 <= num <= 4_102_444_800_000.0:
            return num / 1000.0
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            num = float(s)
            return _parse_epoch_seconds(num)
        except ValueError:
            pass
        if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
            try:
                iso = s.replace(" ", "T", 1) if len(s) > 10 and s[10] == " " else s
                if iso.endswith("Z"):
                    iso = iso[:-1] + "+00:00"
                dt = datetime.fromisoformat(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                return None
    return None


def _profile_columns_from_sample(
    col_names: List[str],
    col_types: List[str],
    sample_rows: List[Tuple[Any, ...]],
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Language-agnostic direct statistical profiling of table columns from a bounded row sample.
    Identifies natural-language content columns, the best entity/grouping column, and any temporal
    timestamp column directly from column types and actual data distributions (no static name dictionaries).
    Returns `(content_cols, entity_col, timestamp_col)`.
    """
    if not sample_rows:
        return [], None, None

    prose_candidates: List[Tuple[float, str]] = []
    entity_candidates: List[Tuple[float, str]] = []
    ts_candidates: List[Tuple[float, str]] = []

    for idx, col_name in enumerate(col_names):
        c_type = (col_types[idx] or "").upper()
        if "BOOL" in c_type or "BLOB" in c_type or "BYTE" in c_type:
            continue

        raw_non_null = [row[idx] for row in sample_rows if idx < len(row) and row[idx] is not None]
        if not raw_non_null:
            continue

        # Direct temporal value profiling (handles UNIX epochs AND ISO strings regardless of column name)
        parsed_ts = [_parse_epoch_seconds(v) for v in raw_non_null]
        ts_hits = sum(1 for t in parsed_ts if t is not None)
        if ts_hits >= max(1, len(raw_non_null) * 0.5):
            type_bonus = 20.0 if any(tk in c_type for tk in ("TIME", "DATE", "REAL", "FLOAT", "DOUBLE")) else 0.0
            ts_candidates.append(((ts_hits / len(raw_non_null)) * 100.0 + type_bonus, col_name))
            continue

        if any(num_t in c_type for num_t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUM")):
            continue

        values: List[str] = []
        for v in raw_non_null:
            s = " ".join(str(v).split()).strip()
            if s:
                values.append(s)

        if not values:
            continue

        # Reject numeric, ISO-timestamp, UUID, or hex-hash identifier columns
        non_id_values = [
            v for v in values
            if not _NUMERIC_OR_DATE_RE.match(v) and not _HEX_OR_UUID_RE.match(v)
        ]
        if len(non_id_values) < max(1, len(values) * 0.5):
            continue

        avg_len = sum(len(v) for v in non_id_values) / len(non_id_values)
        space_ratio = sum(1 for v in non_id_values if " " in v) / len(non_id_values)
        # Distinguish multi-word natural-language prose from short labels or codes
        if (avg_len >= 16.0 and space_ratio >= 0.4) or (avg_len >= 28.0):
            prose_candidates.append((avg_len, col_name))
        elif 2.0 <= avg_len <= 60.0:
            # Prefer concise, human-readable entity/grouping labels
            alpha_ratio = sum(sum(1 for ch in v if ch.isalpha()) / max(len(v), 1) for v in non_id_values) / len(non_id_values)
            if alpha_ratio >= 0.5:
                entity_candidates.append((alpha_ratio * 100.0 - abs(avg_len - 16.0), col_name))

    prose_candidates.sort(key=lambda x: -x[0])
    content_cols = [name for _, name in prose_candidates[:3]]

    entity_col: Optional[str] = None
    if entity_candidates:
        entity_candidates.sort(key=lambda x: -x[0])
        for _, cand_name in entity_candidates:
            if cand_name not in content_cols:
                entity_col = cand_name
                break

    ts_col: Optional[str] = None
    if ts_candidates:
        ts_candidates.sort(key=lambda x: -x[0])
        for _, cand_ts in ts_candidates:
            if cand_ts not in content_cols and cand_ts != entity_col:
                ts_col = cand_ts
                break

    return content_cols, entity_col, ts_col


def _escape_sql_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SelectiveMemoryRetriever:
    """
    Hexagonal Bipartite Concept-Entity-Fact Graph Memory Retriever.
    - Leaves default Hermes system-prompt files (<hermes_home>/MEMORY.md, USER.md, SOUL.md, AGENTS.md)
      untouched so they are never duplicated in the context window.
    - Connects natively to Hermes `state.db` (session & message history), `engram.db` (observations),
      and dynamically introspects any custom SQLite or external SQL database in the user's ecosystem.
    - Passes all retrieved database records and project archive facts through the Bipartite Graph
      (IDF + morphological density + spreading activation) so only relevant historical facts are injected.
    """

    def __init__(self, token_budget: int = 0):
        self.token_budget = token_budget  # 0 = unlimited by default, preserves full context

    def retrieve(
        self,
        entities: Optional[List[str]] = None,
        probe: Optional[MemoryProbe] = None,
        cwd: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        clean_entities = [e.strip() for e in (entities or []) if len(e.strip()) >= 2]
        if probe and probe.entity and probe.entity.strip() not in clean_entities:
            clean_entities.append(probe.entity.strip())

        # Collect native + cross-lingual topic keywords using universal Unicode tokenization
        native_topic_terms: List[str] = []
        xling_topic_terms: List[str] = []
        if probe:
            for text_source in [probe.topic, probe.query_text]:
                if text_source:
                    for t_low in tokenize(text_source):
                        if len(t_low) >= 2 and t_low not in native_topic_terms:
                            native_topic_terms.append(t_low)
            for xl_source in (probe.cross_lingual_terms or []):
                if xl_source:
                    for t_low in tokenize(xl_source):
                        if (
                            len(t_low) >= 2
                            and t_low not in native_topic_terms
                            and t_low not in xling_topic_terms
                        ):
                            xling_topic_terms.append(t_low)

        topic_terms: List[str] = native_topic_terms + xling_topic_terms

        if not clean_entities and not topic_terms:
            return []

        entity_matchers = [
            (e, normalize_text_for_matching(e), set(tokenize(e)))
            for e in clean_entities
        ]
        entity_token_union: Set[str] = set()
        for _, _, etoks in entity_matchers:
            entity_token_union.update(etoks)

        # Specific topic concepts beyond the entity identifier itself
        specific_native_terms = [t for t in native_topic_terms if t not in entity_token_union]
        specific_xling_terms = [t for t in xling_topic_terms if t not in entity_token_union]
        specific_topic_terms = specific_native_terms + specific_xling_terms

        # -------------------------------------------------------------
        # 1. Discover Candidate Stores (Databases + Non-Default Project Archives)
        # -------------------------------------------------------------
        candidate_files: List[Path] = []
        sqlite_candidates: List[Path] = []
        external_db_uris: List[str] = []

        hermes_home = get_hermes_home()
        hermes_home_resolved = hermes_home.resolve() if hermes_home.exists() else hermes_home

        # Tier 1: Project / Workspace Local Stores (excluding Hermes default system-prompt files)
        if cwd:
            cwd_p = Path(cwd)
            if cwd_p.is_dir():
                cwd_resolved = cwd_p.resolve()
                md_candidates = ["PROJECT_MEMORY.md", ".agents/PROJECT_MEMORY.md", ".agents/MEMORY.md"]
                # Only read cwd/MEMORY.md if cwd is a project workspace distinct from hermes_home
                if cwd_resolved != hermes_home_resolved:
                    md_candidates.append("MEMORY.md")

                for fname in md_candidates:
                    fp = cwd_p / fname
                    if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                        candidate_files.append(fp)

                for sub in ["memories", "memory"]:
                    sub_dir = cwd_p / sub
                    if sub_dir.is_dir():
                        for fp in sub_dir.glob("*.md"):
                            if fp.name.upper() in ("USER.MD", "SOUL.MD", "AGENTS.MD"):
                                continue
                            if cwd_resolved == hermes_home_resolved and fp.name.upper() == "MEMORY.MD":
                                continue
                            if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                                candidate_files.append(fp)

                # Discover any SQLite database in the project workspace or data subdirectories
                for pattern in ["*.db", "*.sqlite", "*.sqlite3", "memories/*.db", "memory/*.db", "data/*.db"]:
                    for dbp in cwd_p.glob(pattern):
                        if "sidecar" in dbp.name.lower() or "receipt" in dbp.name.lower():
                            continue
                        if dbp.is_file() and is_safe_memory_file(dbp) and dbp not in sqlite_candidates:
                            sqlite_candidates.append(dbp)

        # Tier 2: Hermes Agent Native History Databases & Archive Subdirectories
        # NOTE: <hermes_home>/MEMORY.md, USER.md, SOUL.md, and AGENTS.md are intentionally NOT read here
        # because Hermes loads them in full into the base system prompt by default.
        if hermes_home.is_dir():
            db_search_paths = [
                hermes_home / "state.db",
                hermes_home / "memories.db",
                hermes_home / "memory.db",
                hermes_home / "engram.db",
                hermes_home / ".engram" / "engram.db",
            ]
            for dbp in db_search_paths:
                if dbp.is_file() and is_safe_memory_file(dbp) and dbp not in sqlite_candidates:
                    sqlite_candidates.append(dbp)

            # Archive subdirectories (<hermes_home>/memories/**/*.md) are NOT loaded into default prompt
            for sub in ["memories", "memory"]:
                mem_dir = hermes_home / sub
                if mem_dir.is_dir():
                    for fp in mem_dir.glob("**/*.md"):
                        if fp.name.upper() in ("MEMORY.MD", "USER.MD", "SOUL.MD", "AGENTS.MD"):
                            continue
                        if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                            candidate_files.append(fp)
                            if len(candidate_files) >= 50:
                                break

        # Tier 3: Custom Environment Memory Databases, URIs, or Archive Directories
        for env_var in ("HERMES_MEMORY_DB", "FOCUS_MEMORY_DB", "HERMES_MEMORY_DIR", "HERMES_MEMORY_PATH"):
            custom_val = os.environ.get(env_var, "").strip()
            if not custom_val:
                continue
            if "://" in custom_val:
                if custom_val not in external_db_uris:
                    external_db_uris.append(custom_val)
                continue
            cmd_p = Path(custom_val)
            if cmd_p.is_file() and is_safe_memory_file(cmd_p):
                if cmd_p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
                    if cmd_p not in sqlite_candidates:
                        sqlite_candidates.append(cmd_p)
                elif cmd_p.suffix.lower() == ".md" and cmd_p not in candidate_files:
                    candidate_files.append(cmd_p)
            elif cmd_p.is_dir():
                for dbp in cmd_p.glob("**/*.db"):
                    if "sidecar" not in dbp.name.lower() and dbp.is_file() and is_safe_memory_file(dbp):
                        if dbp not in sqlite_candidates:
                            sqlite_candidates.append(dbp)
                for fp in cmd_p.glob("**/*.md"):
                    if fp.name.upper() in ("USER.MD", "SOUL.MD", "AGENTS.MD"):
                        continue
                    if fp.is_file() and is_safe_memory_file(fp) and fp not in candidate_files:
                        candidate_files.append(fp)
                        if len(candidate_files) >= 50:
                            break

        # -------------------------------------------------------------
        # 2. Build Unified Bipartite Concept-Entity-Fact Graph
        #    (Combining SQLite DB Nodes + External DB Nodes + Project Archive Nodes)
        # -------------------------------------------------------------
        all_nodes: List[_MemoryFactNode] = []
        concept_to_nodes: Dict[str, List[int]] = {}

        def _register_node(node: _MemoryFactNode) -> None:
            idx = len(all_nodes)
            all_nodes.append(node)
            for tok in node.tokens:
                if tok not in concept_to_nodes:
                    concept_to_nodes[tok] = []
                concept_to_nodes[tok].append(idx)

        for fpath in candidate_files:
            for node in _load_file_fact_nodes(fpath):
                _register_node(node)

        for db_path in sqlite_candidates:
            for node in self._extract_sqlite_fact_nodes(
                db_path,
                clean_entities=clean_entities,
                topic_terms=topic_terms,
                specific_topic_terms=specific_topic_terms,
                active_session_id=session_id,
            ):
                _register_node(node)

        for db_uri in external_db_uris:
            for node in self._extract_external_uri_fact_nodes(
                db_uri,
                clean_entities=clean_entities,
                topic_terms=topic_terms,
            ):
                _register_node(node)

        if not all_nodes:
            return []

        total_fact_nodes = len(all_nodes)

        def _mem_term_idf(tok: str) -> float:
            df = len(concept_to_nodes.get(tok, ()))
            return term_information_weight(tok, df, max(total_fact_nodes, 1))

        def _temporal_multiplier(
            node_ts: Optional[float],
            now_ts: float,
            window_hours: Optional[int],
            recency_bias: bool,
        ) -> Tuple[float, Optional[bool]]:
            if node_ts is None:
                return (0.85 if window_hours else 1.0), None
            age_hours = max(0.0, (now_ts - node_ts) / 3600.0)
            if window_hours is not None and window_hours > 0:
                if age_hours <= (window_hours * 1.1):
                    return 2.2, True
                over = age_hours - window_hours
                decay = max(0.05, 0.30 * math.exp(-over / max(float(window_hours), 24.0)))
                return decay, False
            if recency_bias:
                return 1.0 + 0.65 / (1.0 + (age_hours / 168.0)), None
            return 1.0, None

        # -------------------------------------------------------------
        # 3. Iterative Bipartite Personalized PageRank (Concept <-> Fact)
        #    Propagates: Seed Concepts -> Facts -> Discriminative Bridge Concepts -> Related Facts
        # -------------------------------------------------------------
        seed_concepts: Dict[str, float] = {}
        for tok in entity_token_union:
            w = 3.0 * _mem_term_idf(tok)
            if w > 0.0:
                seed_concepts[tok] = seed_concepts.get(tok, 0.0) + w
        seed_native = specific_native_terms if entity_matchers else native_topic_terms
        for tok in seed_native:
            w = 2.5 * _mem_term_idf(tok)
            if w > 0.0:
                seed_concepts[tok] = seed_concepts.get(tok, 0.0) + w
        seed_xling = specific_xling_terms if entity_matchers else xling_topic_terms
        for tok in seed_xling:
            w = 2.2 * _mem_term_idf(tok)
            if w > 0.0:
                seed_concepts[tok] = seed_concepts.get(tok, 0.0) + w

        seed_sum = sum(seed_concepts.values()) or 1.0
        p0_concepts: Dict[str, float] = {k: v / seed_sum for k, v in seed_concepts.items()}

        # Direct entity / native / cross-lingual signals per Fact Node
        node_matched_ent: List[Optional[str]] = [None] * total_fact_nodes
        node_ent_score: List[float] = [0.0] * total_fact_nodes
        node_native_idf: List[float] = [0.0] * total_fact_nodes
        node_xling_idf: List[float] = [0.0] * total_fact_nodes
        node_direct_topic_idf: List[float] = [0.0] * total_fact_nodes
        node_seed_eligible: List[bool] = [False] * total_fact_nodes

        eval_native = specific_native_terms if entity_matchers else native_topic_terms
        eval_xling = specific_xling_terms if entity_matchers else xling_topic_terms
        total_native_idf = sum(_mem_term_idf(t) for t in native_topic_terms) or 1.0
        total_xling_idf = sum(_mem_term_idf(t) for t in xling_topic_terms) or 1.0

        for idx, node in enumerate(all_nodes):
            norm_sec = normalize_text_for_matching(node.section_entity or "")
            if entity_matchers:
                for ent_name, norm_ent, _ in entity_matchers:
                    if norm_ent and norm_ent in node.norm_fact:
                        node_matched_ent[idx] = ent_name
                        node_ent_score[idx] = 1.5
                        break
                    if norm_ent and norm_sec and norm_ent in norm_sec:
                        node_matched_ent[idx] = ent_name
                        node_ent_score[idx] = 1.0
                        break

            n_idf = sum(
                _mem_term_idf(t)
                for t in eval_native
                if (t in node.tokens or t in node.norm_fact or (norm_sec and t in norm_sec))
            )
            x_idf = sum(
                _mem_term_idf(t)
                for t in eval_xling
                if (t in node.tokens or t in node.norm_fact or (norm_sec and t in norm_sec))
            )
            node_native_idf[idx] = n_idf
            node_xling_idf[idx] = x_idf
            node_direct_topic_idf[idx] = max(n_idf, x_idf) + (0.35 * min(n_idf, x_idf))

            if entity_matchers:
                # In entity-scoped retrieval, only facts belonging to the target entity AND matching
                # a topic term (or when no topic was specified) act as PageRank anchors.
                if node_matched_ent[idx] is not None and (node_direct_topic_idf[idx] > 0.0 or not specific_topic_terms):
                    node_seed_eligible[idx] = True
            else:
                matched_native = [t for t in native_topic_terms if (t in node.tokens or t in node.norm_fact)]
                matched_xling = [t for t in xling_topic_terms if (t in node.tokens or t in node.norm_fact)]
                n_cov = (n_idf / total_native_idf) if native_topic_terms else 0.0
                x_cov = (x_idf / total_xling_idf) if xling_topic_terms else 0.0
                if (
                    (bool(native_topic_terms) and (len(matched_native) == len(native_topic_terms) or (n_cov >= 0.45 and n_idf >= 0.85)))
                    or (bool(xling_topic_terms) and (len(matched_xling) == len(xling_topic_terms) or (x_cov >= 0.45 and x_idf >= 0.85)))
                ):
                    node_seed_eligible[idx] = True

        damping = 0.85
        max_iter = 3
        fact_ppr: List[float] = [0.0] * total_fact_nodes
        fact_bridge_act: List[float] = [0.0] * total_fact_nodes

        if any(node_seed_eligible):
            p_feedback: Dict[str, float] = {}
            for step in range(max_iter):
                step_facts: List[float] = [0.0] * total_fact_nodes

                # Teleport / seed concept energy goes strictly to seed-eligible facts (never self-loops on unmatched stops)
                for concept, c_act in p0_concepts.items():
                    if c_act <= 0.0:
                        continue
                    connected = concept_to_nodes.get(concept)
                    if not connected:
                        continue
                    share = c_act / len(connected)
                    for f_idx in connected:
                        if node_seed_eligible[f_idx]:
                            step_facts[f_idx] += share

                # Multi-hop bridge energy propagates from active facts through discriminative concepts (df >= 2)
                for concept, b_act in p_feedback.items():
                    if b_act <= 0.0:
                        continue
                    connected = concept_to_nodes.get(concept)
                    if not connected or len(connected) < 2:
                        continue
                    share = (damping * b_act) / len(connected)
                    for f_idx in connected:
                        step_facts[f_idx] += share
                        if not node_seed_eligible[f_idx]:
                            fact_bridge_act[f_idx] += share

                # Entity phrase teleportation anchor for seed-eligible nodes
                if entity_matchers:
                    for f_idx in range(total_fact_nodes):
                        if node_seed_eligible[f_idx]:
                            step_facts[f_idx] += 0.35 * node_ent_score[f_idx] + 0.45 * node_direct_topic_idf[f_idx]

                fact_ppr = step_facts

                if step + 1 < max_iter:
                    # Continuous Hill-function specificity gate calibrated to the graph's own seed IDF scale
                    max_seed_idf = max((_mem_term_idf(k) for k in p0_concepts), default=1.0)
                    bridge_anchor = max(0.35, 0.45 * max_seed_idf)
                    bridge_anchor_sq = bridge_anchor * bridge_anchor
                    feedback_concepts: Dict[str, float] = {}
                    for f_idx, f_act in enumerate(step_facts):
                        if f_act <= 0.0:
                            continue
                        node = all_nodes[f_idx]
                        deg = max(len(node.tokens), 1)
                        out_share = f_act / deg
                        for tok in node.tokens:
                            # Suppress entity tokens, original seeds, degree-1 leaves, and all-node hubs
                            if tok in entity_token_union or tok in p0_concepts:
                                continue
                            tok_df = len(concept_to_nodes.get(tok, ()))
                            if tok_df < 2 or (total_fact_nodes > 2 and tok_df >= total_fact_nodes):
                                continue
                            tok_idf = _mem_term_idf(tok)
                            if tok_idf <= 0.0:
                                continue
                            tok_sq = tok_idf * tok_idf
                            gate = tok_sq / (tok_sq + bridge_anchor_sq)
                            feedback_concepts[tok] = feedback_concepts.get(tok, 0.0) + (out_share * gate)
                    p_feedback = feedback_concepts

        # Temporal modifiers across all candidates
        now_ts = time.time()
        window_hours = probe.time_window_hours if probe else None
        recency_bias = bool(probe.recency_bias) if probe else False
        temp_info = [
            _temporal_multiplier(node.timestamp_epoch, now_ts, window_hours, recency_bias)
            for node in all_nodes
        ]

        has_strict_in_window = False
        if window_hours:
            for idx in range(total_fact_nodes):
                if temp_info[idx][1] is True and (
                    node_direct_topic_idf[idx] > 0.0 or fact_bridge_act[idx] > 0.0 or node_ent_score[idx] > 0.0
                ):
                    has_strict_in_window = True
                    break

        scored_candidates: List[Tuple[float, int, str, _MemoryFactNode]] = []
        max_bridge = max(fact_bridge_act) if fact_bridge_act else 0.0
        bridge_floor = max(max_bridge * 0.25, 0.025) if max_bridge > 0.0 else 0.0

        if entity_matchers:
            entity_activated: Dict[str, List[Tuple[float, float, int, _MemoryFactNode]]] = {
                ent_name: [] for ent_name, _, _ in entity_matchers
            }
            for idx, node in enumerate(all_nodes):
                if has_strict_in_window and temp_info[idx][1] is False:
                    continue
                t_mult = temp_info[idx][0]
                matched_ent = node_matched_ent[idx]
                ent_base = node_ent_score[idx]
                topic_or_bridge = node_direct_topic_idf[idx] + (6.0 * fact_bridge_act[idx])
                total_act = (ent_base + (2.5 * topic_or_bridge) + (4.0 * fact_ppr[idx])) * t_mult
                if matched_ent:
                    entity_activated[matched_ent].append((total_act, topic_or_bridge, idx, node))

            for ent_name, items in entity_activated.items():
                if not items:
                    continue
                # In large entity collections (> 6 facts) or DB history stores with a specific topic,
                # filter out zero-topic/sub-floor bleed so only genuinely relevant facts are injected.
                topic_hits = [
                    it
                    for it in items
                    if node_direct_topic_idf[it[2]] > 0.0
                    or (bridge_floor > 0.0 and fact_bridge_act[it[2]] >= bridge_floor)
                ]
                has_db_items = any(":" in it[3].source_file for it in items)
                if (len(items) > 6 or (specific_topic_terms and has_db_items)) and topic_hits:
                    selected_items = sorted(topic_hits, key=lambda x: (-x[0], x[2]))
                else:
                    selected_items = sorted(items, key=lambda x: (-x[0], x[2]))

                for total_act, _, idx, node in selected_items:
                    scored_candidates.append((total_act, idx, ent_name, node))

            # Include multi-hop bridge facts discovered in DB / unheaded notes linked by high-IDF concepts
            # without cross-contaminating distinct Markdown entity headings (e.g. `## OtherClient`).
            if max_bridge > 0.0:
                primary_label = entity_matchers[0][0]
                for idx, node in enumerate(all_nodes):
                    if node_matched_ent[idx] is not None:
                        continue
                    if has_strict_in_window and temp_info[idx][1] is False:
                        continue
                    if node.section_entity and ":" not in node.source_file:
                        # Explicit conflicting section heading in Markdown file belongs to a different entity
                        continue
                    if fact_bridge_act[idx] >= bridge_floor:
                        score = (2.5 * node_direct_topic_idf[idx] + 6.0 * fact_bridge_act[idx] + 4.0 * fact_ppr[idx]) * temp_info[idx][0]
                        scored_candidates.append((score, idx, node.section_entity or primary_label, node))
        elif topic_terms:
            candidate_indices: Set[int] = set()
            for idx in range(total_fact_nodes):
                if node_seed_eligible[idx] or (max_bridge > 0.0 and fact_bridge_act[idx] >= max(max_bridge * 0.30, 0.025)):
                    candidate_indices.add(idx)

            label = probe.topic if (probe and probe.topic) else "Context"
            for idx in sorted(candidate_indices):
                if has_strict_in_window and temp_info[idx][1] is False:
                    continue
                node = all_nodes[idx]
                combined = (node_direct_topic_idf[idx] + 6.0 * fact_bridge_act[idx] + 3.0 * fact_ppr[idx]) * temp_info[idx][0]
                scored_candidates.append((combined, idx, node.section_entity or label, node))

        scored_candidates.sort(key=lambda x: (-x[0], x[1]))

        results: List[MemoryItem] = []
        current_tokens = 0
        seen_facts: Set[str] = set()

        for _, _, ent_label, node in scored_candidates:
            if node.fact in seen_facts:
                continue
            if self.token_budget > 0 and current_tokens + node.token_estimate > self.token_budget:
                continue
            seen_facts.add(node.fact)
            results.append(
                MemoryItem(
                    entity=ent_label,
                    fact=node.fact,
                    source_file=node.source_file,
                    token_estimate=node.token_estimate,
                )
            )
            current_tokens += node.token_estimate

        return results

    def _build_db_search_terms(self, clean_entities: List[str], topic_terms: List[str]) -> List[str]:
        terms: List[str] = []
        for e in clean_entities:
            e_clean = e.strip()
            if len(e_clean) >= 2 and e_clean not in terms:
                terms.append(e_clean)
        for t in topic_terms:
            if len(t) >= 2 and t not in terms:
                terms.append(t)
        return terms

    def _discover_bridge_terms(
        self,
        seed_nodes: List[_MemoryFactNode],
        existing_terms: List[str],
        clean_entities: List[str],
        specific_topic_terms: Optional[List[str]] = None,
        max_bridge: int = 8,
    ) -> List[str]:
        """Extract highest-specificity non-entity tokens from Stage-1 topic-relevant rows for Hop-2 SQL expansion."""
        if not seed_nodes:
            return []
        excluded: Set[str] = set()
        for t in existing_terms:
            excluded.update(tokenize(t))
        for e in clean_entities:
            excluded.update(tokenize(e))

        spec_set = set(specific_topic_terms or ())
        anchor_nodes = (
            [n for n in seed_nodes if any(t in n.tokens or t in n.norm_fact for t in spec_set)]
            if spec_set
            else seed_nodes
        )
        effective_nodes = anchor_nodes or seed_nodes

        df: Dict[str, int] = {}
        n_seed = len(effective_nodes)
        for node in effective_nodes:
            for tok in node.tokens:
                if tok not in excluded:
                    df[tok] = df.get(tok, 0) + 1

        ranked = sorted(
            (
                (
                    term_information_weight(tok, cnt, max(n_seed, 2))
                    * min(cnt, 3)
                    * (1.35 if ("-" in tok or "_" in tok or any(c.isdigit() for c in tok)) else 1.0),
                    len(tok),
                    tok,
                )
                for tok, cnt in df.items()
            ),
            key=lambda x: (-x[0], -x[1], x[2]),
        )
        return [tok for score, _, tok in ranked[:max_bridge] if score >= 0.65]

    def _make_db_fact_node(
        self,
        raw_text: str,
        source_label: str,
        section_entity: Optional[str] = None,
        timestamp_val: Any = None,
        max_chars: int = 800,
    ) -> Optional[_MemoryFactNode]:
        val = " ".join(str(raw_text or "").split()).strip()
        if len(val) < 10 or _INJECTION_PATTERNS.search(val):
            return None
        if len(val) > max_chars:
            val = val[:max_chars].rstrip() + "..."
        norm_val = normalize_text_for_matching(val)
        toks = set(tokenize(val))
        if not toks:
            return None
        return _MemoryFactNode(
            fact=val,
            norm_fact=norm_val,
            tokens=toks,
            section_entity=section_entity,
            source_file=source_label,
            token_estimate=estimate_tokens(val),
            timestamp_epoch=_parse_epoch_seconds(timestamp_val),
        )

    def _extract_sqlite_fact_nodes(
        self,
        db_path: Path,
        clean_entities: List[str],
        topic_terms: List[str],
        specific_topic_terms: Optional[List[str]] = None,
        active_session_id: Optional[str] = None,
    ) -> List[_MemoryFactNode]:
        """
        Read-only universal SQLite adapter:
        1. Natively queries Hermes `state.db` (`messages` / `messages_fts` + `sessions`).
        2. Natively queries Engram `engram.db` (`observations` / `observations_fts`).
        3. Dynamically introspects ANY custom SQLite user schema (`sqlite_master` + `PRAGMA table_info`)
           and converts matching rows into graph `_MemoryFactNode` instances.
        """
        nodes: List[_MemoryFactNode] = []
        search_terms = self._build_db_search_terms(clean_entities, topic_terms)
        if not search_terms:
            return nodes

        try:
            uri = f"file:{db_path.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=5.0) as conn:
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.cursor()
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
                table_sql_rows = [(str(r[0]), str(r[1]) if r[1] is not None else None) for r in cursor.fetchall()]
                tables = [name for name, _ in table_sql_rows]
                table_set_lower = {t.lower(): t for t in tables}
                fts_tables, fts_shadows = _detect_fts_tables_and_shadows(table_sql_rows)

                # 1. Direct structural verification of Hermes `state.db` (`messages` with `content`, `session_id`, `role`)
                is_hermes_state_db = False
                if "messages" in table_set_lower:
                    msg_tbl = table_set_lower["messages"]
                    cursor.execute(f'PRAGMA table_info("{msg_tbl}")')
                    msg_cols = {str(r[1]).lower() for r in cursor.fetchall()}
                    if {"content", "session_id", "role"}.issubset(msg_cols):
                        is_hermes_state_db = True
                        nodes.extend(
                            self._query_hermes_state_db(
                                cursor,
                                db_path.name,
                                msg_tbl,
                                table_set_lower,
                                search_terms,
                                clean_entities=clean_entities,
                                specific_topic_terms=specific_topic_terms,
                                active_session_id=active_session_id,
                            )
                        )

                # In verified Hermes state.db, session history lives exclusively in `messages` (`sessions`),
                # so avoid scanning runtime/lock coordination tables.
                if is_hermes_state_db:
                    return nodes

                # 2. Direct structural + statistical profiling for any user/ecosystem SQLite database
                for tbl in tables:
                    t_low = tbl.lower()
                    if t_low.startswith("sqlite_") or t_low in fts_tables or t_low in fts_shadows:
                        continue
                    nodes.extend(
                        self._query_introspected_sqlite_table(
                            cursor,
                            db_path.name,
                            tbl,
                            fts_tables,
                            search_terms,
                            clean_entities=clean_entities,
                            specific_topic_terms=specific_topic_terms,
                        )
                    )
        except Exception:
            pass  # Read-only DB probe must never crash or block
        return nodes

    def _query_hermes_state_db(
        self,
        cursor: sqlite3.Cursor,
        db_name: str,
        msg_table: str,
        table_set_lower: Dict[str, str],
        search_terms: List[str],
        clean_entities: Optional[List[str]] = None,
        specific_topic_terms: Optional[List[str]] = None,
        active_session_id: Optional[str] = None,
    ) -> List[_MemoryFactNode]:
        nodes: List[_MemoryFactNode] = []
        seen_contents: Set[str] = set()

        cursor.execute(f'PRAGMA table_info("{msg_table}")')
        msg_cols = {str(r[1]).lower() for r in cursor.fetchall()}
        if "content" not in msg_cols:
            return nodes

        has_sessions = "sessions" in table_set_lower
        has_active = "active" in msg_cols
        has_role = "role" in msg_cols
        has_session_id = "session_id" in msg_cols
        has_ts = "timestamp" in msg_cols
        has_deleted_at = "deleted_at" in msg_cols

        where_clauses = ["m.content IS NOT NULL", "length(m.content) >= 10"]
        base_params: List[Any] = []
        if has_role:
            where_clauses.append("m.role IN ('user', 'assistant')")
        if has_active:
            where_clauses.append("m.active = 1")
        if has_deleted_at:
            where_clauses.append("m.deleted_at IS NULL")
        if has_session_id and active_session_id:
            where_clauses.append("m.session_id != ?")
            base_params.append(active_session_id)

        title_expr = "s.title" if (has_sessions and has_session_id) else "NULL"
        ts_expr = "m.timestamp" if has_ts else "NULL"
        order_expr = "m.timestamp DESC, m.id DESC" if has_ts else "m.id DESC"
        join_sql = (
            f'LEFT JOIN "{table_set_lower["sessions"]}" s ON s.id = m.session_id'
            if (has_sessions and has_session_id)
            else ""
        )

        def _fetch_state_rows(terms: List[str], limit: int) -> None:
            fetched_any = False
            if "messages_fts" in table_set_lower:
                fts_terms = ['"' + t.replace('"', '""') + '"' for t in terms if len(t) >= 2]
                if fts_terms:
                    fts_query = " OR ".join(fts_terms)
                    try:
                        sql = (
                            f'SELECT m.content, {title_expr}, {ts_expr} FROM "{msg_table}" m {join_sql} '
                            f'JOIN "messages_fts" f ON f.rowid = m.id '
                            f'WHERE {" AND ".join(where_clauses)} AND messages_fts MATCH ? '
                            f'ORDER BY {order_expr} LIMIT {limit}'
                        )
                        cursor.execute(sql, base_params + [fts_query])
                        rows_fts = cursor.fetchall()
                        if rows_fts:
                            fetched_any = True
                        for content_val, sess_title, ts_val in rows_fts:
                            c_str = str(content_val or "").strip()
                            if c_str and c_str not in seen_contents:
                                seen_contents.add(c_str)
                                node = self._make_db_fact_node(
                                    c_str,
                                    source_label=f"{db_name}:{msg_table}",
                                    section_entity=str(sess_title).strip() if sess_title else None,
                                    timestamp_val=ts_val,
                                )
                                if node is not None:
                                    nodes.append(node)
                    except Exception:
                        pass

            if not fetched_any:
                like_clauses = []
                like_params: List[Any] = list(base_params)
                for term in terms[:6]:
                    like_clauses.append("m.content LIKE ? ESCAPE '\\'")
                    like_params.append(f"%{_escape_sql_like(term)}%")
                if like_clauses:
                    sql = (
                        f'SELECT m.content, {title_expr}, {ts_expr} FROM "{msg_table}" m {join_sql} '
                        f'WHERE {" AND ".join(where_clauses)} AND ({" OR ".join(like_clauses)}) '
                        f'ORDER BY {order_expr} LIMIT {limit}'
                    )
                    try:
                        cursor.execute(sql, like_params)
                        for content_val, sess_title, ts_val in cursor.fetchall():
                            c_str = str(content_val or "").strip()
                            if c_str and c_str not in seen_contents:
                                seen_contents.add(c_str)
                                node = self._make_db_fact_node(
                                    c_str,
                                    source_label=f"{db_name}:{msg_table}",
                                    section_entity=str(sess_title).strip() if sess_title else None,
                                    timestamp_val=ts_val,
                                )
                                if node is not None:
                                    nodes.append(node)
                    except Exception:
                        pass

        # Stage 1: Seed fetch
        _fetch_state_rows(search_terms, limit=120)

        # Stage 2: Multi-hop bridge expansion across past sessions via highest-IDF discovered concepts
        if nodes:
            bridge_terms = self._discover_bridge_terms(
                nodes,
                search_terms,
                clean_entities or [],
                specific_topic_terms=specific_topic_terms,
            )
            if bridge_terms:
                _fetch_state_rows(bridge_terms, limit=40)

        return nodes

    def _query_introspected_sqlite_table(
        self,
        cursor: sqlite3.Cursor,
        db_name: str,
        table_name: str,
        fts_tables: Dict[str, str],
        search_terms: List[str],
        clean_entities: Optional[List[str]] = None,
        specific_topic_terms: Optional[List[str]] = None,
    ) -> List[_MemoryFactNode]:
        nodes: List[_MemoryFactNode] = []
        seen_contents: Set[str] = set()
        safe_tbl = table_name.replace('"', '""')
        try:
            cursor.execute(f'PRAGMA table_info("{safe_tbl}")')
            col_rows = cursor.fetchall()
        except Exception:
            return nodes

        if not col_rows:
            return nodes

        col_names = [str(r[1]) for r in col_rows]
        col_types = [str(r[2] or "") for r in col_rows]
        deleted_at_col = next((name for name in col_names if name.lower() == "deleted_at"), None)
        active_row_filter = f' WHERE "{deleted_at_col.replace(chr(34), chr(34) * 2)}" IS NULL' if deleted_at_col else ""

        # Direct sample check: skip empty tables and profile actual column data distributions
        try:
            cursor.execute(f'SELECT * FROM "{safe_tbl}"{active_row_filter} LIMIT 30')
            sample_rows = cursor.fetchall()
        except Exception:
            return nodes

        if not sample_rows:
            return nodes

        selected_content_cols, entity_col_orig, ts_col_orig = _profile_columns_from_sample(
            col_names, col_types, sample_rows
        )
        if not selected_content_cols:
            return nodes

        def _qcol(name: str) -> str:
            return f'"{name.replace(chr(34), chr(34) * 2)}"'

        select_cols = [_qcol(c) for c in selected_content_cols]
        if entity_col_orig:
            select_cols.append(_qcol(entity_col_orig))
        if ts_col_orig:
            select_cols.append(_qcol(ts_col_orig))

        search_cols = list(selected_content_cols)
        if entity_col_orig:
            search_cols.append(entity_col_orig)

        order_sql = f' ORDER BY {_qcol(ts_col_orig)} DESC' if ts_col_orig else ""
        t_order_sql = f' ORDER BY t.{_qcol(ts_col_orig)} DESC' if ts_col_orig else ""
        fts_match_tbl = fts_tables.get(f"{table_name.lower()}_fts")

        n_content = len(selected_content_cols)
        ent_idx = n_content if entity_col_orig else None
        ts_idx = (n_content + (1 if entity_col_orig else 0)) if ts_col_orig else None

        def _collect_rows(fetched: List[Tuple[Any, ...]]) -> None:
            for row in fetched:
                ent_val = str(row[ent_idx]).strip() if (ent_idx is not None and row[ent_idx] is not None) else None
                ts_val = row[ts_idx] if ts_idx is not None else None
                parts = [str(row[i]).strip() for i in range(n_content) if row[i] is not None and str(row[i]).strip()]
                if not parts:
                    continue
                combined_text = " | ".join(parts)
                if combined_text in seen_contents:
                    continue
                seen_contents.add(combined_text)
                node = self._make_db_fact_node(
                    combined_text,
                    source_label=f"{db_name}:{table_name}",
                    section_entity=ent_val or None,
                    timestamp_val=ts_val,
                )
                if node is not None:
                    nodes.append(node)

        def _query_terms_pass(terms: List[str], limit: int) -> bool:
            rows: List[Tuple[Any, ...]] = []
            if fts_match_tbl:
                fts_terms = ['"' + t.replace('"', '""') + '"' for t in terms if len(t) >= 2]
                if fts_terms:
                    safe_fts = fts_match_tbl.replace('"', '""')
                    fts_query = " OR ".join(fts_terms)
                    try:
                        t_select = ", ".join(f't.{c}' for c in select_cols)
                        sql = (
                            f'SELECT {t_select} FROM "{safe_tbl}" t '
                            f'JOIN "{safe_fts}" f ON f.rowid = t.rowid '
                            f'WHERE "{safe_fts}" MATCH ?'
                            f'{(" AND t." + _qcol(deleted_at_col) + " IS NULL") if deleted_at_col else ""}'
                            f'{t_order_sql} LIMIT {limit}'
                        )
                        cursor.execute(sql, (fts_query,))
                        rows = cursor.fetchall()
                    except Exception:
                        rows = []

            if not rows:
                where_parts: List[str] = []
                params: List[Any] = []
                for term in terms[:6]:
                    pat = f"%{_escape_sql_like(term)}%"
                    for scol in search_cols:
                        where_parts.append(f'CAST({_qcol(scol)} AS TEXT) LIKE ? ESCAPE \'\\\'')
                        params.append(pat)
                if where_parts:
                    sql = (
                        f'SELECT {", ".join(select_cols)} FROM "{safe_tbl}" '
                        f'WHERE {active_row_filter[7:] + " AND " if active_row_filter else ""}'
                        f'({" OR ".join(where_parts)}){order_sql} LIMIT {limit}'
                    )
                    try:
                        cursor.execute(sql, params)
                        rows = cursor.fetchall()
                    except Exception:
                        rows = []

            if rows:
                _collect_rows(rows)
                return True
            return False

        # Stage 1: Seed fetch
        stage1_hit = _query_terms_pass(search_terms, limit=120)

        # Stage 2: Multi-hop bridge expansion for custom SQLite tables / engram.db
        if nodes:
            bridge_terms = self._discover_bridge_terms(
                nodes,
                search_terms,
                clean_entities or [],
                specific_topic_terms=specific_topic_terms,
            )
            if bridge_terms:
                _query_terms_pass(bridge_terms, limit=40)

        # Diacritic/script-tolerant fallback for small tables (< 300 rows) when SQL LIKE/FTS missed
        if not stage1_hit and not nodes:
            try:
                cursor.execute(
                    f'SELECT {", ".join(select_cols)} FROM "{safe_tbl}"'
                    f'{active_row_filter}{order_sql} LIMIT 300'
                )
                _collect_rows(cursor.fetchall())
            except Exception:
                pass

        return nodes

    def _extract_external_uri_fact_nodes(
        self,
        db_uri: str,
        clean_entities: List[str],
        topic_terms: List[str],
    ) -> List[_MemoryFactNode]:
        """
        SELECT-only adapter for external user-ecosystem databases configured via URI
        (sqlite:///..., postgresql://..., mysql://..., etc. via SQLAlchemy when available).
        Database-enforced read-only access depends on the caller's URI credentials and server.
        Uses direct statistical sample profiling (zero static column/table name lists).
        """
        if db_uri.startswith("sqlite:///"):
            local_path = Path(db_uri[len("sqlite:///"):])
            if local_path.is_file() and is_safe_memory_file(local_path):
                return self._extract_sqlite_fact_nodes(
                    local_path,
                    clean_entities=clean_entities,
                    topic_terms=topic_terms,
                )
            return []

        search_terms = self._build_db_search_terms(clean_entities, topic_terms)
        if not search_terms:
            return []

        nodes: List[_MemoryFactNode] = []
        try:
            from sqlalchemy import create_engine, inspect, text
        except ImportError:
            return nodes

        engine = None
        try:
            engine = create_engine(db_uri, pool_pre_ping=True)
            inspector = inspect(engine)
            identifier_preparer = engine.dialect.identifier_preparer

            def _q_identifier(name: str) -> str:
                return identifier_preparer.quote(name, force=True)

            table_names = inspector.get_table_names()
            with engine.connect() as conn:
                for tbl in table_names:
                    cols = inspector.get_columns(tbl)
                    if not cols:
                        continue
                    col_names = [c["name"] for c in cols]
                    col_types = [str(c.get("type", "")) for c in cols]
                    deleted_at_col = next((name for name in col_names if name.lower() == "deleted_at"), None)

                    q_table = _q_identifier(tbl)
                    sample_filter = f' WHERE {_q_identifier(deleted_at_col)} IS NULL' if deleted_at_col else ""
                    sample_res = conn.execute(text(f'SELECT * FROM {q_table}{sample_filter} LIMIT 30'))
                    sample_rows = [tuple(r) for r in sample_res.fetchall()]
                    if not sample_rows:
                        continue

                    selected_cols, ent_col, ts_col = _profile_columns_from_sample(
                        col_names, col_types, sample_rows
                    )
                    if not selected_cols:
                        continue

                    quoted_cols = (
                        [_q_identifier(c) for c in selected_cols]
                        + ([_q_identifier(ent_col)] if ent_col else [])
                        + ([_q_identifier(ts_col)] if ts_col else [])
                    )
                    search_cols = list(selected_cols) + ([ent_col] if ent_col else [])

                    like_clauses = []
                    bind_params: Dict[str, Any] = {}
                    p_idx = 0
                    for term in search_terms[:8]:
                        for scol in search_cols:
                            pname = f"p{p_idx}"
                            p_idx += 1
                            like_clauses.append(f'CAST({_q_identifier(scol)} AS TEXT) LIKE :{pname}')
                            bind_params[pname] = f"%{term}%"

                    if not like_clauses:
                        continue

                    order_sql = f' ORDER BY {_q_identifier(ts_col)} DESC' if ts_col else ""
                    active_filter = f' AND {_q_identifier(deleted_at_col)} IS NULL' if deleted_at_col else ""
                    stmt = text(
                        f'SELECT {", ".join(quoted_cols)} FROM {q_table} '
                        f'WHERE ({" OR ".join(like_clauses)}){active_filter}{order_sql} LIMIT 120'
                    )
                    res = conn.execute(stmt, bind_params)
                    n_content = len(selected_cols)
                    ent_idx = n_content if ent_col else None
                    ts_idx = (n_content + (1 if ent_col else 0)) if ts_col else None
                    for row in res.fetchall():
                        ent_val = str(row[ent_idx]).strip() if (ent_idx is not None and row[ent_idx] is not None) else None
                        ts_val = row[ts_idx] if ts_idx is not None else None
                        parts = [str(row[i]).strip() for i in range(n_content) if row[i] is not None and str(row[i]).strip()]
                        if not parts:
                            continue
                        node = self._make_db_fact_node(
                            " | ".join(parts),
                            source_label=f"external_db:{tbl}",
                            section_entity=ent_val or None,
                            timestamp_val=ts_val,
                        )
                        if node is not None:
                            nodes.append(node)
        except Exception:
            pass
        finally:
            if engine is not None:
                try:
                    engine.dispose()
                except Exception:
                    pass
        return nodes
