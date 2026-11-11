---
name: bm25-sidecar-search
description: >-
  Ultra-fast in-memory and SQLite-backed BM25 scoring over skill descriptions
  and lead summaries, cached by SHA256 without mutating source files.
---

# BM25 Sidecar Search & Indexing

Use this skill when building the candidate generation engine and fast lexical scorer for skills in the plugin.

## Principles

1. **Zero Mutation of User Files**:
   - The plugin must NEVER modify or overwrite user `SKILL.md` files.
   - Any parsed lead, extracted metadata, or inverted index must reside in `plugin_data_dir() / "sidecar_index.db"`.
2. **Deterministic Invalidation by SHA256**:
   - Cache each skill by its file content hash (`sha256(SKILL.md)`).
   - If the file has not changed, reuse the cached description and lead paragraph. If it changed, re-index that single file.
3. **Sub-5ms Query Latency**:
   - For catalogs of 500–1000 skills, BM25 query time must remain under 5 ms.
   - Index only the skill name, `description` (from frontmatter), and the first lead paragraph (up to 400 characters). Never index the full multi-thousand-token body.

## Python BM25 Implementation (Standard Library / Zero Heavy Dependencies)

```python
import math
import re
from typing import List, Tuple, Dict

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}  # Document frequency
        self.tf: Dict[str, Dict[str, int]] = {}  # doc_id -> term -> count
        self.n_docs: int = 0

    def add_document(self, doc_id: str, content: str):
        tokens = tokenize(content)
        self.doc_len[doc_id] = len(tokens)
        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        self.tf[doc_id] = counts
        for t in counts:
            self.df[t] = self.df.get(t, 0) + 1
        self.n_docs += 1
        self.avg_doc_len = sum(self.doc_len.values()) / max(1, self.n_docs)

    def score(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        q_tokens = tokenize(query)
        scores: Dict[str, float] = {}

        for t in q_tokens:
            if t not in self.df:
                continue
            # Standard Lucene/Robertson IDF
            idf = math.log(1.0 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for doc_id, counts in self.tf.items():
                if t not in counts:
                    continue
                tf = counts[t]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / max(1.0, self.avg_doc_len)))
                score_t = idf * (num / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + score_t

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
```

## Sidecar Database Schema (SQLite)

```sql
CREATE TABLE IF NOT EXISTS skills_sidecar (
    skill_name TEXT PRIMARY KEY,
    skill_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    description TEXT,
    lead_paragraph TEXT,
    indexed_tokens INTEGER,
    last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_skills_sha256 ON skills_sidecar(sha256);
```
