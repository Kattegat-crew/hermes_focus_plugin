---
name: llm-structured-output
description: >-
  Deterministic JSON schema generation, strict validation, and self-repairing
  extraction for LLM-based cognitive triage and entity analysis.
---

# LLM Structured Output & Cognitive Triage Extraction

Use this skill when designing the cognitive semantic interpreter that analyzes incoming user messages and produces structured extraction targets without hallucination or malformed JSON errors.

## Core Contract

When performing cognitive semantic triage before main inference:
1. **Never pass raw text directly into search**:
   - The user query may be ambiguous, multi-intent, or written in Spanish while the catalog is in English.
   - The model must translate the intent into canonical technical capabilities and explicit entities.
2. **Deterministic Schema**:
   - Every inference must output a strict JSON payload conforming to `TriageAnalysis`.
3. **Fail-Closed on Validation Failure**:
   - If the output fails validation after a single repair attempt, emit an empty triage result (`skills=[]`, `entities=[]`) and log the incident to the receipt. Never crash the turn.

## Data Models (Pydantic v2)

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class DomainClassification(BaseModel):
    is_software_task: bool = Field(
        ..., 
        description="True if the request involves coding, debugging, architecture, git, or devops."
    )
    detected_stack: List[str] = Field(
        default_factory=list,
        description="Detected technologies, frameworks, or languages (e.g. ['python', 'fastapi', 'docker'])."
    )

class TriageAnalysis(BaseModel):
    is_actionable_task: bool = Field(
        ...,
        description="False for casual greetings or small talk; True for actionable requests."
    )
    intent_summary_en: str = Field(
        ...,
        max_length=280,
        description="Clear, 1-sentence imperative description in English of what needs to be done."
    )
    required_capabilities: List[str] = Field(
        default_factory=list,
        max_items=6,
        description="3 to 6 technical capability nouns in English (e.g., ['campaign-planning', 'copywriting'])."
    )
    extracted_entities: List[str] = Field(
        default_factory=list,
        description="Named entities for selective memory recall (e.g. client names, projects, campaign IDs)."
    )
    domain: DomainClassification
```

## Validation & Repair Loop

1. **Prompt Structure**:
   - Specify the exact JSON schema in the system prompt.
   - For models supporting JSON mode or structured outputs (`response_format={"type": "json_object"}`), enable it.
2. **Self-Healing Parser**:
   ```python
   import json
   import re

   def clean_and_parse_json(raw_text: str) -> dict:
       raw = raw_text.strip()
       # Strip markdown fences if present
       match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
       if match:
           raw = match.group(1).strip()
       return json.loads(raw)
   ```
3. **Fallback Guarantee**:
   - Any unparseable response yields `TriageAnalysis(is_actionable_task=False, intent_summary_en="", required_capabilities=[], extracted_entities=[], domain=DomainClassification(is_software_task=False))` with `status="parse_failed"` in the execution receipt.
