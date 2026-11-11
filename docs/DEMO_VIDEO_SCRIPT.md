# Neural Context Plugin — Official Hackathon Demo Video Script
**Target Duration:** 75–85 seconds (Strictly < 90 seconds requirement)  
**Channel:** X (Twitter) posting tagging `@NousResearch`  
**Author:** Jesús Díaz (Jemadiar) & NeuralCrew Labs Architecture Team  
**Event:** Nous Research Hermes Agent Plugin Hackathon (November 11, 2026)

---

## Visual Setup & Screen Layout
- **Terminal 1 (Left / Main)**: Hermes Agent running in interactive CLI mode with token usage logging (`token-lens` or verbose token counter).
- **Terminal 2 (Right / Split)**: Split terminal showing live sidecar receipts (`hermes neural-context verify --limit 5`) and doctor diagnostics.

---

## Scene 1: The Hook — The Mature Agent Dilemma (00:00 - 00:15)
- **Visual**: Show a split-screen comparison:
  - *Standard Hermes*: System prompt loaded with 500 skills = **14,500 prompt tokens** on a simple greeting.
  - *Voiceover / Audio*:
    > *"When your Hermes Agent grows from 10 skills to 500 skills, you face a brutal dilemma: either bloat every turn by 15,000 tokens in the system prompt, or run blind with on-demand tools and burn expensive round-trips. Meet Neural Context Plugin."*

---

## Scene 2: The Core Innovation — Two-Phase Enriched Routing (00:15 - 00:45)
- **Visual**: In Terminal 1, type a real shorthand ambiguous prompt:
  ```bash
  > hermes
  User: Revisa lo de AcmeCorp
  ```
- **Terminal Action**: Hermes responds immediately with expert procedural advice for Meta Conversions API (CAPI) and FastAPI event deduplication.
- **Voiceover / Audio**:
  > *"Watch this prompt: 'Revisa lo de AcmeCorp'. It doesn't mention Meta, CAPI, or Python. But Neural Context intercepts the turn before inference in two phases:*
  > *Phase 1 extracts the entity 'AcmeCorp' and pulls domain ground truth from selective memory—discovering AcmeCorp is an online retail service using Meta CAPI.*
  > *Phase 2 enriches the skill search with those memory facts, instantly ranking the exact winning skill from our local SHA-256 sidecar index.*
  > *Both the procedural skill and memory facts are injected into the user turn in a SINGLE step—zero round trips, and the system prompt stays 100% byte-stable for instant KV-cache hits."*

---

## Scene 3: Mathematical Proof & Verification CLI (00:45 - 01:10)
- **Visual**: Switch to Terminal 2 and run the verification command:
  ```bash
  hermes neural-context verify --limit 1
  ```
- **Terminal Output**:
  ```text
  === Neural Context Verifiable Receipts (Last 1 turns) ===
    Turns Audited       : 1
    Skills Routed       : 1 (meta-pixel-capi)
    Injected Tokens     : 380 tokens (Skill + Factual Context)
    Conversational Exit : Active (Zero tokens injected on pure chit-chat)
    Tokens Saved Total  : 11,620 tokens vs static preloading
  ```
- **Visual**: Now show a casual greeting fast-exit:
  ```bash
  User: Hola amigo
  Hermes: ¡Hola! ¿En qué te puedo ayudar hoy?
  ```
- **Voiceover / Audio**:
  > *"Every turn produces a deterministic, verifiable receipt. Notice that turn: selective ephemeral context without preloading the entire catalog.*
  > *And when you say 'Hola'? The Cognitive Arbiter classifies it as DIRECT_REPLY and exits immediately with ZERO token injection."*

---

## Scene 4: Official Hermes Validation & Outro (01:10 - 01:25)
- **Visual**: Run doctor and validation commands:
  ```bash
  hermes plugins validate .
  hermes plugins doctor .
  ```
- **Terminal Output**: Clean green checkmarks across all tools, hooks, and isolation probes.
- **Voiceover / Audio**:
  > *"100% compliant with Hermes Plugin architecture, 34/34 unit tests passing, and zero file mutations.*
  > *Built by Jesús Díaz and Kattegat Crew. Install it today with `hermes plugins install Kattegat-crew/neural-context-plugin`."*

---

## Accompanying X (Twitter) Post Copy
```text
Proud to submit Neural Context Plugin to the @NousResearch Hermes Agent Plugin Hackathon! 🚀🧠

Solves the Mature Agent Dilemma:
✅ Cognitive Arbiter: 4-Quadrant Intent Model (DIRECT_REPLY, SKILL_ONLY, MEMORY_ONLY, ENRICHED_TASK)
✅ Two-Phase Context-Enriched Routing (Selective Memory ➔ Skill Selection)
✅ Free, local Bipartite Concept GraphRouter with Spreading Activation
✅ Multilingual native-language memory probes & skill gap detection
✅ Single-step ephemeral injection into user turn (100% Byte-stable System Prompt / KV-cache friendly)
✅ Verifiable Audit Receipts & CLI (`hermes neural-context verify`)

GitHub: https://github.com/Kattegat-crew/neural-context-plugin

#HermesAgent #NousResearch #AI #AgenticAI #OpenSource
```
