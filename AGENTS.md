# AGENTS.md — Standards, SDD Lifecycle & Skills Contract (`neural-context-plugin`)

This document governs all agentic development (Hermes fleet, Claude Code, Codex, Antigravity/Toallín, OpenCode) on the **`neural-context-plugin`** repository (`/opt/neural-context-plugin`).

---

## 1. Golden Rule: Explicit Human Approval Gate (Jesús Díaz / Jemadiar)

> [!CRITICAL]
> **PROHIBIDO IMPLEMENTAR CUALQUIER CAMBIO SIN APROBACIÓN PREVIA.**  
> Ningún agente o desarrollador automatizado tiene autorización para mutar código, modificar contratos de API, alterar esquemas de datos o agregar dependencias sin antes:
> 1. Informar detalladamente la propuesta técnica y el diseño a **Jesús (`Jemadiar`)**.
> 2. Discutir trade-offs, impacto en latencia y consumo de tokens.
> 3. Obtener su **aprobación explícita** antes de escribir código en la Fase 4.

---

## 2. Spec-Driven Development (SDD) 4-Phase Lifecycle

Este repositorio adopta estrictamente el ciclo de vida SDD formal de 4 fases para cualquier funcionalidad, refactor o integración:

```
[Fase 1: Especificación] ──(Aprobación Jesús)──> [Fase 2: Diseño Técnico] ──(Aprobación Jesús)──> [Fase 3: Tareas] ──> [Fase 4: Implementación]
    01-spec.md                                        02-design.md                                     03-tasks.md            TDD & Verificación
```

1. **Fase 1 (`01-spec.md`)**:
   - Define el problema, casos de uso, actores y comportamiento esperado.
   - Lista explícita de skills locales aplicables en `## 5. Applicable Project Skills`.
   - **GATE 1**: Bloqueo hasta aprobación de Jesús.
2. **Fase 2 (`02-design.md`)**:
   - Arquitectura técnica, diagramas de secuencia, contratos de modelos (Pydantic), interfaces (`SkillRouterBackend`), estructuras SQLite y firmas de hooks.
   - Justificación de KV-cache byte-estable, análisis de fail-closed y presupuesto de tokens (≤ 1.200 tokens).
   - **GATE 2**: Bloqueo hasta aprobación de Jesús.
3. **Fase 3 (`03-tasks.md`)**:
   - Lista atómica de tareas de implementación.
   - Cada tarea debe llevar la skill local que guía su ejecución (`- Skill: .skills/<skill-name>/SKILL.md`).
4. **Fase 4 (Implementación y Verificación TDD)**:
   - Solo se inicia tras la aprobación de las fases 1 a 3.
   - Desarrollo guiado por pruebas (`tests/test_*.py`), ejecución de `pytest` y validación estricta de Hermes CLI.

---

## 3. Catálogo Local de Skills (`.skills/` -> `.work/skills/`)

Antes de implementar o modificar cualquier subsistema, los agentes **DEBEN** leer el archivo `SKILL.md` correspondiente:

| Capacidad Requerida | Skill Local | Propósito y Alcance |
| :--- | :--- | :--- |
| **API de Plugins & Hooks** | [`.skills/hermes-plugin-development/SKILL.md`](file:///opt/neural-context-plugin/.skills/hermes-plugin-development/SKILL.md) | Ciclo de vida de `register(ctx)`, contratos de `pre_llm_call`, registro seguro de tools (`schema`) y CLI. |
| **Seguridad Fail-Closed** | [`.skills/design-fail-closed-gate/SKILL.md`](file:///opt/neural-context-plugin/.skills/design-fail-closed-gate/SKILL.md) | Garantía estructural de que un fallo de red o parseo jamás rompa el turno del agente. |
| **Triage & Extracción LLM** | [`.skills/llm-structured-output/SKILL.md`](file:///opt/neural-context-plugin/.skills/llm-structured-output/SKILL.md) | Extracción cognitiva basada en modelo, tolerancia a errores humanos y salida Pydantic determinista. |
| **Búsqueda Sidecar BM25** | [`.skills/bm25-sidecar-search/SKILL.md`](file:///opt/neural-context-plugin/.skills/bm25-sidecar-search/SKILL.md) | Base SQLite indexada por SHA-256 sin mutar archivos del usuario y ranking léxico en memoria sub-5ms. |
| **Empaquetado Python** | [`.skills/python-packaging/SKILL.md`](file:///opt/neural-context-plugin/.skills/python-packaging/SKILL.md) | Estándares de `pyproject.toml`, UV, sin scripts ocultos ni dependencias innecesarias. |
| **Criterios del Concurso** | [`.skills/hackathon-judge/SKILL.md`](file:///opt/neural-context-plugin/.skills/hackathon-judge/SKILL.md) | 33 requisitos y preflight de validación para el jurado de Nous Research. |
| **Autoría de Skills** | [`.skills/skill-creator/SKILL.md`](file:///opt/neural-context-plugin/.skills/skill-creator/SKILL.md) & [`.skills/skill-auditor/SKILL.md`](file:///opt/neural-context-plugin/.skills/skill-auditor/SKILL.md) | Pautas para la skill empaquetada que acompaña al plugin. |

---

## 4. Estándares Técnicos No Negociables

1. **Cero Listas Rígidas de Palabras**: El análisis semántico en `pre_llm_call` es siempre cognitivo e interpretativo. Prohibido usar diccionarios estáticos de palabras para clasificar tareas o saludos.
2. **Preservación de KV-Cache**: El System Prompt base debe permanecer 100% byte-estable a lo largo de la conversación. Cualquier contexto inyectado debe viajar en el mensaje de usuario efímero mediante `{"context": "..."}`.
3. **Cero Mutación de Archivos de Usuario**: Los archivos `SKILL.md` y de memoria nunca se alteran ni reescriben. Todos los datos derivados viven en `sidecar_index.db` dentro del directorio de datos del plugin.
4. **Validación Obligatoria Pre-Commit**:
   - `pytest /opt/neural-context-plugin/tests -v` ➔ 100% pruebas pasando.
   - `hermes plugins validate /opt/neural-context-plugin` ➔ Cero errores / Cero advertencias.
   - `hermes plugins doctor /opt/neural-context-plugin` ➔ Diagnóstico en estado `OK`.
