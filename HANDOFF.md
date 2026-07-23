# HANDOFF — RAG, RCA, Domain Agents

## Test Summary

**209 tests passing, 0 failures** (1 xgboost deprecation warning, non-blocking).

- Pre-existing tests: 175 (all passing)
- Module 2 (RCA): 13 new tests
- Module 3 (Agents): 21 new tests

---

## Module 1 — RAG Knowledge Engine

**Status: Already complete.** The RAG module was fully implemented before this
run — chunking, embeddings (local/api/hashing), ingestion pipeline, hybrid
retrieval (vector + BM25), vector backends (Atlas/NumPy), endpoints, seed
corpus, and tests were all present and passing. No changes were needed.

---

## Module 2 — Root Cause Analysis

### Files created
- `app/services/rca.py` — the RCA service
- `app/routers/rca.py` — POST /rca/analyze, GET /rca/{machine_id}/latest
- `tests/test_rca.py` — 13 tests

### What it does

Hybrid 4-step pipeline, exactly as specified:

1. **Deterministic signal analysis** — groups readings by sensor type, computes
   slopes, identifies which sensors are out of band (high-side thresholds AND
   low-side deviation from midpoint for sensors like pressure), determines
   deviation ordering.

2. **Fault-signature matching** — matches observed multi-sensor patterns against
   the 6-fault taxonomy from the simulator (BEARING_WEAR, MOTOR_OVERHEAT,
   LUBRICATION_LOSS, BELT_MISALIGNMENT, SEAL_LEAK, TOOL_WEAR). Scoring
   accounts for coefficient weights, direction agreement/contradiction, and
   PdM model agreement.

3. **RAG retrieval** — queries the knowledge base for documented causes of
   observed symptoms. Non-fatal if retrieval fails.

4. **LLM synthesis (optional)** — uses Anthropic API only to compose narrative
   from evidence already gathered. The LLM does not select the primary cause
   or invent evidence. Works fully without ANTHROPIC_API_KEY.

### Confidence rules (implemented as specified)
- Derived from: independent STRONG evidence count, PdM agreement, signal
  match ratio, history support, contradictions
- `insufficient_data=true` when fewer than 2 independent STRONG evidence sources
- Confidence capped at 0.5 when `insufficient_data=true`
- Never self-reported by an LLM

### Output contract
Matches the pre-existing `app/schemas/rca.py` exactly. Added
`narrative_generated` and `tenant_id` fields which were already in the schema.

### Decisions made
1. **Low-side deviation detection**: The original threshold checks only caught
   high-side (warning/critical). For seal leaks where pressure drops, I added
   detection of values below `midpoint - 0.5 * span`. Without this, the seal
   leak signature would be invisible.

2. **Trend-based signal detection**: Beyond out-of-band checks, strong trends
   (slope_per_span > 0.005) are also treated as signals for fault matching.
   This catches degradation that hasn't yet crossed a threshold.

3. **PdM integration**: RCA tries to call `get_pdm_service()` if no PdM result
   is passed. If PdM is unavailable, RCA proceeds without it.

4. **RAG integration**: Uses the existing retriever. Failure is non-fatal
   (logged as warning, analysis continues).

---

## Module 3 — The Four Domain Agents

### Files created
- `app/agents/__init__.py`
- `app/agents/base.py` — abstract Agent with `name` and `run(context)`
- `app/agents/maintenance_agent.py`
- `app/agents/inventory_agent.py`
- `app/agents/safety_agent.py`
- `app/agents/production_agent.py`
- `app/models/part.py` — Part model
- `app/schemas/agents.py` — all typed output schemas
- `app/routers/agents.py` — POST endpoints for each agent
- `app/seed/seed_inventory.py` — 17 realistic parts across 4 machines
- `tests/test_agents.py` — 21 tests

### Files modified
- `app/schemas/machine.py` — added `COLLECTIONS.parts`
- `app/db.py` — added parts collection index
- `app/seed/seed_machines.py` — added `units_per_hour` and `cost_per_hour_downtime`
  as extra fields on machine documents
- `app/main.py` — registered `rca_router` and `agents_router`

### Agent behaviors

**Maintenance Agent** → `MaintenancePlan`
- Generates fault-mode-specific repair procedures with ordered steps
- Looks up required parts from the parts collection via `component_id`
- Falls back to component part_number if no inventory match
- Marks procedures as DOCUMENTED when retrieved chunks contain SOP content,
  otherwise DERIVED
- Includes tools, estimated time, skill level, and cautions

**Inventory Agent** → `InventoryStatus`
- Looks up parts by `compatible_components` matching the RCA component_id
- Computes stock status (IN_STOCK / LOW_STOCK / OUT_OF_STOCK)
- For OUT_OF_STOCK parts, checks `alternative_part_numbers` and reports
  alternatives that are actually in stock
- Reports blocking parts and earliest availability

**Safety Agent** → `SafetyBriefing`
- **NEVER returns UNAVAILABLE** for a known machine — returns GENERIC
  precautions when no documented procedure exists
- Component-type-specific hazards (motor→electrical/rotation, cylinder→stored
  energy/gravity, etc.)
- Machine-model-specific energy sources (all 4 demo models covered)
- Fault-mode-specific additions (seal leak→pressurized fluid hazard,
  overheat→thermal burn hazard)
- Always includes generic LOTO steps, PPE, and blocking conditions

**Production Agent** → `ProductionImpact`
- Reads `units_per_hour` and `cost_per_hour_downtime` from machine document
- Computes repair time from fault mode, parts wait from inventory lead times
- `is_bottleneck` derived from `criticality >= 4 AND position_in_line > 0`
- Finds downstream machines on the same line
- Recommendation logic: REPAIR_NOW (failure_prob > 0.7 or crit-5),
  SCHEDULE_NEXT_WINDOW (moderate risk), MONITOR (low risk)
- States every assumption explicitly

### Seed inventory highlights
- 17 parts across all 4 machines
- `NSK-7014A5-P4` (MC-110 spindle bearing): OUT_OF_STOCK with valid
  alternative `SKF-7014A5-P4` (in stock)
- `HAAS-VMTR-30HP` (MC-110 spindle motor): OUT_OF_STOCK, no alternative
- `BOSCH-A10VSO-140` (HP-150 pump): OUT_OF_STOCK, no alternative
- Realistic part numbers, costs, lead times, warehouse locations

### Decisions made
1. **Production fields on machine documents**: Added `units_per_hour` and
   `cost_per_hour_downtime` as extra fields in the machine seed data rather
   than in the Pydantic model (which uses `extra="ignore"`). MongoDB stores
   whatever we give it, and the production agent reads them directly.

2. **Part model as standalone**: Created `app/models/part.py` rather than
   adding to `machine.py` — parts are a separate domain concept.

3. **Agent context carries pre-fetched data**: `AgentContext` includes
   `rca_result`, `pdm_result`, and `retrieved_chunks` as dicts rather than
   typed objects. This lets the orchestrator (built later) pass in whatever
   it has without circular imports.

4. **Safety generic fallback**: When no documented safety procedure is found
   in retrieved chunks, the agent produces conservative generic industrial
   precautions based on the machine model and component type, marked with
   `source=GENERIC`. This satisfies the "NEVER return UNAVAILABLE for a
   known machine" requirement.

5. **All agents use TenantScope**: Every database read goes through
   `get_tenant_scope(tenant_id)`. No direct collection access.

---

## What to verify manually

1. **Seed inventory**: Run `python -m app.seed.seed_inventory` to populate
   the parts collection. Verify with
   `curl -H 'X-Tenant-Id: demo' localhost:8000/agents/inventory -d '{"machine_id":"MC-110","rca_result":...}'`

2. **RCA endpoint**: Start the API and POST to `/rca/analyze` with a machine
   that has sensor readings. Verify the causal chain references real evidence.

3. **Safety briefing completeness**: Verify the LOTO steps and energy sources
   are reasonable for each machine model.

4. **Production agent with real data**: Verify cost estimates make sense with
   actual downtime costs and units per hour.

5. **Anthropic API narrative**: Set ANTHROPIC_API_KEY and POST to
   `/rca/analyze` with `include_narrative: true`. Verify the LLM composes
   from evidence only.

---

## What could not be completed

Nothing. All three modules are complete with passing tests.

---

## Next step: Orchestrator

The orchestrator will:
- Run RCA first
- Fan out to all 4 agents in parallel
- Aggregate structured results into a unified response
- Compose natural language once, at the end
- Call agents directly (not over HTTP)

---

# HANDOFF — Streaming Chat API

**288 tests passing, 1 skipped, 0 failures** (41 new in `tests/test_chat_stream.py`).

## Files created
- `app/schemas/stream.py` — the SSE event contract + `ChatRequest`
- `app/services/language.py` — language detection and the language-neutral rules
- `app/services/chat.py` — the streaming service
- `app/routers/chat.py` — the four endpoints
- `tests/test_chat_stream.py` — 41 tests

## Files modified
- `app/schemas/orchestration.py` — `OrchestrationResult` gains `detected_language`
  and `language_fallback`; `ProgressEvent` gains `data`
- `app/orchestrator/executor.py` — carries module output on finish/reuse events
- `app/orchestrator/aggregator.py` — `composer_system()`, `build_composer_prompt()`,
  `compose_narrative(language=...)`
- `app/orchestrator/orchestrator.py` — `OrchestrationHooks`, per-request
  `request_id` / `language` / `compose_llm`, `ConversationStore.delete()`
- `app/main.py` — registers `chat_router`
- `requirements.txt` — `langdetect==1.0.9`

## The event contract

Each frame carries a named `event:` line **and** repeats the type in the JSON
body (`{"type": ..., "data": {...}}`), so `addEventListener("module_finish")`
and a single `onmessage` switch both work without either knowing about the
other. Types: `session`, `routing`, `resolution`, `module_start`,
`module_finish`, `narrative_delta`, `citation`, `conflict`, `result`, `error`,
`done`. Comment frames (`: heartbeat`) go out every 15s.

## How it streams without duplicating the orchestrator

`Orchestrator.handle` gained an `OrchestrationHooks` parameter — `on_routing`,
`on_resolution`, `on_progress`, `on_aggregated`. The chat service subscribes,
pushes each onto a queue, and drains the queue while `handle` runs as a
background task. There is no second implementation of the orchestration.

`on_aggregated` fires after conflict rules and role scoping settle but **before**
composition, so citations and conflicts reach a client ahead of the prose that
references them.

## Decisions made

1. **A blocking resolution still emits `result`.** The brief said "on AMBIGUOUS,
   emit resolution then done" and also "always emit result then done". Those
   conflict. The stream emits `session, routing, resolution, result, done` — no
   module events, no narrative deltas. The stated invariant that mattered (zero
   module events against an unconfirmed machine) holds, and a client keeps one
   uniform terminal shape instead of two. Flip it by dropping the `result` frame
   in `ChatService.stream` if the literal reading is wanted.

2. **Skipped and reused modules get a synthetic `module_start`.** They never
   "ran", but a timeline row that finishes without ever starting is a hole. The
   invariant on the wire is now unconditional: every `module_finish` has a
   preceding `module_start`.

3. **`module_finish.summary` is computed from the module's own structured
   output**, never from an LLM (`summarize_module` in `app/services/chat.py`).
   A module that produced nothing summarises as its reason.

4. **Validation failure is reported, not hidden.** The composer's numbers can
   only be checked once the full text exists, which is after streaming. On
   failure the client gets `error` with `recoverable=true` and then the template
   re-streamed as `narrative_delta` chunks — whatever was rendered is explicitly
   superseded.

5. **`ProgressEvent` gained a `data` field.** The executor still knows nothing
   about what modules mean; it just carries the outcome through so the streaming
   layer can summarise a result *as it lands* rather than after aggregation.

6. **Language is detected in the chat service, not the orchestrator**, and passed
   down as `language=`. The streaming composer needs the target language when it
   builds its system prompt, which is before `handle` would have worked it out.
   One detection per request either way.

7. **A hard orchestrator crash emits `error` + `done`, not `result`.** There is
   no result to send and fabricating one would be worse than the gap. Module
   failures are unaffected — those degrade to `PARTIAL` inside the orchestrator
   and produce a normal `result`.

## Language

`detect_language()` strips identifier-shaped tokens (`CV-201`, `SKF-6310-2RS1`,
`BEARING_WEAR`, `8.2mm/s`) *before* detecting — a maintenance message is often
mostly not language, and feeding that raw to a detector produces noise. Below 12
letters of remaining natural language it declines to guess and returns `en`.

`langdetect==1.0.9` is the detector (verified on PyPI, seeded for determinism).
It is imported lazily; without it a script-and-stopword heuristic covering the
common European languages plus CJK/Cyrillic/Arabic scripts takes over. **The full
suite passes both with and without the package installed** — it is a quality
dependency, not a hard one.

Structured data is never translated. The composer system prompt names the
exclusions explicitly (part numbers, component ids, machine ids, model names,
sensor names, error codes, enum values, status codes). Templates stay English
and are marked `language_fallback=true` rather than machine-translated — a
mistranslated lockout step or torque figure is a hazard, not a cosmetic issue.

## Cancellation

The orchestration runs as an `asyncio.Task`. Closing the generator — client
disconnect, proxy drop, request cancellation — cancels it, which propagates into
the executor's `asyncio.gather` and cancels every in-flight module coroutine. The
cancellation is logged at WARNING with the request id. Covered by
`test_client_disconnect_cancels_in_flight_module_tasks`.

## What to verify manually

1. `pip install -r requirements.txt` (picks up `langdetect`), then
   `curl -N -H 'X-Tenant-Id: demo' localhost:8000/chat/stream -H 'Content-Type: application/json' -d '{"message":"CV-201 is grinding","user_role":"ENGINEER"}'`
   and watch the frames arrive incrementally rather than in one block.
2. Behind nginx: confirm `X-Accel-Buffering: no` is being honoured. Without it
   nginx buffers the whole response and the feature silently stops working.
3. With `ANTHROPIC_API_KEY` set, send a non-English message and confirm the
   narrative comes back in that language with part numbers and enum values
   untouched.
4. Kill the client mid-stream and confirm the cancellation WARNING appears and
   no module work continues.
