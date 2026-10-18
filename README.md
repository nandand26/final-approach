# FinalApproach — Aviation Fleet & Operational Intelligence Agent

A multi-agent aviation operations assistant built with **LangGraph**, **Streamlit**, **OpenSky Network** (physical ADS-B), **FlightAware / NOAA / Tavily** (commercial + web), **OpenAI** (supervisor planning), **Google Gemini** (OpenSky tool planning + final synthesis), and **Mem0** (`mem0ai`) for long-term, cross-session user preferences.

It answers operational questions by combining:

| Source | Role |
|--------|------|
| OpenSky Network (Spoke 1 / FastMCP) | Live ADS-B vectors, airport movements, tracks (keyed by **icao24** hex) |
| FlightAware AeroAPI | Live / near-live commercial flight status, aircraft, codeshares (Spoke 2) |
| NOAA Aviation Weather Center METARs | Conditional weather on delay / hold / divert (Spoke 2) |
| Tavily web search | Unstructured aviation research / route discovery (Spoke 3) |
| Auth SQLite (`auth_sessions.db`) | Google user → owned chat `thread_id`s only |
| LangGraph `SqliteSaver` (`checkpoints.db`) | Durable chat transcript + graph state per `thread_id` |
| Mem0 (`mem0ai`) | Cross-session user preferences / facts scoped by Google `user_id` |
| LangSmith | Optional run tracing (`LANGCHAIN_TRACING_V2`) + tagged turns / tool spans |

Access is gated by **Google OAuth**. Each user owns isolated **chat threads**; LangGraph checkpoints are keyed by `thread_id` so **chat transcript** does not leak across users or chats. Separately, Mem0 stores **preferences** keyed by the same Google `user_id` so facts can follow the person across threads and restarts of a given process’s local Mem0 store.

> **Scope note:** Strong portfolio / demo system (real orchestration, real APIs, real auth boundaries). Not a production ops platform: chat checkpoints are local SQLite (`checkpoints.db`), Mem0 OSS uses a local store under `~/.mem0`, there is no registration→icao24 bridge yet, and there is no formal offline eval suite (LangSmith tracing is wired for live runs).

> **History:** Spoke 1 previously used a local BTS-style SQLite + text-to-SQL path. That was replaced by OpenSky. Old files live under `archive/`.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Frontend ↔ backend connection](#2-frontend--backend-connection-streamlit--langgraph)
3. [Authentication & multi-tenant isolation](#3-authentication--multi-tenant-isolation)
4. [Agent state (`AgentState`)](#4-agent-state-agentstate)
5. [LangGraph topology & Plan-and-Execute](#5-langgraph-topology--plan-and-execute)
6. [Node responsibilities](#6-node-responsibilities)
7. [Spoke 1 — OpenSky Physical Airspace](#7-spoke-1--opensky-physical-airspace)
8. [Spoke 2 & Spoke 3 tools](#8-spoke-2--spoke-3-tools-flightaware-noaa-tavily)
9. [Metro progress UI](#9-metro-progress-ui)
10. [Long-term memory (Mem0)](#10-long-term-memory-mem0)
11. [Observability (LangSmith)](#11-observability-langsmith)
12. [LLM usage map](#12-llm-usage-map)
13. [Codebase structure](#13-codebase-structure)
14. [Setup & run](#14-setup--run)
15. [Example prompts](#15-example-prompts)
16. [Current limitations](#16-current-limitations)
17. [Interview / design talking points](#17-interview--design-talking-points)

---

## 1) System overview

### What the product does

A logged-in user opens a chat thread and asks something like “What’s the status of BA117?” or “Live ADS-B for icao24 4cc2a3?” Behind the UI:

1. Streamlit authenticates the user and selects an owned `thread_id`.
2. The prompt is sent into a LangGraph `StateGraph` with that `thread_id` in config **and** the Google `user_id` in graph state.
3. A **supervisor** retrieves up to 3 Mem0 facts for that `user_id`, then builds a short **plan** of `{worker, task}` steps once per turn.
4. **Spokes** run tools for the popped `current_task` and write scratch into `turn_notes`.
5. Spokes return to the supervisor until the plan is drained.
6. A **synthesizer** turns scratch + recent chat into the final assistant message (and stamps a durable OpenSky vector block when live states exist), then **`memory.add`s** the user/assistant turn into Mem0.
7. Streamlit streams metro progress live (`st.status` + station board), then reloads durable history from the checkpointer.

### Domain split (important)

| Spoke | Handles | Does **not** handle |
|-------|---------|---------------------|
| **Spoke 1 — OpenSky** | `icao24`, altitude / GS / VS, trajectories, ADS-B airport arrivals/departures | Marketed flight numbers (BA117, AI8303), schedules, web research |
| **Spoke 2 — Live ops** | Flight numbers / tails, FlightAware status, codeshares, METAR-on-anomaly | Tavily / web research; pure physical radar without a commercial ident |
| **Spoke 3 — Web search** | Tavily research, schedules, airline/route → flight-number discovery, transit | FlightAware live status, OpenSky ADS-B |

Airline/route asks **without** a flight number (e.g. “Batik Air from India to KL”) typically schedule **Spoke 3 then Spoke 2**. Hybrid asks (BA117 **and** icao24 `4cc2a3`) may schedule OpenSky + LiveOps. The synthesizer is prompted to say when commercial ↔ ADS-B identities **cannot be joined**.

### Runtime layers

```text
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                      │
│  - Google login / logout                                    │
│  - Sidebar thread picker (auth_sessions.db ownership)       │
│  - Chat render from LangGraph get_state(thread_id)          │
│  - stream(updates) + metro board + st.status                │
│  - recursion-limit fallback synthesizer                     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HumanMessage + turn resets
                            │ graph_input.user_id (Google identity)
                            │ config.configurable.thread_id
┌───────────────────────────▼─────────────────────────────────┐
│  LangGraph StateGraph (workflow.py)                         │
│  supervisor ⇄ spokes → synthesizer → END                    │
│  checkpointer: SqliteSaver → checkpoints.db (by thread_id)  │
│  Mem0: search @ supervisor, add @ synthesizer (by user_id)  │
│  LangSmith: run_name/tags on stream + @traceable tools      │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┬───────────────────┐
        ▼                   ▼                   ▼                   ▼
  Spoke 1 OpenSky     Spoke 2 LiveOps     Spoke 3 WebSearch   Synthesizer
  (Gemini → tool      (FA + NOAA;         (Tavily; optional   (Gemini +
   plan → MCP)         no Tavily)          Gemini refine)      OpenSky stamp + Mem0)
```

---

## 2) Frontend ↔ backend connection (Streamlit ↔ LangGraph)

The UI is not a separate API server — Streamlit **is** the presentation process and imports the compiled graph directly.

### Key files

| File | Role |
|------|------|
| `app.py` | Auth gate, sidebar threads, chat I/O, `user_id` stamp, LangSmith `_thread_config`, graph stream, metro paint, recursion fallback |
| `src/ui/metro_progress.py` | Station board, layman cards, OpenSky metrics columns, progress snapshots |
| `src/graph/workflow.py` | Builds/compiles `app_graph` with on-disk `SqliteSaver` (`checkpoints.db`) |
| `src/db/auth_db.py` | Maps Google identity → owned `thread_id`s |
| `src/graph/nodes.py` | Supervisor / spokes / synthesizer + lazy Mem0 search/add |
| `src/tools/live_ops.py` | FlightAware / Tavily / NOAA + LangSmith `@traceable` tool spans |

### Boot sequence (`app.py`)

1. Add project root to `sys.path`, `load_dotenv(.env)`.
2. `init_auth_db()` — ensure `users` / `user_threads` tables exist.
3. `st.set_page_config` + title.
4. **Auth gate:** if `not st.user.is_logged_in`, show Sign in → `st.stop()`.
5. `register_or_get_user(...)` upserts the Google identity into SQLite.
6. Sidebar lists **only this user’s** threads; ensures an active `current_thread_id` in `st.session_state`.

### How chat history is loaded

Durable history is **not** primarily stored in Streamlit `session_state`.

```python
snapshot = app_graph.get_state(config={"configurable": {"thread_id": thread_id}})
messages = snapshot.values.get("messages") or []
```

Each message becomes a Streamlit chat row via `_message_to_chat_row`. Collapsed **Agent steps** expanders are restored from `st.session_state["agent_steps_log"]` snapshots when available.

### What happens on each user prompt

1. Echo the user bubble immediately.
2. Build a **partial** graph input:

```python
graph_input = {
    "messages": [HumanMessage(content=prompt)],  # only the NEW message
    "user_id": current_user_id,  # Google OAuth identity for Mem0 scoping
    "turn_notes": [],   # explicit reset → merge_turn_notes clears scratch
    "plan": [],         # explicit reset → prior turn queue cannot leak
    "next_worker": "",
    "current_task": "", # hop-local sub-query cleared each turn
    "extract_memory": False,  # Memory Gate OFF until supervisor decides
    "final_report": "",
}
```

3. Build `thread_cfg = _thread_config(active_thread_id, user_id=..., entrypoint="streamlit")` — stamps `thread_id`, `recursion_limit=22`, and LangSmith `run_name` / `tags` / `metadata`.
4. Stream inside `st.status("Orchestrating Flight Intelligence Pipeline...")`:

```python
app_graph.stream(
    graph_input,
    stream_mode="updates",
    config=thread_cfg,  # thread_id + LangSmith run_name/tags/metadata
)
```

5. For each node update:
   - `advance_on_node_update(progress, node_name, update)` advances metro lights / cards.
   - Flush `progress.status_log` into the open `st.status` panel.
   - Repaint metro / answer via `render_progress`.
6. Snapshot agent steps → `st.session_state`, then `st.rerun()` so the UI reloads **canonical** messages from the checkpointer.

### Important frontend/backend contracts

| Concern | Contract |
|---------|----------|
| Isolation | Every graph call must include `configurable.thread_id` owned by the logged-in user |
| Preference identity | `graph_input["user_id"]` = auth DB Google identity (for Mem0) |
| History | `SqliteSaver` checkpointer is source of truth for chat text; UI is a view |
| Observability | `_thread_config` sets `run_name="Aviation_Ops_Turn"` + dynamic tags (`entrypoint:…`, `user:…`) |
| Turn scratch | `turn_notes=[]` and `plan=[]` at every new prompt |
| Progress | Streaming `updates` is UX-only; durable answer lands in `messages` / `final_report` |
| Recursion fuse | `GraphRecursionError` → synthesize partial answer from collected notes + `update_state` (also stamps `user_id`) |

### Recursion-limit fallback

Plan-and-Execute can hop supervisor ↔ spokes several times. `recursion_limit` is **22** (headroom for up to 2+2+5 spoke runs plus supervisor visits). If LangGraph still aborts:

1. Append a system closer into collected turn notes.
2. Call `synthesizer_node(partial_state)` **directly**.
3. Persist with `app_graph.update_state(...)`.
4. Mark unfinished metro stations as skipped / partial.

---

## 3) Authentication & multi-tenant isolation

### Google OAuth (Streamlit native)

Configured in `.streamlit/secrets.toml` under `[auth]` (`client_id`, `client_secret`, `redirect_uri`, `cookie_secret`, `server_metadata_url`).

Runtime: `st.login` / `st.logout` / `st.user.is_logged_in` (prefer Google `sub`, fallback `email`).

### Local ownership DB (`auth_sessions.db`)

Implemented in `src/db/auth_db.py` — `users` + `user_threads` (UUID `thread_id` owned by `user_id`).

### Isolation model (four locks)

1. **Auth lock:** no chat UI without Google login.
2. **Ownership lock:** sidebar only queries threads for `current_user_id`.
3. **Chat memory lock:** LangGraph `SqliteSaver` partitions transcript/state by `thread_id` in `checkpoints.db`.
4. **Preference memory lock:** Mem0 search/add always pass the same Google `user_id` (auth DB PK: prefer `sub`, fallback `email`).

> **Durability:** Chat history survives Streamlit / process restarts via on-disk `checkpoints.db` (same machine/workspace). Auth ownership lives in `auth_sessions.db`. Mem0 OSS persists under `~/.mem0` for that OS user. All three are local demo-grade stores — not managed multi-tenant cloud DBs. Logout does not wipe checkpoints.

---

## 4) Agent state (`AgentState`)

Defined in `src/graph/state.py`.

| Field | Reducer / write semantics | Lifetime | Purpose |
|-------|---------------------------|----------|---------|
| `messages` | `add_messages_window` (append, trim to last **10**) | Durable per `thread_id` | User + final assistant answers |
| `user_id` | Replace | Stamped each turn from auth | Google identity for Mem0 (not chat text) |
| `turn_notes` | `merge_turn_notes` (`[]` clears; else append) | Turn-local | Supervisor + spoke scratch |
| `plan` | Replace | Turn-local | Ordered `{"worker","task"}` steps still left to run |
| `next_worker` | Replace | Hop-local | `Spoke_1_OpenSky` \| `Spoke_2_LiveOps` \| `Spoke_3_WebSearch` \| `FINISH` |
| `current_task` | Replace | Hop-local | Sub-query for the active spoke (from plan head `task`) |
| `extract_memory` | Replace | Turn-local (reset `False` each prompt) | Memory Gate: supervisor True → synthesizer may `memory.add` |
| `final_report` | Replace | Turn result | Synthesizer answer string |

`user_id` is **not** how chats were loaded before Mem0. Chat history always came from the checkpointer via `thread_id`. `user_id` was already the PK in `auth_sessions.db`; it is now also passed into graph state so nodes can call Mem0 without re-reading Streamlit session.

### Why `turn_notes` has a custom reset

With a checkpointer, naive “append forever” scratch would leak prior-turn tool output. `app.py` always sends `turn_notes=[]` at prompt start; the reducer treats empty list as **clear**.

### What is *not* top-level state

`flight_ident`, OpenSky args, `weather_context`, etc. live inside spoke JSON embedded in `turn_notes` (and durable OpenSky fields are also stamped into the final `messages` text). Exact tool args are extracted **inside spokes** from `current_task` / messages / notes — the supervisor only queues intent.

Constants: `MESSAGE_WINDOW = 10`, `SPOKE_CAPS = {Spoke_1: 2, Spoke_2: 2, Spoke_3: 5}`.

---

## 5) LangGraph topology & Plan-and-Execute

### Graph wiring (`src/graph/workflow.py`)

```text
ENTRY → supervisor
          │
          ├─(Spoke_1_OpenSky)→ spoke_1_opensky ──┐
          ├─(Spoke_2_LiveOps)→ spoke_2_liveops ──┤
          ├─(Spoke_3_WebSearch)→ spoke_3_websearch┤
          └─(FINISH / other)→ synthesizer → END  │
                                                 │
                     ◄───────────────────────────┘
                     (spokes always return to supervisor)
```

```python
# workflow.py — durable on-disk checkpointer (survives Streamlit restarts)
app_graph = workflow.compile(checkpointer=_build_sqlite_checkpointer())
# → SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
```

### Plan-and-Execute lifecycle (one user message)

**Phase A — turn start** (`turn_notes` empty after app reset):

1. Supervisor runs Mem0 `search(query=latest_prompt, user_id=..., limit=3)` (soft-fail if Mem0 unavailable).
2. Supervisor LLM (or heuristic fallback) builds ordered `plan: list[{worker, task}]` **and** Memory Gate `extract_memory`.
3. Caps from `SPOKE_CAPS` (OpenSky 2 / LiveOps 2 / WebSearch 5); invalid workers dropped.
4. Empty plan → conversational bypass → `FINISH` (`current_task=""`).
5. Non-empty → pop head into `next_worker` + `current_task`, keep remainder in `plan`.

**Phase B — drain queue:**

1. Spoke runs using `current_task` → `[Spoke_X_...]` into `turn_notes` → back to supervisor.
2. Re-sanitize remaining plan against execution counts.
3. Empty / cap → `FINISH` → synthesizer.

This is **not** unbounded ReAct. Short queue + hard caps = predictable cost/latency for demos.

**Supervisor passes intent, not tool kwargs.** It sets `next_worker` + `current_task`. Spokes extract concrete tool args (FlightAware ident, OpenSky tool plan, Tavily query) themselves.

---

## 6) Node responsibilities

### Supervisor (`supervisor_node`)

- **LLM:** OpenAI `supervisor_llm` (`SUPERVISOR_MODEL`, default `gpt-4o-mini`).
- **Mem0 (turn start):** `_search_user_memories` → up to 3 facts injected into the planner system prompt (units, formatting prefs, etc.) and briefly mirrored in `turn_notes`.
- **Structured output:** `SupervisorPlan { plan: list[PlannedStep], extract_memory: bool }` where each `PlannedStep` is `{worker, task}`.
- **Memory Gate:** `extract_memory=true` only for explicit durable prefs (“always use meters…”); `false` for BA117 / ADS-B / weather / filler.
- **Fallback:** `_heuristic_plan` + `_heuristic_extract_memory` when supervisor LLM unavailable.
- Writes routing notes into `turn_notes` for metro / `st.status` (full worker name list for the metro parser).

### Spoke 1 — OpenSky (`spoke_1_opensky_node`)

See [§7](#7-spoke-1--opensky-physical-airspace).

### Spoke 2 — live ops (`spoke_2_liveops_node`)

1. Resolve ident via `_resolve_flight_ident`:
   - `current_task` + latest user message first
   - then `[Spoke_3_WebSearch]` notes from this turn (Tavily-discovered flight numbers)
   - then prior `messages` **only** if the latest prompt has follow-up phrases (“this flight”, “mentioned before”, …)
   - stopword filter blocks acronym false hits (`MEL`, `UTC`, `ILS`, …)
2. If ident found → `flightaware_live_status`; else skip FA with a JSON note.
3. Pack into one `[Spoke_2_LiveOps]` `turn_notes` entry (**no Tavily** — that is Spoke 3).
4. Return to supervisor.

**Spoke 2 does not call an LLM** for tool selection — FA (+ conditional NOAA inside the FA tool).

### Spoke 3 — web search (`spoke_3_websearch_node`)

1. Read `current_task` (fallback: latest user text).
2. **Fast path (0 LLM):** no prior spoke outputs this turn → Tavily on `task` as-is.
3. **Chained path (1 `spoke_llm`):** if Spoke 1/2/3 already wrote notes, refine the Tavily query via `WebSearchRefine` using those facts; soft-fail keeps the original task.
4. Append `[Spoke_3_WebSearch] Query: …` so Spoke 2 can regex-discover idents next.

### Synthesizer (`synthesizer_node`)

- **LLM:** Gemini `spoke_llm`.
- Inputs: `turn_notes`, recent conversation, resolved ident.
- Rules: trust FlightAware `headline`; honest codeshares; don’t invent marketed LH numbers / types from callsign alone; OpenSky gaps ≠ “flight doesn’t exist.”
- **Durable OpenSky stamp:** if Spoke 1 returned live `states`, append a fixed `### OpenSky vector (durable)` block (icao24, callsign, alt, GS, VS, track, lat/lon, on_ground, squawk) so follow-ups can regex those ids from `messages`.
- **Mem0 ingest (gated):** after the final string is ready, `_add_user_memory(...)` runs **only if** `state["extract_memory"]` is True. Otherwise skip ingestion entirely. Failures are logged and **never** crash the graph / stream.
- Writes `final_report` + `AIMessage` into `messages`.

---

## 7) Spoke 1 — OpenSky Physical Airspace

Implementation: `src/mcp_servers/opensky_mcp.py` (+ planning in `nodes.py`).

### Tools

| Tool | Purpose |
|------|---------|
| `get_live_aircraft_telemetry` | Live state vector(s) by `icao24` (+ optional bbox) |
| `get_airport_arrivals` | ADS-B arrival movements (window ≤ 7 days) |
| `get_airport_departures` | ADS-B departure movements (window ≤ 7 days) |
| `get_aircraft_flight_history` | Legs for an `icao24` (window ≤ 30 days) |
| `get_aircraft_trajectory_track` | Waypoint path / track for an `icao24` |

### Planning

1. Prefer hop-local `current_task`, else latest user ask + recent `messages` (**not** from `turn_notes`).
2. **Fast path (0 LLM):** no `icao24` hex and no airport arrival/departure signal → heuristic only. Commercial flight numbers alone (e.g. AI101) cannot key OpenSky, so Spoke 1 returns a structured soft error instead of stalling on Gemini.
3. Otherwise Gemini `with_structured_output(OpenSkyToolPlan)` → tool name + args, hard-capped at **~20s** wall clock (Gemini 429 retries otherwise hung the metro on “Physical Airspace”).
4. Fallback: `_heuristic_opensky_plan` (regex hex / airport keywords). No fake default hex — missing `icao24` yields `{"ok": false, ...}`.
5. `invoke_opensky_tool(plan.tool, **kwargs)` — drops `None`s; missing required args return soft-error JSON instead of crashing the graph.

### Timeouts

| Call | Timeout |
|------|---------|
| OpenSky OAuth token POST | **10s** |
| OpenSky REST GET (`_opensky_get`) | **10s** |
| Gemini OpenSky tool planning | **20s** wall clock, then heuristic |

### Auth

Preference order in `_opensky_get`:

1. OAuth2 client credentials (`OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`) with in-process token cache.
2. Optional username/password basic auth.
3. Anonymous (lowest rate limits).

### Contract note

OpenSky live endpoints key on **icao24 hex**, not marketed flight numbers or registration. FlightAware may return a tail like `9V-SKQ` without a hex — there is **no** reg→icao24 bridge in this repo yet. Hybrid demos should either include an explicit hex or accept “cannot join commercial airframe ↔ ADS-B target.”

State vectors from `/states/all` are **positional lists** (official column order); the MCP layer maps them to named dicts.

---

## 8) Spoke 2 & Spoke 3 tools (FlightAware, NOAA, Tavily)

Tool implementations live in `src/tools/live_ops.py`. Spoke 2 calls FlightAware (+ NOAA on anomalies). Spoke 3 calls Tavily.

### FlightAware AeroAPI (`flightaware_live_status`) — Spoke 2

- HTTP via `requests` + `certifi`.
- Candidate loop: as-given ident, then IATA→ICAO airline rewrite (e.g. `AI8303` → also try `AIC8303`).
- Builds a **headline** (status, route, aircraft, registration, codeshare flags).
- Honest empty / error JSON when AeroAPI fails.

### Tavily (`tavily_aviation_web_search`) — Spoke 3

- Live when `TAVILY_API_KEY` is set.
- Structured mock / “web search not connected” path otherwise (metro UI does not show fake articles).
- Invoked only from `spoke_3_websearch_node` (not from LiveOps).

### NOAA METAR enrichment (conditional, tool-internal) — Spoke 2 / FA path

Triggered **inside** `flightaware_live_status` after formatting — not a separate graph node.

1. Anomaly gate (DELAY / HOLD / DIVERT or delay &gt; 900s).
2. One multi-station METAR GET for origin + dest ICAO.
3. Phase-aware primary station (predeparture → origin; enroute/hold → destination).
4. Attach `weather_context` + headline weather summary; fail soft.

---

## 9) Metro progress UI

File: `src/ui/metro_progress.py`.

### Friendly station names (`NODE_DISPLAY_MAP`)

- `supervisor` → Flight Ops Router  
- Spoke 3 → Web Research Engine  
- Spoke 2 → Commercial Flight Engine  
- Spoke 1 → Physical Airspace Radar Engine  
- `synthesizer` → Intelligence Synthesizer  

### Plan-ordered track (visual)

The metro board is **not** a static Web → Commercial → Radar layout. On the first supervisor hop it rebuilds from the live plan list:

`Router → <spokes in supervisor plan order, unique first-seen> → Synthesizer`

Unplanned spokes are **omitted** (not shown as skipped placeholders). The strip is a **vertical numbered list** (top → bottom = execution order) rendered via `st.html` so Streamlit does not strip layout CSS. A **Plan order:** caption mirrors `plan_workers`. Active station highlights; API / rate-limit style failures can show **warn** (amber). Bare `"429"` digit noise inside Tavily scores is ignored so successful web research is not false-flagged. Spoke 1 live states also drive Altitude / Ground Speed / Vertical Rate / Transponder Hex metrics.

### Streaming

`app.py` calls `advance_on_node_update` on each LangGraph `updates` event and keeps snapshots in `session_state` for collapsed **Agent steps** after `st.rerun()`.

---

## 10) Long-term memory (Mem0)

Implementation lives in `src/graph/nodes.py` (lazy helpers). Dependency: `mem0ai` in `requirements.txt`.

### Two memory systems (do not conflate)

| System | Key | Stores | Survives Streamlit restart? |
|--------|-----|--------|------------------------------|
| LangGraph `SqliteSaver` (`checkpoints.db`) | `thread_id` | Chat transcript + graph fields for that thread | **Yes** (local SQLite file) |
| Mem0 OSS `Memory()` | Google `user_id` | Extracted preferences / facts | **Mostly yes** on local disk (`~/.mem0`) |

### Lifecycle

1. **Lazy init:** `_get_memory()` imports `mem0` and constructs `Memory()` once per process on first use. Import is **not** done in `state.py` because `import mem0` has filesystem side effects (`~/.mem0` mkdir) and would couple app boot to Mem0 health.
2. **Search (supervisor, turn start):** `memory.search(query=..., user_id=..., limit=3)` with fallback to newer `filters={"user_id": ...}` if the SDK rejects top-level `user_id`.
3. **Memory Gate:** supervisor sets `extract_memory` in structured output (default False). Treat Mem0 as an immutable config file — only explicit durable rules flip the gate.
4. **Add (synthesizer, end of turn):** `memory.add([{role:user}, {role:assistant}], user_id=...)` **only when** `extract_memory` is True.
5. Soft no-ops on missing package / empty `user_id` / API errors / gate=False — the aviation graph still completes.

### Infra note

First Mem0 hit in a process pays cold-start (`Memory()` → embedder + LLM client + local vector store). Later users/turns in that process reuse the **same** engine; facts remain partitioned by `user_id`. There is no per-user separate DB file and no delete/forget UI yet.

---

## 11) Observability (LangSmith)

Optional live tracing — not a full offline eval suite. Dependencies: `langsmith` in `requirements.txt`. Env is loaded in `app.py` with `load_dotenv(..., override=True)` and LangSmith aliases are mirrored (`LANGCHAIN_*` → `LANGSMITH_*` when needed).

### Turn-level tagging (`app.py`)

Every graph stream / `update_state` uses `_thread_config(...)`:

| Config field | Value |
|--------------|--------|
| `run_name` | `Aviation_Ops_Turn` |
| `tags` | `entrypoint:{streamlit\|api\|…}`, optional `user:{user_id}` |
| `metadata` | `thread_id`, `entrypoint`, optional `user_id` |
| `configurable.thread_id` | Chat isolation key (unchanged) |
| `recursion_limit` | `22` |

Tags are built dynamically from `entrypoint` so a future API/notebook caller can pass `entrypoint="api"` without hardcoding UI labels.

### Tool-level spans (`src/tools/live_ops.py`)

| Span | Notes |
|------|--------|
| `FlightAware_AeroAPI_GET` | `@traceable(run_type="tool")`; `hide_inputs` redacts `api_key` |
| `NOAA_METAR_Batch_GET` | `@traceable(run_type="tool")` |

### Enable tracing

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com  # EU/US hosts if your org is there
LANGCHAIN_API_KEY=lsv2_pt_...   # or LANGSMITH_API_KEY
LANGCHAIN_PROJECT=FinalApproach # or LANGSMITH_PROJECT
```

With tracing on, each Streamlit turn should appear in the LangSmith project as `Aviation_Ops_Turn` (plus nested model/tool runs). If you see 403 on `/runs/multipart` with a valid key, you're almost always on the wrong regional endpoint (US default vs EU/APAC) — not a bug in the graph wiring.

---

## 12) LLM usage map

| Component | Model path | Uses LLM? |
|-----------|------------|-----------|
| Supervisor plan | OpenAI via `get_supervisor_llm()` | Yes (`SupervisorPlan`); heuristic fallback |
| Mem0 fact extract / embed (OSS default) | OpenAI via `Memory()` | Yes on `add` / embed on `search` |
| Spoke 1 tool plan | Gemini via `get_spoke_llm()` | Only when icao24/airport signal present (`OpenSkyToolPlan`, ≤20s); else heuristic |
| OpenSky HTTP tools | — | No (10s timeouts) |
| Spoke 2 FA / NOAA | — | No |
| Spoke 3 Tavily | — | No tool LLM; optional Gemini `WebSearchRefine` when chaining |
| Synthesizer | Gemini via `get_spoke_llm()` | Yes (+ deterministic OpenSky durable stamp) |

Env overrides: `SUPERVISOR_MODEL` (default `gpt-4o-mini`), `SPOKE_MODEL` (default **`gemini-2.5-flash`**). Models are built lazily in `models.py` and **rebuild when the env model name changes** so a Streamlit session does not keep a stale Gemini id after you edit `.env` (still restart once after pulling this change). Mem0 OSS defaults also expect `OPENAI_API_KEY`; optional `MEM0_LLM_MODEL=gpt-4o-mini` avoids Mem0’s `gpt-5-mini` + `temperature=0.1` rejection.

---

## 13) Codebase structure

```text
aviation_agent/
├── app.py                      # Streamlit UI + graph stream + auth + LangSmith config
├── README.md
├── requirements.txt            # mem0ai, langgraph-checkpoint-sqlite, langsmith, …
├── .env / .env.example
├── .streamlit/secrets.toml     # Google OAuth (local)
├── auth_sessions.db            # Runtime: users ↔ threads
├── checkpoints.db              # Runtime: LangGraph SqliteSaver chat/state
├── archive/                    # Former BTS SQLite / MCP (not on hot path)
├── notes.md                    # Learning notes (local; often gitignored)
└── src/
    ├── db/
    │   └── auth_db.py          # User + thread ownership
    ├── graph/
    │   ├── models.py           # get_supervisor_llm / get_spoke_llm (env-fresh)
    │   ├── state.py            # AgentState (+ plan/{worker,task}, current_task, SPOKE_CAPS)
    │   ├── nodes.py            # Supervisor, 3 spokes, synthesizer, Mem0 helpers
    │   ├── supervisor.py       # Thin re-export of supervisor_node
    │   └── workflow.py         # StateGraph (3 spokes) + SqliteSaver(checkpoints.db)
    ├── mcp_servers/
    │   └── opensky_mcp.py      # OpenSky tools + OAuth + 10s timeouts + invoke dispatch
    ├── tools/
    │   └── live_ops.py         # FlightAware + Tavily + NOAA (+ @traceable)
    └── ui/
        └── metro_progress.py   # Plan-ordered metro board + layman cards
```

### Streamlit UX notes (`app.py`)

- Brand **FinalApproach ✈️** lives in the sidebar (and on the login page); the main chat pane opens with **“Hey, where we off to today?”**
- Empty chats center the `st.chat_input` dock under the greeting; the first send animates it down to the normal bottom dock.
- **Login / + New Chat** always opens a **draft** `thread_id` in session only. A row is written to `auth_sessions.db` **only when the user sends the first message** (title = truncated prompt). Empty drafts are never saved.

---

## 14) Setup & run

### Prerequisites

- Python 3.10+ recommended
- Google Cloud OAuth client for Streamlit redirect URI
- API keys for the paths you want live (OpenAI, Gemini, FlightAware, OpenSky OAuth; Tavily optional)
- OpenAI key also used by Mem0 OSS defaults for embedding / fact extraction
- Optional LangSmith API key if you want live tracing in the LangSmith UI (use the **regional** API host)

### Install

```bash
cd aviation_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment (`.env`)

```bash
OPENAI_API_KEY=...              # supervisor + Mem0 OSS defaults
GOOGLE_API_KEY=...
FLIGHTAWARE_API_KEY=...
TAVILY_API_KEY=...                 # optional
OPENSKY_CLIENT_ID=...
OPENSKY_CLIENT_SECRET=...
SUPERVISOR_MODEL=gpt-4o-mini
SPOKE_MODEL=gemini-2.5-flash
MEM0_LLM_MODEL=gpt-4o-mini         # optional; avoid gpt-5-mini with Mem0 temp=0.1

# Optional LangSmith (leave TRACING_V2 unset/false to disable)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://apac.api.smith.langchain.com  # required if your org is APAC
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=FinalApproach
```

### Streamlit auth (`.streamlit/secrets.toml`)

Provide the `[auth]` block for Streamlit native Google login. Do **not** commit secrets.

### Run

```bash
streamlit run app.py
```

On first graph compile, `workflow.py` creates/opens `checkpoints.db` beside `app.py`. Chat threads then persist across Streamlit restarts on that machine. After changing `SPOKE_MODEL` / `SUPERVISOR_MODEL`, a full Streamlit restart is safest the first time (getters rebuild on name change thereafter).

---

## 15) Example prompts

**Commercial / codeshare (Spoke 2)**

- `What's the live status of AI8303? Include route, aircraft type, and codeshare operator if any.`
- Follow-up: `Status of the flight I mentioned before?` (gated history walk should resolve the ident)

**Airline / route without flight number (Spoke 3 → Spoke 2)**

- `Is there a Batik Air flight from India to KL today? What's the live status if you find a number?`

**Web research only (Spoke 3)**

- `Research typical aircraft types on the Delhi–Singapore route and any recent disruption news.`

**Physical airspace (OpenSky)**

- `Pull live OpenSky ADS-B telemetry for icao24 4cc2a3 — altitude, ground speed, vertical rate, track, on_ground.`

**Hybrid (multiple spokes + disambiguation)**

- `How’s British Airways flight BA117 looking right now — on time, what plane, what route? Also pull live radar for aircraft code 4cc2a3 and tell me straight: same plane as BA117, or a different one?`
- Follow-up: `Ignore the radar track from earlier. Just for BA117 — with Heathrow weather chaos lately, does this flight look delayed or pretty normal?`

**Conversational bypass**

- `Rewrite your last answer as three bullet points for a dispatcher.`

**Cross-session preferences (Mem0)**

- `Always report altitudes in meters and keep answers in tight dispatcher bullets.`
- Later (even in a **new** chat thread): `Status of BA117?` — supervisor should retrieve the preference facts and plan/format accordingly.

---

## 16) Current limitations

1. **Checkpoints are local SQLite** — `SqliteSaver` survives process restart on the same workspace file (`checkpoints.db`); not a managed multi-tenant / multi-host store.
2. **Mem0 is OSS/local** — shared store under `~/.mem0`; first hit pays cold-start; no user-facing delete/forget yet; not the managed Mem0 cloud unless you swap clients.
3. **No registration → icao24 bridge** — FlightAware tails (e.g. `9V-SKQ`) do not auto-unlock OpenSky telemetry.
4. **OpenSky ≠ commercial schedules** — cannot enumerate “all flights DEL→SIN.”
5. **State vectors are brittle positional APIs** — mitigated by length checks + named mapping in MCP.
6. **Spoke 2 tool choice is fixed** (FA if ident; NOAA on anomaly) — not Gemini-planned like Spoke 1. Spoke 3→2 ident handoff is still regex on notes, not a structured `discovered_ident` field.
7. **NOAA enrichment is heuristic** — METAR ≠ proven root cause of delay.
8. **LangSmith is live tracing, not a formal eval suite** — no golden-set harness / CI assertions yet; wrong regional endpoint → 403; bad keys fail soft.
9. **Metro Agent-steps snapshots** are session-scoped extras (chat text is `SqliteSaver`-backed).
10. **Offline heuristic planner** has some demo-tuned route phrases; live routing should prefer the supervisor LLM.
11. **Gemini free-tier quotas are per-model** — exhausting `gemini-3.6-flash` does not mean `gemini-2.5-flash` is dead, but the process must actually load the new `SPOKE_MODEL`.
12. **OpenSky without icao24 soft-fails fast** — flight-number-only radar asks will not invent a hex.

---

## 17) Interview / design talking points

- Why Plan-and-Execute with a capped `{worker, task}` queue beats unbounded tool loops for demo cost control.
- Why `messages` (durable per `thread_id` in `SqliteSaver`) vs `turn_notes` (scratch) vs Mem0 (prefs per `user_id`).
- Why OpenSky vs FlightAware vs Tavily is a **three-way domain split**.
- Why Spoke 3 is fast-path (0 LLM) vs chained refine (1 Gemini) for cost control.
- Why Spoke 1 uses structured `OpenSkyToolPlan` (with icao24 gate + 20s planning cap + 10s HTTP) while Spoke 2 stays tool-only and Spoke 3 only optionally refines.
- Why the metro board is rebuilt from the supervisor plan list (not a static Web→FA→Radar track).
- Why gated history walk + stopwords for FlightAware idents (bleed vs follow-up).
- Why synthesizer must be allowed to say **“cannot join”** commercial airframe ↔ ADS-B hex.
- Why durable OpenSky stamps in `messages` matter for follow-up regex / memory.
- Why NOAA is conditional, single-RTT, phase-aware.
- Why Mem0 import stays lazy in nodes (import side effects / boot isolation) and failures soft-noop.
- Why `SqliteSaver` + `check_same_thread=False` fits Streamlit (multi-thread touches, still local file).
- Why LangSmith tags are dynamic (`entrypoint` / `user`) and FlightAware uses `hide_inputs` for the API key.
- Honest gaps: local SQLite / Mem0 stores, no reg→icao24 map, no offline eval suite, regex ident handoff.

---

## License / status

Personal / portfolio project. Not affiliated with FlightAware, OpenSky Network, NOAA, Google, OpenAI, Tavily, Mem0, or any airline.
