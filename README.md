# FinalApproach — Aviation Fleet & Operational Intelligence Agent

A multi-agent aviation operations assistant built with **LangGraph**, **Streamlit**, **OpenSky Network** (physical ADS-B), **FlightAware / NOAA / Tavily** (commercial + web), **OpenAI** (supervisor planning), **Google Gemini** (OpenSky tool planning + final synthesis), and **Mem0** (`mem0ai`) for long-term, cross-session user preferences.

It answers operational questions by combining:

| Source | Role |
|--------|------|
| OpenSky Network (`opensky` / FastMCP) | Live ADS-B vectors, airport movements, tracks (keyed by **icao24** hex) |
| FlightAware AeroAPI (`flightaware`) | Live / near-live commercial flight status, aircraft, codeshares |
| NOAA Aviation Weather Center METARs | Conditional weather on delay / hold / divert (**inside** FlightAware, not a spoke) |
| Tavily web search (`tavily`) | Unstructured aviation research / route discovery / NOTAMs |
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
5. [LangGraph topology & parallel fan-out](#5-langgraph-topology--parallel-fan-out)
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
3. A **supervisor** retrieves up to 3 Mem0 facts for that `user_id`, then chooses `spokes_to_call` (parallel `Send` targets) and `context_priority_order` (how the synthesizer packs notes).
4. Selected **spokes** run **concurrently** (LangGraph `Send` fan-out). Each writes `{spoke, content}` into `turn_notes`.
5. All spoke hops **fan in** to the synthesizer (they do not return to the supervisor).
6. The **synthesizer** sanitizes/reorders notes, writes one assistant message (plus a durable OpenSky vector block when live states exist), then **`memory.add`s** the turn into Mem0 only if the Memory Gate is open.
7. Streamlit streams metro progress live (`st.status` + station board), then reloads durable history from the checkpointer.

### Domain split (important)

| Spoke id | Handles | Does **not** handle |
|----------|---------|---------------------|
| **`opensky`** | `icao24`, altitude / GS / VS, trajectories, ADS-B airport arrivals/departures | Marketed flight numbers (BA117, AI8303), schedules, web research |
| **`flightaware`** | Flight numbers / tails, FlightAware status, codeshares, METAR-on-anomaly | Tavily / web research; pure physical radar without a commercial ident |
| **`tavily`** | Tavily research, schedules, airline/route → flight-number discovery, transit, NOTAMs | FlightAware live status, OpenSky ADS-B |

NOAA is **not** a fourth spoke. `_enrich_formatted_with_noaa` runs inside `flightaware_live_status` only when AeroAPI reports delay / hold / divert.

Airline/route asks **without** a flight number (e.g. “Batik Air from KL to BLR”) schedule **`tavily` and `flightaware` in parallel**. That is a known gap vs the old sequential hop — see [§16](#16-current-limitations). Hybrid asks (BA117 **and** icao24 `4cc2a3`) may fan out OpenSky + FlightAware together. The synthesizer is prompted to say when commercial ↔ ADS-B identities **cannot be joined**.

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
│  supervisor ─Send()─► spokes (parallel) ─► synthesizer → END│
│  checkpointer: SqliteSaver → checkpoints.db (by thread_id)  │
│  Mem0: search @ supervisor, add @ synthesizer (by user_id)  │
│  LangSmith: run_name/tags on stream + @traceable tools      │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┬───────────────────┐
        ▼                   ▼                   ▼                   ▼
  opensky             flightaware         tavily              Synthesizer
  (Gemini → tool      (FA + NOAA-on-      (Tavily;            (reorder notes
   plan → MCP)         delay; no Tavily)   0-LLM fast path)    + Gemini + Mem0)
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
    "plan": [],         # filled by the supervisor: one {worker, task} per spoke
    "spokes_to_call": [],
    "context_priority_order": [],
    "intent": "",
    "flight_number": None,
    "search_query": None,
    "next_worker": "",
    "current_task": "",
    "extract_memory": False,  # Memory Gate OFF until supervisor decides
    "final_report": "",
}
```

3. Build `thread_cfg = _thread_config(active_thread_id, user_id=..., entrypoint="streamlit")` — stamps `thread_id`, `recursion_limit=10`, and LangSmith `run_name` / `tags` / `metadata`.
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

The longest happy path is discovery → interrupt → supervisor → parallel spokes → synthesizer. `recursion_limit` is **10**. If LangGraph still aborts:

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
| `turn_notes` | `merge_turn_notes` (`[]` clears; else `operator.add`) | Turn-local | List of `{spoke, content}` dicts (parallel spokes append) |
| `plan` | Replace | Turn-local | One `{"worker","task"}` per id in `spokes_to_call`. Empty only for `direct_chat` |
| `intent` | Replace | Turn-local | `telemetry_known` \| `telemetry_discovery` \| `general_knowledge` \| `direct_chat` |
| `flight_number` | Replace | Turn-local | Marketed ident (`BA117`) once the user typed or picked one. Otherwise `None` |
| `search_query` | Replace | Turn-local | Tavily query for discovery or research. `None` when no web search is needed |
| `spokes_to_call` | Replace | Turn-local | Fan-out targets: `opensky` \| `flightaware` \| `tavily` |
| `context_priority_order` | Replace | Turn-local | Synthesizer pack order. One slot per spoke; all of that spoke's notes are grouped there |
| `next_worker` | Replace | Hop-local | First spoke id or `FINISH` (metro / bypass) |
| `current_task` | Replace | Hop-local | Default sub-query; each spoke prefers its own `plan` task |
| `extract_memory` | Replace | Turn-local (reset `False` each prompt) | Memory Gate: supervisor True → synthesizer may `memory.add` |
| `final_report` | Replace | Turn result | Synthesizer answer string |

`user_id` is **not** how chats were loaded before Mem0. Chat history always came from the checkpointer via `thread_id`. `user_id` was already the PK in `auth_sessions.db`; it is now also passed into graph state so nodes can call Mem0 without re-reading Streamlit session.

### Why `turn_notes` has a custom reset

With a checkpointer, naive “append forever” scratch would leak prior-turn tool output. `app.py` always sends `turn_notes=[]` at prompt start; the reducer treats empty list as **clear**.

### What is *not* top-level state

OpenSky args, `weather_context`, and the resolved ident string inside a tool payload live in spoke JSON embedded in `turn_notes` (durable OpenSky fields are also stamped into the final `messages` text). `flight_number` and `search_query` are top-level fields. Each spoke reads its instruction from its own `plan` task, then `current_task`, then the latest user message.

Constants: `MESSAGE_WINDOW = 10`, `VALID_FANOUT_SPOKES = (opensky, flightaware, tavily)`, `SPOKE_CAPS = 2` each. Fan-out still Sends each spoke once per turn. The cap is the ceiling if a plan lists the same worker more than once. The supervisor and the synthesizer both read all 10 stored messages, with no per-message character cut.

---

## 5) LangGraph topology & parallel fan-out

### Graph wiring (`src/graph/workflow.py`)

```text
ENTRY → supervisor
          │
          ├─ intent=telemetry_known
          │     Send(each spoke the supervisor chose) ──► synthesizer → END
          │
          ├─ intent=telemetry_discovery
          │     tavily → clarification_node  (interrupt)
          │                 │  user picks ident
          │                 └── Command(goto="supervisor")   # no static edge
          │
          ├─ intent=general_knowledge → tavily → synthesizer → END
          └─ intent=direct_chat → synthesizer → END
```

Path 1 spokes run **at the same time**. The supervisor chooses which of OpenSky, FlightAware, and Tavily are in that set. LangGraph waits for every `Send` branch, then merges `turn_notes` before the synthesizer. Path 2 is Tavily, then the human picks an ident. It does **not** call FlightAware or OpenSky on that hop.

```python
# workflow.py — durable on-disk checkpointer (survives Streamlit restarts)
app_graph = workflow.compile(checkpointer=_build_sqlite_checkpointer())
# → SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))
```

### Fan-out / fan-in lifecycle (one user message)

1. Supervisor runs Mem0 `search(...)` (soft-fail if unavailable).
2. Supervisor LLM (or heuristic) returns `SupervisorPlan`:
   - `intent` — `telemetry_known` | `telemetry_discovery` | `general_knowledge` | `direct_chat`
   - `flight_number` / `search_query`
   - `spokes_to_call` — Send targets, or `[tavily]` for discovery / research
   - `plan` — exactly one `{worker, task}` for every id in `spokes_to_call`. A mismatch is rejected
   - `context_priority_order` — synthesizer pack order. Missing or invalid lists fall back to FlightAware, then Tavily, then OpenSky
   - Memory Gate `extract_memory`
3. A known ident keeps the supervisor's spoke list and those spokes run in parallel. An unknown ident and general research are forced to Tavily only. A greeting is forced to no spokes. Unknown spoke names are dropped. A keyword plan runs only when the supervisor model is unavailable.
4. `_route_after_supervisor` returns a node name or `[Send(spoke, state) for spoke in spokes]` for Path 1.
5. Discovery: Tavily writes candidates into `turn_notes` only, then `clarification_node` calls `interrupt(...)`. Streamlit resumes with `Command(resume=user_text)`. The node appends that text as a `HumanMessage`, clears `turn_notes`, and `Command(goto="supervisor")`. There is **no static edge** off `clarification_node`.
6. Second supervisor pass sees the ident and takes Path 1.
7. Synthesizer sanitizes pack order, sorts notes, appends any spoke that returned data but was missing from the list.

`recursion_limit` is **10**. Discovery JSON stays in `turn_notes`, not in durable `messages`.

### Discovery handoff (Path 2)

“Status of Batik Air from KL to BLR” calls Tavily, then pauses for a human ident. After the reply, the supervisor runs again as a known-ident turn and chooses which spokes run in parallel. OpenSky still soft-fails without an `icao24` hex. There is no registration-to-icao24 map.

---

## 6) Node responsibilities

### Supervisor (`supervisor_node`)

- **LLM:** OpenAI `supervisor_llm` (`SUPERVISOR_MODEL`, default `gpt-4o-mini`).
- **Mem0 (turn start):** `_search_user_memories` → up to 3 facts injected into the planner system prompt.
- **Structured output:** `SupervisorPlan { intent, flight_number, search_query, spokes_to_call, plan, context_priority_order, extract_memory }`. `plan` is required: one `{worker, task}` per selected spoke, no extras, no blank tasks. Empty only when `spokes_to_call` is empty.
- **Safety pack order:** if `flightaware` (may carry NOAA weather on delay) and/or `tavily` (NOTAMs) are selected, they must sit at the **start or end** of `context_priority_order` (lost-in-the-middle).
- **Memory Gate:** `extract_memory=true` only for explicit durable prefs; `false` for BA117 / ADS-B / weather / filler.
- **Fallback:** `_heuristic_plan` + `_heuristic_extract_memory` when supervisor LLM unavailable.
- Writes one `{spoke: "supervisor", content: "..."}` note for metro.

### `opensky` (`spoke_1_opensky_node`)

See [§7](#7-spoke-1--opensky-physical-airspace). Does **not** read FlightAware notes. Hex / airport ICAO come from the user text, supervisor task, or prior chat (including a durable OpenSky stamp from an earlier turn).

### `flightaware` (`spoke_2_liveops_node`)

1. Resolve ident via `_resolve_flight_ident`:
   - `flight_number` first (typed by the user or chosen at clarification)
   - then this spoke's `plan` task, then `current_task`, then the latest user message
   - then Tavily notes from this turn (almost never present under parallel `Send`)
   - then prior `messages` **only** if the latest prompt has follow-up phrases
   - stopword filter blocks acronym false hits (`MEL`, `UTC`, `ILS`, …)
2. If ident found → `flightaware_live_status` (NOAA METARs attach **only** on delay/hold/divert); else skip FA with a JSON note.
3. Pack `{spoke: "flightaware", content: "..."}`.

**No LLM** on this spoke — FA + conditional NOAA inside the tool.

### `tavily` (`spoke_3_websearch_node`)

1. Read this spoke’s `plan` task (fallback: latest user text).
2. **Fast path (0 LLM):** usual parallel case — Tavily on the task as-is.
3. **Chained path (1 `spoke_llm`):** only if sibling notes are already in the Send copy (rare now). Soft-fail keeps the original task.
4. Pack `{spoke: "tavily", content: "..."}`. Discovery still works; same-turn FA lookup does **not** wait for it.

### Synthesizer (`synthesizer_node`)

- Sanitize `context_priority_order` to `{opensky, flightaware, tavily}` only; drop hallucinations and duplicate ids.
- Sort `turn_notes` to that order. Every note for a spoke is grouped at that spoke's single slot. A spoke that returned data but was omitted from the list is appended. Supervisor notes go last.
- Format the sorted list as markdown → Gemini `spoke_llm`.
- Rules: trust FlightAware `headline`; if FA skipped but Tavily named flights, say live status was unavailable; don’t invent marketed numbers from callsign; OpenSky gaps ≠ “flight doesn’t exist.”
- **Durable OpenSky stamp** when live `states` exist.
- **Mem0 ingest** only if `extract_memory` is True.

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

1. Prefer this spoke’s `plan` task, else latest user ask + recent `messages` (**not** this-turn FlightAware notes).
2. **Fast path (0 LLM):** no `icao24` hex and no airport arrival/departure signal → heuristic only. Commercial flight numbers alone (e.g. AI101) cannot key OpenSky, so the spoke returns a structured soft error instead of stalling on Gemini.
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
- `tavily` → Web Research Engine  
- `flightaware` → Commercial Flight Engine  
- `opensky` → Physical Airspace Radar Engine  
- `synthesizer` → Intelligence Synthesizer  

### Plan-ordered track (visual)

The metro board rebuilds from `spokes_to_call` on the first supervisor hop:

`Router → <planned spokes> → Synthesizer`

Unplanned spokes are **omitted**. After the supervisor finishes, **all** planned spokes light **active at once** (parallel fan-out). The strip is a vertical numbered list via `st.html`. API / rate-limit style failures can show **warn** (amber). Bare `"429"` digit noise inside Tavily scores is ignored. OpenSky live states also drive Altitude / Ground Speed / Vertical Rate / Transponder Hex metrics.

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
| `recursion_limit` | `10` |

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
    │   ├── state.py            # AgentState (+ spokes_to_call, context_priority_order)
    │   ├── nodes.py            # Supervisor, 3 spokes, synthesizer, Mem0 helpers
    │   ├── supervisor.py       # Thin re-export of supervisor_node + SupervisorPlan
    │   └── workflow.py         # StateGraph + Send() fan-out + SqliteSaver
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

**Airline / route without flight number (Path 2: Tavily, then you pick)**

- `What's the status of the Batik Air flight from KL to BLR?`
- The run pauses with candidate idents. Reply with one (for example `ID8085`) and the next hop is Path 1 (FlightAware + OpenSky).

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
6. **FlightAware tool choice is fixed** (FA if ident; NOAA on anomaly) — not Gemini-planned like OpenSky.
7. **Path 2 needs a human ident.** Discovery no longer calls FlightAware blind, but it also does not auto-pick a candidate — the user must reply with one. A nonsense reply can loop until `recursion_limit=10`.
8. Candidate flights are still regexed out of Tavily note text, not a structured `discovered_ident` field. `flight_number` is set only after the user (or the original prompt) supplies an ident.
9. **NOAA enrichment is heuristic** — METAR ≠ proven root cause of delay. NOAA is not a graph spoke.
10. **LangSmith is live tracing, not a formal eval suite** — no golden-set harness / CI assertions yet; wrong regional endpoint → 403; bad keys fail soft.
11. **Metro Agent-steps snapshots** are session-scoped extras (chat text is `SqliteSaver`-backed).
12. **Offline heuristic planner** has some demo-tuned route phrases; live routing should prefer the supervisor LLM.
13. **Gemini free-tier quotas are per-model** — exhausting `gemini-3.6-flash` does not mean `gemini-2.5-flash` is dead, but the process must actually load the new `SPOKE_MODEL`.
14. **OpenSky without icao24 soft-fails fast** — flight-number-only radar asks will not invent a hex.

---

## 17) Interview / design talking points

- Why `Send` fan-out / fan-in beats a sequential supervisor loop for independent domains (OpenSky hex vs FA ident vs Tavily research).
- Why that same pattern **breaks** Tavily discovery → FlightAware lookup on one turn — and what a second map-reduce hop / structured `discovered_ident` would fix.
- Why `messages` (durable per `thread_id`) vs `turn_notes` (`operator.add` + `[]` clear) vs `context_priority_order` (synthesizer pack, not execution order).
- Why OpenSky vs FlightAware vs Tavily is a **three-way domain split**, and why NOAA stays inside FlightAware (delay gate + origin/dest from AeroAPI).
- Why FlightAware / Tavily must sit at an edge of `context_priority_order` (lost-in-the-middle).
- Why OpenSky keys only on hex / airport ICAO in the user/chat text — not on FlightAware tails.
- Why the metro board lights **all** planned spokes at once.
- Why gated history walk + stopwords for FlightAware idents (bleed vs follow-up).
- Why synthesizer must be allowed to say **“cannot join”** commercial airframe ↔ ADS-B hex.
- Why durable OpenSky stamps in `messages` matter for follow-up regex / memory.
- Why NOAA is conditional, single-RTT, phase-aware.
- Why Mem0 import stays lazy in nodes (import side effects / boot isolation) and failures soft-noop.
- Why `SqliteSaver` + `check_same_thread=False` fits Streamlit (multi-thread touches, still local file).
- Why LangSmith tags are dynamic (`entrypoint` / `user`) and FlightAware uses `hide_inputs` for the API key.
- Honest gaps: local SQLite / Mem0 stores, no reg→icao24 map, no same-turn Tavily→FA handoff, no offline eval suite.

---

## License / status

Personal / portfolio project. Not affiliated with FlightAware, OpenSky Network, NOAA, Google, OpenAI, Tavily, Mem0, or any airline.
