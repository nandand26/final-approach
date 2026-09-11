# Aviation Fleet & Operational Intelligence Agent

A multi-agent aviation operations assistant built with **LangGraph**, **Streamlit**, **OpenSky Network** (physical ADS-B), **FlightAware / NOAA / Tavily** (commercial + web), **OpenAI** (supervisor planning), and **Google Gemini** (OpenSky tool planning + final synthesis).

It answers operational questions by combining:

| Source | Role |
|--------|------|
| OpenSky Network (Spoke 1 / FastMCP) | Live ADS-B vectors, airport movements, tracks (keyed by **icao24** hex) |
| FlightAware AeroAPI | Live / near-live commercial flight status, aircraft, codeshares (Spoke 2) |
| NOAA Aviation Weather Center METARs | Conditional weather on delay / hold / divert (Spoke 2) |
| Tavily web search | Unstructured aviation research (Spoke 2) |
| Auth SQLite (`auth_sessions.db`) | Google user → owned chat `thread_id`s only |

Access is gated by **Google OAuth**. Each user owns isolated **chat threads**; LangGraph checkpoints are keyed by `thread_id` so memory does not leak across users or chats.

> **Scope note:** Strong portfolio / demo system (real orchestration, real APIs, real auth boundaries). Not a production ops platform: the checkpointer is in-process `MemorySaver`, there is no registration→icao24 bridge yet, and there is no formal eval / LangSmith harness.

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
8. [Spoke 2 tools (FlightAware, Tavily, NOAA)](#8-spoke-2-tools-flightaware-tavily-noaa)
9. [Metro progress UI](#9-metro-progress-ui)
10. [LLM usage map](#10-llm-usage-map)
11. [Codebase structure](#11-codebase-structure)
12. [Setup & run](#12-setup--run)
13. [Example prompts](#13-example-prompts)
14. [Current limitations](#14-current-limitations)
15. [Interview / design talking points](#15-interview--design-talking-points)

---

## 1) System overview

### What the product does

A logged-in user opens a chat thread and asks something like “What’s the status of BA117?” or “Live ADS-B for icao24 4cc2a3?” Behind the UI:

1. Streamlit authenticates the user and selects an owned `thread_id`.
2. The prompt is sent into a LangGraph `StateGraph` with that `thread_id` in config.
3. A **supervisor** builds a short **plan** (ordered spoke queue) once per turn.
4. **Spokes** run tools and write scratch into `turn_notes`.
5. Spokes return to the supervisor until the plan is drained.
6. A **synthesizer** turns scratch + recent chat into the final assistant message (and stamps a durable OpenSky vector block when live states exist).
7. Streamlit streams metro progress live (`st.status` + station board), then reloads durable history from the checkpointer.

### Domain split (important)

| Spoke | Handles | Does **not** handle |
|-------|---------|---------------------|
| **Spoke 1 — OpenSky** | `icao24`, altitude / GS / VS, trajectories, ADS-B airport arrivals/departures | Marketed flight numbers (BA117, AI8303), schedules, A380-type claims from callsign alone |
| **Spoke 2 — Live ops** | Flight numbers / tails, FlightAware status, codeshares, METAR-on-anomaly, Tavily | Pure physical radar without a commercial ident |

Hybrid asks (e.g. BA117 **and** icao24 `4cc2a3`) may schedule **both**. The synthesizer is prompted to say when those identities **cannot be joined**.

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
                            │ config.configurable.thread_id
┌───────────────────────────▼─────────────────────────────────┐
│  LangGraph StateGraph (workflow.py)                         │
│  supervisor ⇄ spokes → synthesizer → END                    │
│  checkpointer: MemorySaver (keyed by thread_id)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  Spoke 1 OpenSky     Spoke 2 LiveOps      Synthesizer
  (Gemini → tool      (no LLM in tools;    (Gemini +
   plan → MCP)         FA + Tavily + NOAA)  durable OpenSky stamp)
```

---

## 2) Frontend ↔ backend connection (Streamlit ↔ LangGraph)

The UI is not a separate API server — Streamlit **is** the presentation process and imports the compiled graph directly.

### Key files

| File | Role |
|------|------|
| `app.py` | Auth gate, sidebar threads, chat I/O, graph stream, metro paint, recursion fallback |
| `src/ui/metro_progress.py` | Station board, layman cards, OpenSky metrics columns, progress snapshots |
| `src/graph/workflow.py` | Builds/compiles `app_graph` with `MemorySaver` |
| `src/db/auth_db.py` | Maps Google identity → owned `thread_id`s |
| `src/graph/nodes.py` | Supervisor / spokes / synthesizer (+ recursion-fallback import) |

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
    "turn_notes": [],   # explicit reset → merge_turn_notes clears scratch
    "plan": [],         # explicit reset → prior turn queue cannot leak
    "next_worker": "",
    "final_report": "",
}
```

3. Stream inside `st.status("Orchestrating Flight Intelligence Pipeline...")`:

```python
app_graph.stream(
    graph_input,
    stream_mode="updates",
    config={
        "recursion_limit": 16,
        "configurable": {"thread_id": active_thread_id},
    },
)
```

4. For each node update:
   - `advance_on_node_update(progress, node_name, update)` advances metro lights / cards.
   - Flush `progress.status_log` into the open `st.status` panel.
   - Repaint metro / answer via `render_progress`.
5. Snapshot agent steps → `st.session_state`, then `st.rerun()` so the UI reloads **canonical** messages from the checkpointer.

### Important frontend/backend contracts

| Concern | Contract |
|---------|----------|
| Isolation | Every graph call must include `configurable.thread_id` owned by the logged-in user |
| History | Checkpointer is source of truth for chat text; UI is a view |
| Turn scratch | `turn_notes=[]` and `plan=[]` at every new prompt |
| Progress | Streaming `updates` is UX-only; durable answer lands in `messages` / `final_report` |
| Recursion fuse | `GraphRecursionError` → synthesize partial answer from collected notes + `update_state` |

### Recursion-limit fallback

Plan-and-Execute can hop supervisor ↔ spokes several times. `recursion_limit` is **16**. If LangGraph still aborts:

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

### Isolation model (three locks)

1. **Auth lock:** no chat UI without Google login.
2. **Ownership lock:** sidebar only queries threads for `current_user_id`.
3. **Memory lock:** LangGraph checkpointer partitions state by `thread_id`.

> **Durability caveat:** `MemorySaver` is process-local. Restarting Streamlit loses graph checkpoints (auth ownership rows in SQLite remain). Logout alone does not intentionally wipe MemorySaver while the process stays up.

---

## 4) Agent state (`AgentState`)

Defined in `src/graph/state.py`.

| Field | Reducer / write semantics | Lifetime | Purpose |
|-------|---------------------------|----------|---------|
| `messages` | `add_messages_window` (append, trim to last **10**) | Durable per `thread_id` | User + final assistant answers |
| `turn_notes` | `merge_turn_notes` (`[]` clears; else append) | Turn-local | Supervisor + spoke scratch |
| `plan` | Replace | Turn-local | Ordered spoke queue still left to run |
| `next_worker` | Replace | Hop-local | `Spoke_1_OpenSky` \| `Spoke_2_LiveOps` \| `FINISH` |
| `final_report` | Replace | Turn result | Synthesizer answer string |

### Why `turn_notes` has a custom reset

With a checkpointer, naive “append forever” scratch would leak prior-turn tool output. `app.py` always sends `turn_notes=[]` at prompt start; the reducer treats empty list as **clear**.

### What is *not* top-level state

`flight_ident`, OpenSky args, `weather_context`, etc. live inside spoke JSON embedded in `turn_notes` (and durable OpenSky fields are also stamped into the final `messages` text).

Constants: `MESSAGE_WINDOW = 10`, `MAX_SPOKE_EXECUTIONS_PER_TURN = 2`.

---

## 5) LangGraph topology & Plan-and-Execute

### Graph wiring (`src/graph/workflow.py`)

```text
ENTRY → supervisor
          │
          ├─(Spoke_1_OpenSky)→ spoke_1_opensky ──┐
          ├─(Spoke_2_LiveOps)→ spoke_2_liveops ──┤
          └─(FINISH / other)→ synthesizer → END  │
                                                 │
                     ◄───────────────────────────┘
                     (spokes always return to supervisor)
```

```python
app_graph = workflow.compile(checkpointer=MemorySaver())
```

### Plan-and-Execute lifecycle (one user message)

**Phase A — turn start** (`turn_notes` empty after app reset):

1. Supervisor LLM (or heuristic fallback) builds ordered `plan` from latest query + recent `messages`.
2. Caps: each spoke ≤ 2 appearances per turn; invalid names dropped.
3. Empty plan → conversational bypass → `FINISH`.
4. Non-empty → pop head into `next_worker`, keep remainder in `plan`.

**Phase B — drain queue:**

1. Spoke runs → `[Spoke_X_...]` into `turn_notes` → back to supervisor.
2. Re-sanitize remaining plan against execution counts.
3. Empty / cap → `FINISH` → synthesizer.

This is **not** unbounded ReAct. Short queue + hard caps = predictable cost/latency for demos.

**Supervisor does not pass tool args to spokes.** It only sets `next_worker`. Spokes read the user question from shared `messages` and extract their own args.

---

## 6) Node responsibilities

### Supervisor (`supervisor_node`)

- **LLM:** OpenAI `supervisor_llm` (`SUPERVISOR_MODEL`, default `gpt-4o-mini`).
- **Structured output:** `SupervisorPlan { plan: list[str] }`.
- **Fallback:** `_heuristic_plan` keyword routing (OpenSky signals vs commercial/live signals).
- Writes routing notes into `turn_notes` for metro / `st.status`.

### Spoke 1 — OpenSky (`spoke_1_opensky_node`)

See [§7](#7-spoke-1--opensky-physical-airspace).

### Spoke 2 — live ops (`spoke_2_liveops_node`)

1. Resolve ident via `_resolve_flight_ident` (latest text, else walk `messages`).
   - Matches airline flights (`BA117`), US tails (`N12345`), hyphen regs (`9V-SKQ`, `D-AIZZ`), and ATC-style callsigns (`DLH6K`).
2. If ident found → `flightaware_live_status`; else skip FA with a JSON note.
3. Always call Tavily (query enriched with ident + history).
4. Pack both into one `[Spoke_2_LiveOps]` `turn_notes` entry.
5. Return to supervisor.

**Spoke 2 itself does not call an LLM** for tool selection — fixed pipeline (FA if ident + always Tavily). NOAA may run inside the FA tool on anomalies.

### Synthesizer (`synthesizer_node`)

- **LLM:** Gemini `spoke_llm`.
- Inputs: `turn_notes`, recent conversation, resolved ident.
- Rules: trust FlightAware `headline`; honest codeshares; don’t invent marketed LH numbers / types from callsign alone; OpenSky gaps ≠ “flight doesn’t exist.”
- **Durable OpenSky stamp:** if Spoke 1 returned live `states`, append a fixed `### OpenSky vector (durable)` block (icao24, callsign, alt, GS, VS, track, lat/lon, on_ground, squawk) so follow-ups can regex those ids from `messages`.
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

1. Build `planning_text` from latest user ask + recent `messages` (**not** from `turn_notes`).
2. Gemini `with_structured_output(OpenSkyToolPlan)` → tool name + args.
3. Fallback: `_heuristic_opensky_plan` (regex hex / airport keywords).
4. `invoke_opensky_tool(plan.tool, **kwargs)` — drops `None`s; missing required args return `{"ok": false, ...}` JSON instead of crashing the graph.

### Auth

Preference order in `_opensky_get`:

1. OAuth2 client credentials (`OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`) with in-process token cache.
2. Optional username/password basic auth.
3. Anonymous (lowest rate limits).

### Contract note

OpenSky live endpoints key on **icao24 hex**, not marketed flight numbers or registration. FlightAware may return a tail like `9V-SKQ` without a hex — there is **no** reg→icao24 bridge in this repo yet. Hybrid demos should either include an explicit hex or accept “cannot join commercial airframe ↔ ADS-B target.”

State vectors from `/states/all` are **positional lists** (official column order); the MCP layer maps them to named dicts.

---

## 8) Spoke 2 tools (FlightAware, Tavily, NOAA)

All live in `src/tools/live_ops.py`.

### FlightAware AeroAPI (`flightaware_live_status`)

- HTTP via `requests` + `certifi`.
- Candidate loop: as-given ident, then IATA→ICAO airline rewrite (e.g. `AI8303` → also try `AIC8303`).
- Builds a **headline** (status, route, aircraft, registration, codeshare flags).
- Honest empty / error JSON when AeroAPI fails.

### Tavily (`tavily_aviation_web_search`)

- Live when `TAVILY_API_KEY` is set.
- Structured mock / “web search not connected” path otherwise (metro UI does not show fake articles).

### NOAA METAR enrichment (conditional, tool-internal)

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
- Spoke 2 → Commercial & Web Engine  
- Spoke 1 → Physical Airspace Radar Engine  
- `synthesizer` → Intelligence Synthesizer  

### Fixed track (visual)

Router → Commercial → Physical → Synthesizer  

Spokes not in the plan render as **Skipped**. Active station pulses; API / rate-limit style failures can show **warn** (amber). Spoke 1 live states also drive Altitude / Ground Speed / Vertical Rate / Transponder Hex metrics.

### Streaming

`app.py` calls `advance_on_node_update` on each LangGraph `updates` event and keeps snapshots in `session_state` for collapsed **Agent steps** after `st.rerun()`.

---

## 10) LLM usage map

| Component | Model path | Uses LLM? |
|-----------|------------|-----------|
| Supervisor plan | OpenAI `supervisor_llm` | Yes (`SupervisorPlan`); heuristic fallback |
| Spoke 1 tool plan | Gemini `spoke_llm` | Yes (`OpenSkyToolPlan`); regex fallback |
| OpenSky HTTP tools | — | No |
| Spoke 2 FA / Tavily / NOAA | — | No |
| Synthesizer | Gemini `spoke_llm` | Yes (+ deterministic OpenSky durable stamp) |

Env overrides: `SUPERVISOR_MODEL` (default `gpt-4o-mini`), `SPOKE_MODEL` (default `gemini-3.6-flash`).

---

## 11) Codebase structure

```text
aviation_agent/
├── app.py                      # Streamlit UI + graph stream + auth gate
├── README.md
├── requirements.txt
├── .env / .env.example
├── .streamlit/secrets.toml     # Google OAuth (local)
├── auth_sessions.db            # Runtime: users ↔ threads
├── archive/                    # Former BTS SQLite / MCP (not on hot path)
├── notes.md                    # Learning notes
└── src/
    ├── db/
    │   └── auth_db.py          # User + thread ownership
    ├── graph/
    │   ├── models.py           # supervisor_llm / spoke_llm
    │   ├── state.py            # AgentState + reducers + caps
    │   ├── nodes.py            # Supervisor, spokes, synthesizer
    │   ├── supervisor.py       # Thin re-export of supervisor_node
    │   └── workflow.py         # StateGraph + MemorySaver
    ├── mcp_servers/
    │   └── opensky_mcp.py      # OpenSky tools + OAuth + invoke dispatch
    ├── tools/
    │   └── live_ops.py         # FlightAware + Tavily + NOAA
    └── ui/
        └── metro_progress.py   # Metro board + layman cards
```

---

## 12) Setup & run

### Prerequisites

- Python 3.10+ recommended
- Google Cloud OAuth client for Streamlit redirect URI
- API keys for the paths you want live (OpenAI, Gemini, FlightAware, OpenSky OAuth; Tavily optional)

### Install

```bash
cd aviation_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment (`.env`)

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
FLIGHTAWARE_API_KEY=...
TAVILY_API_KEY=...                 # optional
OPENSKY_CLIENT_ID=...
OPENSKY_CLIENT_SECRET=...
SUPERVISOR_MODEL=gpt-4o-mini
SPOKE_MODEL=gemini-3.6-flash
```

### Streamlit auth (`.streamlit/secrets.toml`)

Provide the `[auth]` block for Streamlit native Google login. Do **not** commit secrets.

### Run

```bash
streamlit run app.py
```

---

## 13) Example prompts

**Commercial / codeshare**

- `What's the live status of AI8303? Include route, aircraft type, and codeshare operator if any.`
- Follow-up: `If I fly this two months from now, what aircraft should I usually expect?`

**Physical airspace (OpenSky)**

- `Pull live OpenSky ADS-B telemetry for icao24 4cc2a3 — altitude, ground speed, vertical rate, track, on_ground.`

**Hybrid (both spokes + disambiguation)**

- `How’s British Airways flight BA117 looking right now — on time, what plane, what route? Also pull live radar for aircraft code 4cc2a3 and tell me straight: same plane as BA117, or a different one?`
- Follow-up: `Ignore the radar track from earlier. Just for BA117 — with Heathrow weather chaos lately, does this flight look delayed or pretty normal?`

**Conversational bypass**

- `Rewrite your last answer as three bullet points for a dispatcher.`

---

## 14) Current limitations

1. **`MemorySaver` is in-process** — isolation works; durability across process restart does not.
2. **No registration → icao24 bridge** — FlightAware tails (e.g. `9V-SKQ`) do not auto-unlock OpenSky telemetry.
3. **OpenSky ≠ commercial schedules** — cannot enumerate “all flights DEL→SIN.”
4. **State vectors are brittle positional APIs** — mitigated by length checks + named mapping in MCP.
5. **Spoke 2 tool choice is fixed** (FA if ident + always Tavily) — not Gemini-planned like Spoke 1.
6. **NOAA enrichment is heuristic** — METAR ≠ proven root cause of delay.
7. **No formal eval harness / LangSmith tracing** yet.
8. **Metro Agent-steps snapshots** are session-scoped extras (chat text is checkpointer-backed).

---

## 15) Interview / design talking points

- Why Plan-and-Execute with a capped queue beats unbounded tool loops for demo cost control.
- Why `messages` (durable) vs `turn_notes` (scratch) vs checkpointer `thread_id` (tenancy).
- Why OpenSky vs FlightAware is a **domain split**, not two ways to ask the same question.
- Why Spoke 1 uses structured `OpenSkyToolPlan` while Spoke 2 stays a tool-only pipeline.
- Why synthesizer must be allowed to say **“cannot join”** commercial airframe ↔ ADS-B hex.
- Why durable OpenSky stamps in `messages` matter for follow-up regex / memory.
- Why NOAA is conditional, single-RTT, phase-aware.
- Honest gaps: MemorySaver durability, no reg→icao24 map, evals.

---

## License / status

Personal / portfolio project. Not affiliated with FlightAware, OpenSky Network, NOAA, Google, OpenAI, Tavily, or any airline.
