# FinalApproach architecture

This is the design of the LangGraph app. For how to run it, see [README.md](README.md).

A logged-in user asks something like “What’s the status of BA117?” or “Live ADS-B for icao24 4cc2a3?”. Streamlit sends that prompt into a LangGraph `StateGraph` with the chat’s `thread_id` and the Google `user_id`. A supervisor chooses which spokes run. Those spokes run in parallel, write notes, and a synthesizer writes one answer.

## Contents

1. [Domain split](#1-domain-split)
2. [Four paths](#2-four-paths)
3. [Runtime](#3-runtime)
4. [Frontend and auth](#4-frontend-and-auth)
5. [State](#5-state)
6. [Fan-out](#6-fan-out)
7. [Nodes](#7-nodes)
8. [OpenSky](#8-opensky)
9. [FlightAware, NOAA, Tavily](#9-flightaware-noaa-tavily)
10. [Metro board](#10-metro-board)
11. [Mem0](#11-mem0)
12. [LangSmith](#12-langsmith)
13. [Models](#13-models)
14. [Code map](#14-code-map)
15. [Limits](#15-limits)
16. [Evaluation: supervisor routing](#16-evaluation-supervisor-routing)

## 1) Domain split

| Spoke id | Handles | Does not handle |
|----------|---------|-----------------|
| `opensky` | A 6-character `icao24` hex, or airport arrivals/departures | Marketed flight numbers (BA117), tails (9V-SKQ), schedules, web research |
| `flightaware` | Flight numbers and tails, AeroAPI status, codeshares | Web research. Radar without a commercial ident |
| `tavily` | Web research, schedules, finding a flight number, NOTAMs | Live FlightAware status, OpenSky ADS-B |

NOAA is not a fourth spoke. `_enrich_formatted_with_noaa` runs inside `flightaware_live_status` only when AeroAPI reports delay, hold, divert, or a delay over 15 minutes.

OpenSky is added only when the user text has a 6-character icao24. A flight number or a tail, even with the words “radar” or “tracking”, stays on FlightAware. There is no registration-to-icao24 map, so a tail cannot unlock OpenSky.

An airline or route question with no ident is discovery: Tavily only, then the user picks an ident. A known ident may fan out any subset of the three spokes. The same spoke may run more than once when the tasks differ, such as FlightAware for BA117 and FlightAware for BA118. The synthesizer is allowed to say a commercial flight and an ADS-B hex cannot be joined.

## 2) Four paths

```text
ENTRY → supervisor
          │
          ├─ telemetry_known
          │     one Send per plan row ──► synthesizer → END
          │
          ├─ telemetry_discovery
          │     one Send per Tavily row → clarification_node (interrupt)
          │                 │  user picks ident
          │                 └── Command(goto="supervisor")
          │
          ├─ general_knowledge → one Send per Tavily row → synthesizer → END
          └─ direct_chat → synthesizer → END
```

1. **Known ident** (`telemetry_known`). The user already gave an ident, or just picked one. The supervisor chooses any subset of OpenSky, FlightAware, and Tavily. Those calls run together.
2. **Discovery** (`telemetry_discovery`). Live status with no ident, such as “Batik Air from KL to BLR”. Tavily only, then one pause. The reply is stored as a human message, scratch notes are cleared, and the supervisor runs again as a known-ident turn. The app does not pick a candidate itself.
3. **General research** (`general_knowledge`). Regulations, news, or specs with no live tracking. Tavily only, then the synthesizer. No pause.
4. **Direct chat** (`direct_chat`). A greeting, a rewrite, or anything the chat already answers. No spokes.

Discovery and general research force `spokes_to_call` to `["tavily"]` after the model returns, so an OpenSky or FlightAware row is dropped. A greeting clears both lists.

`recursion_limit` is 10. A nonsense reply on the discovery pause can loop until that limit. If LangGraph aborts, Streamlit calls the synthesizer on the notes collected so far and writes that partial answer back with `update_state`.

## 3) Runtime

```text
Streamlit (app.py)
  Google login, owned threads, chat from get_state(thread_id)
        │  HumanMessage + turn resets + user_id
        ▼
LangGraph (workflow.py)
  supervisor ─Send()─► spokes ─► synthesizer → END
  SqliteSaver → checkpoints.db, keyed by thread_id
  Mem0 search at the supervisor, add at the synthesizer, keyed by user_id
        │
        ├── opensky      Gemini tool plan, then the OpenSky API
        ├── flightaware  AeroAPI, NOAA only on a delay
        ├── tavily       web search
        └── synthesizer  reorder notes, Gemini, optional Mem0 add
```

Streamlit is the UI process. It imports the compiled graph. There is no separate API server.

## 4) Frontend and auth

| File | Role |
|------|------|
| `app.py` | Login, threads, chat, graph stream, metro, recursion fallback |
| `src/ui/metro_progress.py` | Station board |
| `src/graph/workflow.py` | Graph plus `checkpoints.db` |
| `src/db/auth_db.py` | Google identity to owned `thread_id`s |
| `src/graph/nodes.py` | Supervisor, spokes, synthesizer, Mem0 helpers |
| `src/tools/live_ops.py` | FlightAware, Tavily, NOAA |

Chat text is not stored in Streamlit session as the source of truth. Each paint reads `app_graph.get_state` for that `thread_id`. A new prompt sends only the new `HumanMessage`. Older turns are already in the checkpoint and the message reducer appends them. The prompt also clears turn scratch: `turn_notes=[]`, empty `plan`, `extract_memory=False`.

Google login uses Streamlit `st.login`. `auth_sessions.db` stores users and the threads they own. Four locks keep users apart: no UI without login, the sidebar lists only that user’s threads, checkpoints are keyed by `thread_id`, and Mem0 is keyed by the same Google `user_id` (`sub`, else email). Logout does not delete checkpoints.

A new chat is a draft until the first message. That first send writes the thread row. Empty drafts are not saved.

## 5) State

Defined in `src/graph/state.py`.

| Field | Write | Lifetime | Purpose |
|-------|-------|----------|---------|
| `messages` | Append, keep last 20 | Durable per thread | User lines and final answers. 20 messages is 10 question-answer pairs |
| `user_id` | Replace | Each turn | Google identity for Mem0 |
| `turn_notes` | `[]` clears, otherwise append | This turn | `{spoke, content}` notes from parallel calls |
| `plan` | Replace | This turn | One `{worker, task}` per call. The same worker may repeat when the tasks differ |
| `intent` | Replace | This turn | One of the four paths |
| `flight_number` | Replace | This turn | Ident once the user typed or picked one. Otherwise `None` |
| `search_query` | Replace | This turn | Tavily query, or `None` |
| `spokes_to_call` | Replace | This turn | Distinct spoke names, each once |
| `context_priority_order` | Replace | This turn | Order the synthesizer packs notes. Not the execution order |
| `next_worker` | Replace | This hop | First spoke id, or `FINISH`. The fan-out does not read it |
| `current_task` | Replace | This hop | First plan task, else the search query or user text |
| `extract_memory` | Replace | This turn | Memory Gate. Reset to false on every new prompt |
| `final_report` | Replace | This turn | Synthesizer answer |

`spokes_to_call` decides who runs. `plan` decides what each call does. A selected spoke with no row, or a row for a spoke that was not selected, fails the model schema and the keyword planner runs instead. An empty task fails earlier, on that row. An unknown name such as `noaa` or `synthesizer` is dropped before the set check. Two rows for one spoke do not fail the schema.

A call’s identity is `(worker, normalized task)`. Normalization lowercases, trims, and collapses whitespace, and that string is only a comparison key. The spoke still receives the original task. Caps count distinct tasks: OpenSky 2, FlightAware 2, Tavily 5.

`turn_notes=[]` at the start of a prompt clears the scratch pad so the previous turn’s tool output does not leak. Parallel calls append. There is no per-message character clip. The supervisor, the synthesizer, and OpenSky read the 20-message window.

Each call reads its task from the `Send` payload first, then that spoke’s first plan row, then `current_task`, then the latest user message. FlightAware looks for an ident inside its own payload task before the shared `flight_number`, so two FlightAware calls do not both look up BA117.

## 6) Fan-out

After the supervisor, the router in `workflow.py` sends one `Send` per remaining plan row:

```text
Send(worker, {**state, "task": task})
```

The extra `task` key is how two calls to the same node stay distinct. Caps, duplicate drops, and the spoke allow-list already happened in the supervisor. The router does not filter again. A greeting, or a turn with no spokes, goes straight to the synthesizer.

LangGraph waits until every send in that step finishes, then merges notes. Discovery then goes to the clarification pause. Every other Tavily hop, and every OpenSky or FlightAware hop, goes to the synthesizer. Spokes do not return to the supervisor except through the clarification resume.

Discovery calls Tavily, possibly several different queries. Gemini reads every Tavily note and returns at most 6 idents (`CandidateFlights`). It does not write the question. The pause is `interrupt()`. If idents exist, the question lists them and asks for the exact identifier. If the list is empty, a fixed sentence asks for a flight number or tail. Resume is `Command(resume=user_text)`. There is no static edge off `clarification_node`.

## 7) Nodes

**Supervisor.** OpenAI structured output, default `gpt-4o-mini`. It returns `intent`, `flight_number`, `search_query`, `spokes_to_call`, `plan`, `context_priority_order`, and `extract_memory`. Up to 3 Mem0 facts are injected into the planner prompt. If the model is missing or its plan fails validation, `_offline_supervisor` builds a keyword plan. `extract_memory` is true only for an explicit durable preference, not for a normal flight lookup.

**OpenSky.** Does not read this turn’s FlightAware notes. See [OpenSky](#8-opensky).

**FlightAware.** Resolves the ident from this call’s payload task, then `flight_number`, then the plan task, `current_task`, and the latest user message. Older messages are searched only when the latest prompt is a follow-up. A stopword list blocks acronym false hits (`MEL`, `UTC`, `ILS`). No LLM on this spoke.

**Tavily.** Uses the payload task. The usual path calls Tavily with no extra LLM. A rare chained path may ask Gemini to refine the query when sibling notes are already present.

**Synthesizer.** Gemini writes one answer. Notes are packed in `context_priority_order`. FlightAware and Tavily stay at an edge of that list so weather and NOTAMs are not buried in the middle. Every note for a spoke is grouped in that spoke’s one slot. The model must not invent a marketed flight number from a callsign, and an OpenSky miss is not “the flight does not exist”. Live OpenSky states are also stamped into the chat message. Mem0 `add` runs only when `extract_memory` is true.

## 8) OpenSky

Implementation: `src/mcp_servers/opensky_mcp.py`, planning in `nodes.py`.

| Tool | Purpose |
|------|---------|
| `get_live_aircraft_telemetry` | Live state vector by `icao24` |
| `get_airport_arrivals` | Arrivals, window at most 7 days |
| `get_airport_departures` | Departures, window at most 7 days |
| `get_aircraft_flight_history` | Legs for an `icao24`, window at most 30 days |
| `get_aircraft_trajectory_track` | Track for an `icao24` |

If the text has no 6-character hex and no airport arrival or departure signal, planning skips Gemini and the spoke returns a soft error. Otherwise Gemini fills `OpenSkyToolPlan`, with a 20 second wall clock. The HTTP calls time out at 10 seconds. Auth prefers OAuth client credentials, then optional basic auth, then anonymous.

`/states/all` returns positional lists. The MCP layer maps them to named fields.

## 9) FlightAware, NOAA, Tavily

Tools live in `src/tools/live_ops.py`.

FlightAware tries the ident as given, then an IATA-to-ICAO airline rewrite (`AI8303` also tries `AIC8303`). It builds a headline for status, route, aircraft, registration, and codeshares. Empty or failed AeroAPI responses stay as JSON notes.

NOAA runs only after that format step, and only on the anomaly gate. One METAR request covers origin and destination. Before departure, origin is primary. En route or holding, destination is primary. A weather failure does not fail the flight note.

Tavily runs only from the Tavily node. Without `TAVILY_API_KEY` it returns a not-connected note. The UI does not invent articles.

## 10) Metro board

`src/ui/metro_progress.py` draws one station per spoke, even when that spoke runs twice. Stations light together after the supervisor, because the calls are parallel. A second finished call appends its card under the first. A later success does not clear an earlier warning.

Names: supervisor is Flight Ops Router, Tavily is Web Research, FlightAware is Commercial Flight Engine, OpenSky is Physical Airspace Radar, synthesizer is Intelligence Synthesizer.

## 11) Mem0

Two stores, two keys.

| Store | Key | What it keeps |
|-------|-----|----------------|
| `checkpoints.db` | `thread_id` | Chat transcript and graph fields |
| Mem0 under `~/.mem0` | Google `user_id` | Preferences |

Search runs at the start of every supervisor visit. It embeds the query. It is not an LLM search. The helper keeps at most 3 facts. Add runs at the synthesizer only when the Memory Gate is open. Mem0 is imported lazily inside `nodes.py` so a missing install does not break boot. Search and add failures are swallowed and the turn still finishes.

## 12) LangSmith

Live traces use `run_name="Aviation_Ops_Turn"`, tags for entrypoint and user, and `recursion_limit=10`. FlightAware tool traces hide the API key. If a valid key gets HTTP 403, the usual cause is the wrong regional endpoint.

Routing eval is separate from those live traces. `scripts/create_langsmith_dataset.py` uploads `FinalApproach_Routing_Eval_v2`. `scripts/evaluate_routing.py` calls `supervisor_node` only and scores `intent`, `spokes_to_call`, and `extract_memory`. It does not run the graph, the spokes, or the synthesizer. Dataset `history` is hand-written prior chat, not a replay of `checkpoints.db`. Results are in [section 16](#16-evaluation-supervisor-routing).

## 13) Models

| Piece | Model | LLM? |
|-------|--------|------|
| Supervisor plan | OpenAI `gpt-4o-mini` by default | Yes, with a keyword fallback |
| Mem0 add | OpenAI, optional `MEM0_LLM_MODEL` | Yes on add. Search embeds only |
| OpenSky tool plan | Gemini `gemini-2.5-flash` by default | Only when a hex or an airport movement is present |
| FlightAware and NOAA | — | No |
| Tavily | — | No, except a rare query rewrite |
| Synthesizer | Gemini | Yes |

Getters in `src/graph/models.py` rebuild when `SUPERVISOR_MODEL` or `SPOKE_MODEL` changes.

## 14) Code map

```text
aviation_agent/
├── app.py
├── README.md
├── ARCHITECTURE.md
├── scripts/
│   ├── create_langsmith_dataset.py
│   └── evaluate_routing.py
└── src/
    ├── db/auth_db.py
    ├── graph/
    │   ├── models.py
    │   ├── state.py
    │   ├── nodes.py
    │   ├── supervisor.py      # re-export only
    │   └── workflow.py
    ├── mcp_servers/opensky_mcp.py
    ├── tools/live_ops.py
    └── ui/metro_progress.py
```

`archive/` holds the old BTS SQLite path. It is not on the live graph.

## 15) Limits

- Checkpoints, auth, and Mem0 are local demo stores.
- OpenSky cannot list every flight on a city pair, and it cannot turn a tail into a hex.
- Discovery does not auto-call FlightAware with a candidate it found.
- NOAA text is a nearby METAR, not a proven cause of the delay.
- The keyword planner is the fallback, not the live router.
- The routing eval does not grade answer quality or Mem0 retrieval.

## 16) Evaluation: supervisor routing

- **What it does:** for one user message (and, in some cases, earlier chat), it checks whether the supervisor picks the right path, the right spokes, and the right memory-gate setting.
- **What it does not do:** it does not run the spokes, call the live aviation APIs, or grade the written answer.

### What is measured

`scripts/evaluate_routing.py` builds the same fresh-turn state that `app.py` sends for a new prompt, calls `supervisor_node` once, and scores three fields of its output. Each check is pass or fail.

| Check | Field | Passes when |
|-------|-------|-------------|
| Intent | `intent` | It equals the expected path exactly: `telemetry_known`, `telemetry_discovery`, `general_knowledge`, or `direct_chat` |
| Spokes | `spokes_to_call` | Every spoke in the case's `must_include` list is present and no spoke in its `must_exclude` list is. Other spokes are allowed. Plan rows, tasks, idents, and caps are not checked |
| Memory gate | `extract_memory` | It equals the expected true/false. True is expected only for an explicit, lasting preference |

The eval runs with an empty `user_id`, so Mem0 search is skipped.

**Pass criterion:** a case passes only when all three checks pass.

### Results

| Metric | Passed | Rate |
|--------|--------|------|
| **All three checks (joint)** | **45/51** | **88.2%** |
| Intent | 50/51 | 98.0% |
| Spokes | 47/51 | 92.2% |
| Memory gate | 49/51 | 96.1% |

By split: dev 8/8, test 37/43.

LangSmith left the scores empty for 2 cases, "Hello there, how are you today?" and "Book me a ticket to London.". Their outputs (`direct_chat`, no spokes, `extract_memory` false) match the expected labels exactly, so they were graded from the exported outputs using the same rules as the eval script. Those rules reproduce every one of the 147 scores LangSmith did record. Counting only the scores LangSmith recorded, the result is 43/51 (84.3%).

### Failures (6 of 51)

| # | Query | Failed | Expected | Actual | Cause |
|---|-------|--------|----------|--------|-------|
| 1 | Always assume I am operating in a Linux environment when discussing software. | Memory | `direct_chat`, no spokes, memory **true** | `direct_chat`, no spokes, memory false | TBD |
| 2 | Keep all future answers in tight dispatcher bullets. | Memory | `direct_chat`, no spokes, memory **true** | `direct_chat`, no spokes, memory false | TBD |
| 3 | What is the UTC arrival time for the flight using the ILS approach? | Intent, spokes | `general_knowledge`, must include `tavily` | `direct_chat`, no spokes | TBD |
| 4 | I need live radar for hex a1b2c3, the commercial status of AI8303, and the latest NOTAMs for DEL. | Spokes | `telemetry_known`, must include `opensky`, `flightaware`, `tavily` | `telemetry_known`, `flightaware` + `tavily` | TBD. A hex is in the message, so the OpenSky rule allows `opensky` here |
| 5 | Pull live OpenSky radar for British Airways flight BA117. | Spokes | `telemetry_known`, must include `opensky`, `flightaware` | `telemetry_known`, `flightaware` only | The label conflicts with the supervisor prompt. The OpenSky rule in `nodes.py` sends a flight number to FlightAware only, even when the user asks for radar, unless a hex is given. Either the label or the rule has to change |
| 6 | Is the radar track still showing it at 35,000 ft? (earlier chat: "Pull radar for 4cc2a3.") | Spokes | `telemetry_known`, must include `opensky` | `telemetry_known`, `flightaware` only | TBD. The hex is only in the earlier chat, and the OpenSky rule requires it "explicitly provided in the user's query" |

### Provenance

| Item | Value |
|------|-------|
| Results file | `Routing_Supervisor_Test-20d2bf97.csv`, a CSV export of the LangSmith experiment |
| Dataset | `FinalApproach_Routing_Eval_v2`: the 51 cases in `scripts/create_langsmith_dataset.py` (8 dev, 43 test), 1 repetition |
| Run date | 21 Nov 2026, about 19:46 IST. The CSV has no run timestamp. The dataset was uploaded at 19:45–19:46 IST and the CSV was exported at 19:48 IST |
| Code version | LangSmith records `5b316e0-dirty`: commit `5b316e0` plus an uncommitted edit to `scripts/create_langsmith_dataset.py` (the case list). No application code was changed |
| Supervisor model | OpenAI `gpt-4o-mini`, from `SUPERVISOR_MODEL` in the local `.env`. The CSV does not record the model name. Every case shows token usage, so the LLM ran, not the keyword fallback |
| Cost and latency | About $0.012 in total, 1.9 s average per case, 2 cases at a time |

To reproduce (needs `LANGCHAIN_API_KEY` and `OPENAI_API_KEY` in `.env`):

```bash
cd aviation_agent
source .venv/bin/activate
python scripts/create_langsmith_dataset.py   # deletes and re-creates the dataset from the case list
python scripts/evaluate_routing.py
```

The model's output is not deterministic, so a rerun can differ by a case or two.

### Limitations

- 51 hand-written cases. The earlier chat in history cases is scripted, not taken from real sessions.
- Routing only. There is no eval of answer quality, grounding, hallucination, or tool output yet.
- The spokes check only tests which spokes are included or excluded. It does not check plan tasks, idents, per-spoke caps, or repeated calls to the same spoke.
- One repetition, so there is no measure of run-to-run variance.
- Mem0 retrieval is not exercised.
