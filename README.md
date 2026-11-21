# FinalApproach

FinalApproach is a chat assistant for live aviation questions. You ask about a flight, a radar track, or a route, and it pulls the right live source, then writes one answer.

For the routing design, state, and node flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

It is a portfolio demo: real APIs, real Google login, local SQLite chat history. It is not a production operations platform.

## What it uses

| Source | Used for |
|--------|----------|
| FlightAware | Live commercial status for a flight number or tail (BA117, 9V-SKQ) |
| OpenSky | Live ADS-B when the question includes a 6-character icao24 hex, or an airport arrival/departure |
| Tavily | Web research, NOTAMs, and finding a flight number when the user did not give one |
| NOAA | Weather, only inside FlightAware, and only when that flight is delayed, holding, or diverted |

A question with no flight number, such as "Batik Air from KL to BLR", searches the web and then pauses so you can pick the ident. A greeting or a rewrite of the last answer does not call any of these.

## Run

Python 3.10+. You need a Google OAuth client for Streamlit, plus the API keys for the sources you want live.

```bash
cd aviation_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the keys you have. The usual ones are `OPENAI_API_KEY` (supervisor and Mem0), `GOOGLE_API_KEY` (spokes and the final answer), `FLIGHTAWARE_API_KEY`, `TAVILY_API_KEY`, and OpenSky `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`. Put the Streamlit Google login block in `.streamlit/secrets.toml`. Do not commit either file.

```bash
streamlit run app.py
```

Sign in with Google, open a chat, and send a question. Threads stay on this machine in `checkpoints.db`. Preferences such as "always report altitude in meters" are stored separately in local Mem0 and can show up in a later chat.

## Try these

- `What's the live status of AI8303?`
- `What's the status of the Batik Air flight from KL to BLR?` then reply with one of the candidate idents.
- `Pull live OpenSky ADS-B telemetry for icao24 4cc2a3.`
- `How’s BA117 looking right now? Also pull live radar for aircraft code 4cc2a3.`
- `Research typical aircraft types on the Delhi–Singapore route.`
- `Rewrite your last answer as three bullet points.`
- `Always report altitudes in meters.`

## Limits

- Chat history and Mem0 are local files, not a hosted multi-user store.
- A tail number does not become an OpenSky hex. OpenSky needs the hex, or it says it cannot look the aircraft up.
- Discovery will not pick a flight for you. You reply with the ident.
- A flight number alone does not call OpenSky.

## Routing checks

`scripts/create_langsmith_dataset.py` uploads labeled supervisor cases to LangSmith. `scripts/evaluate_routing.py` runs the supervisor against that dataset and scores intent, spokes, and the memory gate. That checks routing only. It does not call the live aviation APIs or grade the written answer.

Latest routing eval (21 Nov 2026): **45/51 cases (88.2%) route fully correctly**, meaning intent, spokes, and memory gate are all right. LangSmith left 2 of those 51 cases unscored, so they were graded from their exported outputs using the same rules. Results, failures, and how to rerun it are in [ARCHITECTURE.md](ARCHITECTURE.md#16-evaluation-supervisor-routing).

Personal project. Not affiliated with FlightAware, OpenSky Network, NOAA, Google, OpenAI, Tavily, Mem0, or any airline.
