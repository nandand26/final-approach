"""Live ops tools: FlightAware + Tavily (+ optional NOAA weather).

Easy map of this file (what each part is for):
  - flightaware_live_status  → live / near-live flight status JSON
  - tavily_aviation_web_search → web articles about aviation topics
  - NOAA helpers             → only when a flight looks delayed/held/diverted

Who consumes the JSON?
  - Spoke 2 packs it into turn_notes for the synthesizer (full detail OK there)
  - The Streamlit metro UI (src/ui/metro_progress.py) shows ONLY a friendly
    subset: mainly `headline`, optional `weather_context`, and Tavily titles

REVIEW — recent fixes (search for "REVIEW:"):
1) SSL: urllib → requests + certifi
2) Codeshares (e.g. AI8303 operated as SQ403)
3) IATA→ICAO airline candidates (AI8303 then AIC8303)
4) Compact `headline` block for synthesizer + UI
5) NOAA METARs on delay anomalies (one HTTP call for origin+dest)
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Any

# REVIEW: needed because urllib SSL failed on MITM/corporate proxy; requests+certifi works
import certifi
import requests
from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

# FlightAware AeroAPI base URL
AEROAPI_BASE = "https://aeroapi.flightaware.com/aeroapi"
# NOAA Aviation Weather Center METAR endpoint (no API key required)
NOAA_METAR_URL = "https://aviationweather.gov/api/data/metar"
# Delay threshold for weather enrichment: 15 minutes in seconds
WEATHER_DELAY_ANOMALY_SEC = 900

# Lazy-loaded IATA->airport dict for ICAO fallback (airportsdata.load is heavy).
_IATA_AIRPORTS: dict[str, Any] | None = None # None = None means that the dictionary is not yet loaded.

# REVIEW: AeroAPI prefers ICAO idents; map common IATA airline prefixes
# this is a dictionary that maps IATA airline codes to ICAO airline codes
IATA_TO_ICAO_AIRLINE = {
    "AI": "AIC",  # Air India
    "AA": "AAL",
    "UA": "UAL",
    "DL": "DAL",
    "WN": "SWA",
    "B6": "JBU",
    "SQ": "SIA",
    "BA": "BAW",
    "EK": "UAE",
    "QR": "QTR",
    "LH": "DLH",
    "AF": "AFR",
}


# ---------------------------------------------------------------------------
# Pydantic input schemas (help LangGraph / tool-calling LLMs fill args)
# ---------------------------------------------------------------------------

class FlightAwareInput(BaseModel):
    """Input for live FlightAware AeroAPI lookups."""

    ident: str = Field(
        ...,
        description=(
            "Flight ident or tail/registration to look up, "
            "e.g. 'UAL123', 'AAL100', or 'N12345'."
        ),
    )


class TavilyAviationInput(BaseModel):
    """Input for unstructured aviation web research via Tavily."""

    query: str = Field(
        ...,
        description=(
            "Aviation research query: airline news, fleet moves, "
            "engine recalls, weather advisories, press releases, etc."
        ),
    )


def _usable_key(name: str) -> str | None:
    """Return env API key only if it looks real (not a .env.example placeholder)."""
    value = os.getenv(name, "")
    if not value or "your_" in value or "api_key_here" in value:
        return None
    return value


# ---------------------------------------------------------------------------
# FlightAware — real-time flight / tail telemetry
# ---------------------------------------------------------------------------

def _airport_code(airport: Any) -> str | None:
    """Prefer IATA (LHR) then ICAO (EGLL) from an AeroAPI airport object."""
    if not isinstance(airport, dict):
        return None
    return (
        airport.get("code_iata")
        or airport.get("code")
        or airport.get("code_icao")
    )


def _iata_to_icao_local(iata: str | None) -> str | None:
    """Convert IATA airport code to ICAO using local airportsdata DB (no network)."""
    global _IATA_AIRPORTS
    if not iata:
        return None
    code = str(iata).strip().upper() # first strip the iata code and convert to uppercase
    if len(code) != 3: # if the length of the code is not 3, return None
        return None
    try: # try to convert the iata code to icao code using the airportsdata DB  
        if _IATA_AIRPORTS is None: # if the dictionary is not yet loaded, load it
            import airportsdata 

            # Load once: maps IATA -> airport record (includes ICAO field).
            _IATA_AIRPORTS = airportsdata.load("IATA")
        record = (_IATA_AIRPORTS or {}).get(code) # get the corresponding airport record from the dictionary
        if not record:
            return None
        icao = record.get("icao") if isinstance(record, dict) else getattr(record, "icao", None)
        return str(icao).upper() if icao else None
    except Exception:  # noqa: BLE001
        return None


def _extract_airport_icao(airport: Any) -> str | None:
    """Pull 4-letter ICAO from a FlightAware airport object.

    Extraction order (as required):
    1) Direct from FlightAware nested telemetry: airport.code_icao
    2) Fallback: airport.code_iata -> airportsdata IATA->ICAO local lookup
    """
    if not isinstance(airport, dict):
        return None
    # 1) Preferred: ICAO already present on the AeroAPI payload
    #    e.g. payload["flights"][i]["origin"]["code_icao"] == "VIDP"
    icao = airport.get("code_icao")
    if icao and str(icao).strip():
        return str(icao).strip().upper()
    # 2) Fallback when AeroAPI only gave IATA (DEL) — convert locally
    iata = airport.get("code_iata") or airport.get("code")
    return _iata_to_icao_local(str(iata) if iata else None)


def _summarize_flight(flight: dict[str, Any], query_ident: str) -> dict[str, Any]:
    """Shrink one AeroAPI flight object to fields the agent actually needs.

    REVIEW: now takes query_ident so we can mark codeshares
    (e.g. AI8303 is a codeshare of operating SIA403 / SQ403).
    """
    origin = flight.get("origin") if isinstance(flight.get("origin"), dict) else {}
    dest = flight.get("destination") if isinstance(flight.get("destination"), dict) else {}
    q = query_ident.upper()
    codeshares_iata = [str(c).upper() for c in (flight.get("codeshares_iata") or [])]
    codeshares_icao = [str(c).upper() for c in (flight.get("codeshares") or [])]
    operating_ident = (flight.get("ident") or flight.get("ident_icao") or "").upper()
    operating_iata = (flight.get("ident_iata") or "").upper()
    is_codeshare = q in codeshares_iata or q in codeshares_icao
    is_operating = q in {operating_ident, operating_iata} and not is_codeshare

    return { # this is a dictionary that contains the flight information
        "ident": flight.get("ident") or flight.get("ident_iata"),
        "ident_icao": flight.get("ident_icao"),
        "ident_iata": flight.get("ident_iata"),
        "flight_number": flight.get("flight_number"),
        "operator": flight.get("operator_iata") or flight.get("operator"),
        "operator_icao": flight.get("operator_icao") or flight.get("operator"),
        "registration": flight.get("registration"),  # tail number when present
        "aircraft_type": flight.get("aircraft_type"),
        "status": flight.get("status"),
        "cancelled": flight.get("cancelled"),
        "diverted": flight.get("diverted"),
        # REVIEW: explicit codeshare / operating-flight context for the synthesizer
        "query_match": {
            "queried_ident": q,
            "is_operating_flight": is_operating or (not is_codeshare and bool(operating_ident)),
            "is_codeshare_of_this_flight": is_codeshare,
            "operating_flight_ident": operating_ident or None,
            "operating_flight_iata": operating_iata or None,
        },
        "origin": {
            "code": _airport_code(origin),
            # Keep ICAO explicitly for NOAA METAR enrichment (code alone may be IATA).
            "code_icao": _extract_airport_icao(origin),
            "name": origin.get("name"),
            "city": origin.get("city"),
            "gate": flight.get("gate_origin"),
            "terminal": flight.get("terminal_origin"),
        },
        "destination": {
            "code": _airport_code(dest),
            "code_icao": _extract_airport_icao(dest),
            "name": dest.get("name"),
            "city": dest.get("city"),
            "gate": flight.get("gate_destination"),
            "terminal": flight.get("terminal_destination"),
            "baggage_claim": flight.get("baggage_claim"),
        },
        "departure_delay_sec": flight.get("departure_delay"),
        "arrival_delay_sec": flight.get("arrival_delay"),
        "scheduled_out": flight.get("scheduled_out"),
        "estimated_out": flight.get("estimated_out"),
        "actual_out": flight.get("actual_out"),
        "scheduled_in": flight.get("scheduled_in"),
        "estimated_in": flight.get("estimated_in"),
        "actual_in": flight.get("actual_in"),
        "progress_percent": flight.get("progress_percent"),
        "route_distance_nm": flight.get("route_distance"),
        "codeshares_iata": codeshares_iata,
        "codeshares_icao": codeshares_icao,  # REVIEW: ICAO codeshare list too
    }


def _format_flightaware_payload(
    ident: str,
    payload: dict[str, Any],
    *,
    source: str,
    max_flights: int = 5,
) -> dict[str, Any]:
    """Turn raw AeroAPI JSON into a compact, agent-friendly summary.

    REVIEW: ranks codeshare/operating matches first and adds a `headline`
    so the synthesizer cannot miss status / aircraft / route.
    """
    raw_flights = payload.get("flights")
    if not isinstance(raw_flights, list):
        raw_flights = []

    summaries = [ # this is a list of dictionaries that contain the flight information
        _summarize_flight(f, ident)
        for f in raw_flights[:max_flights]
        if isinstance(f, dict)
    ]

    # REVIEW: prefer legs where the queried number is operating or codeshare
    q = ident.upper()
    ranked = sorted(
        summaries,
        key=lambda s: (
            0
            if q
            in {
                str(s.get("ident") or "").upper(),
                str(s.get("ident_iata") or "").upper(),
                str(s.get("ident_icao") or "").upper(),
            }
            or q in (s.get("codeshares_iata") or [])
            or q in (s.get("codeshares_icao") or [])
            else 1,
            0 if (s.get("status") or "").lower() not in {"arrived", "result unknown"} else 1,
        ),
    )
    shown = ranked[:max_flights]
    headline = shown[0] if shown else None

    # Shape the UI + synthesizer both read.
    # Metro UI mostly uses `headline` (+ weather_context if attached later).
    # Full `flights` list stays for the agent; users do not see raw JSON.
    return {
        "source": source,
        "ident": ident,
        "flight_count_returned": len(raw_flights),
        "flights_shown": len(shown),
        # One-glance block: status, aircraft, route, codeshare, key times
        "headline": (
            {
                "status": headline.get("status"),
                "aircraft_type": headline.get("aircraft_type"),
                "registration": headline.get("registration"),
                "operating_flight": (headline.get("query_match") or {}).get(
                    "operating_flight_ident"
                ),
                "operating_flight_iata": (headline.get("query_match") or {}).get(
                    "operating_flight_iata"
                ),
                # True when the user asked for a codeshare number (e.g. AI8303)
                "queried_is_codeshare": (headline.get("query_match") or {}).get(
                    "is_codeshare_of_this_flight"
                ),
                "route": {
                    "origin": (headline.get("origin") or {}).get("code"),
                    "destination": (headline.get("destination") or {}).get("code"),
                },
                "scheduled_out": headline.get("scheduled_out"),
                "estimated_in": headline.get("estimated_in"),
            }
            if headline
            else None
        ),
        "flights": shown,  # richer list for the synthesizer; UI cards skip this
        "note": (
            f"Showing up to {max_flights} AeroAPI flights for {ident}. "
            "If queried_is_codeshare is true, aircraft/status belong to the operating flight. "
            "Delays are in seconds when provided by FlightAware."
        ),
    }


def _icao_ident_candidates(ident: str) -> list[str]:
    """Build lookup order: as-given, then IATA→ICAO rewrite when applicable.

    REVIEW: e.g. AI8303 -> also try AIC8303 if the first response is empty.
    """
    clean = ident.strip().upper()
    candidates = [clean]
    match = re.fullmatch(r"([A-Z]{2})(\d{1,5})", clean)
    if match:
        iata, number = match.group(1), match.group(2)
        icao = IATA_TO_ICAO_AIRLINE.get(iata)
        if icao:
            alt = f"{icao}{number}"
            if alt not in candidates:
                candidates.append(alt)
    return candidates


@traceable(
    name="FlightAware_AeroAPI_GET",
    run_type="tool",
    hide_inputs=lambda inputs: {
        "ident": inputs.get("ident"),
        "api_key": "[REDACTED]",
    },
)
def _aeroapi_get_flights(ident: str, api_key: str) -> dict[str, Any]:
    """GET /flights/{ident} via requests+certifi.

    REVIEW: replaces urllib.request.urlopen, which raised
    SSL CERTIFICATE_VERIFY_FAILED on this machine and made the agent
    falsely report 'no live status' for AI8303.
    """
    path_ident = urllib.parse.quote(ident.strip().upper(), safe="")
    url = f"{AEROAPI_BASE}/flights/{path_ident}"
    response = requests.get(
        url,
        headers={
            "x-apikey": api_key,
            "Accept": "application/json",
            "User-Agent": "aviation-agent/1.0",
        },
        timeout=25,
        verify=certifi.where(),  # REVIEW: use certifi CA bundle explicitly
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# NOAA Aviation Weather Center — METAR enrichment (single HTTP call)
#
# Easy idea:
#   Only when a flight looks delayed / held / diverted, grab weather for
#   origin + destination in ONE request (ids=ORIG,DEST).
#   Attach `weather_context` onto the FlightAware JSON.
#   Metro UI shows a short "Weather nearby" block from that — not raw METAR.
# ---------------------------------------------------------------------------

@traceable(name="NOAA_METAR_Batch_GET", run_type="tool")
def fetch_noaa_metars(icao_codes: list[str]) -> dict[str, dict]:
    """Fetch METARs for one or more ICAO stations in a SINGLE NOAA request.

    Simple idea:
    - You pass ["VIDP", "WSSS"] (origin + destination).
    - We call NOAA once with ids=VIDP,WSSS.
    - NOAA returns weather for both airports in one response.

    WHY comma-join ids:
    - `ids=VOBL,VOMM` asks AWC for both origin and destination weather at once.
    - One network round-trip (~200ms) instead of two sequential GETs.

    Returns:
        { "VIDP": {"rawOb": "...", "fltcat": "VFR", "temp": 28, ...}, ... }
    """
    # --- Step 1: clean the input list ---
    # Drop empties / None, keep only real 4-letter ICAOs, remove duplicates.
    cleaned: list[str] = []
    seen: set[str] = set()
    for code in icao_codes:
        if not code:
            continue
        icao = str(code).strip().upper()
        # ICAO airport codes are 4 letters (KJFK, VIDP). Skip junk / dupes.
        if len(icao) != 4 or icao in seen:
            continue
        seen.add(icao)
        cleaned.append(icao)

    # Nothing usable → no HTTP call
    if not cleaned:
        return {}

    # --- Step 2: one GET for all stations ---
    # Example URL: .../metar?ids=VIDP,WSSS&format=json
    # safe=',' keeps the comma un-encoded so NOAA still sees multiple ids.
    ids_param = ",".join(cleaned)
    url = f"{NOAA_METAR_URL}?ids={urllib.parse.quote(ids_param, safe=',')}&format=json"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "AviationFleetAgent/1.0"},  # identify our client
            timeout=4,  # weather is enrichment only — fail fast, don't block FlightAware path
            verify=certifi.where(),
        )
        response.raise_for_status() # raise an exception if the response is not successful
        data = response.json()
    except Exception:  # noqa: BLE001
        # Network/API failure → empty dict; caller still has FlightAware data
        return {}

    # --- Step 3: normalize NOAA rows into {ICAO: compact fields} ---
    # AWC JSON is typically a list of observation dicts with icaoId.
    rows = data if isinstance(data, list) else []
    result: dict[str, dict] = {} # initialize an empty dictionary to store the results
    for row in rows:
        if not isinstance(row, dict): # if the row is not a dictionary, continue to the next row        
            continue
        # Prefer icaoId (AWC field name); fall back to icao if present
        station = str(row.get("icaoId") or row.get("icao") or "").strip().upper()
        if not station:
            continue
        # Keep only the fields the synthesizer / turn_notes need
        result[station] = {
            "rawOb": row.get("rawOb") or row.get("raw_ob"),      # full METAR text
            "fltcat": row.get("fltcat"),                        # VFR/IFR/etc.
            "temp": row.get("temp"),                            # temperature
            "wspd": row.get("wspd"),                            # wind speed
            "wxString": row.get("wxString") or row.get("wx_string"),  # weather string
        }
    return result


def _delay_seconds(flight: dict[str, Any], *keys: str) -> float | None:
    """Read the first present numeric delay field (raw AeroAPI or summarized).

    Simple idea:
    - Raw FlightAware uses "departure_delay".
    - Our summarized flights use "departure_delay_sec".
    - Pass both key names; return the first numeric value found.
    """
    for key in keys:
        val = flight.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _has_weather_anomaly(flight: dict[str, Any]) -> bool:
    """True when status/delays suggest weather context would help explain ops.

    Simple idea — only fetch NOAA when something looks off:
    - status text mentions DELAY / HOLD / DIVERT, OR
    - departure or arrival delay is > 15 minutes (900 seconds).

    On-time flights skip weather to save latency/noise.
    """
    status = str(flight.get("status") or "").upper()
    # Keyword anomaly in the status string
    if any(token in status for token in ("DELAY", "HOLD", "DIVERT")):
        return True

    # Numeric delay anomaly (check both raw + summarized field names)
    dep = _delay_seconds(flight, "departure_delay", "departure_delay_sec")
    arr = _delay_seconds(flight, "arrival_delay", "arrival_delay_sec")
    if dep is not None and dep > WEATHER_DELAY_ANOMALY_SEC:
        return True
    if arr is not None and arr > WEATHER_DELAY_ANOMALY_SEC:
        return True
    return False


def _ops_phase(flight: dict[str, Any]) -> str:
    """Classify phase for primary weather station selection.

    Simple idea:
    - Still on ground / not departed → origin weather matters most for delays.
    - Airborne, holding, or diverted → destination weather matters most.

    Returns:
      - "predeparture_or_ground": origin weather is primary delay context
      - "enroute_or_divert": destination weather is primary delay context
    """
    status = str(flight.get("status") or "").upper()
    diverted = bool(flight.get("diverted"))
    progress = flight.get("progress_percent")

    # Divert / hold → care about where it's going (or divert field)
    if diverted or "DIVERT" in status or "HOLD" in status:
        return "enroute_or_divert"
    # Explicitly in the air
    if "EN ROUTE" in status or "ENROUTE" in status or "IN FLIGHT" in status:
        return "enroute_or_divert"
    # Progress > 0 usually means airborne / underway
    if isinstance(progress, (int, float)) and progress > 0:
        return "enroute_or_divert"
    # Actual out/off times mean it already left the gate/runway
    if flight.get("actual_out") or flight.get("actual_off"):
        return "enroute_or_divert"
    # Default: still pre-departure / ground delay world
    return "predeparture_or_ground"


def _metar_summary_line(icao: str, metar: dict[str, Any] | None, *, primary: bool) -> str:
    """One-line METAR blurb for headline / turn_notes.

    Simple idea: turn a METAR dict into a short readable string like:
      "VIDP (PRIMARY): fltcat=VFR, temp=28, wspd=6 | METAR text..."
    """
    tag = "PRIMARY" if primary else "secondary"
    if not metar:
        return f"{icao} ({tag}): METAR unavailable"

    # Compact field snippets; skip empties/None so the line stays clean
    bits = [
        f"fltcat={metar.get('fltcat')}",
        f"temp={metar.get('temp')}",
        f"wspd={metar.get('wspd')}",
        f"wx={metar.get('wxString')}",
    ]
    raw = metar.get("rawOb")
    raw_snip = f" | {raw}" if raw else ""
    return (
        f"{icao} ({tag}): "
        + ", ".join(str(b) for b in bits if b.split("=", 1)[-1] not in {"None", ""})
        + raw_snip
    )


def _select_primary_raw_flight(payload: dict[str, Any], query_ident: str) -> dict[str, Any] | None:
    """Pick the same AeroAPI flight object our formatter would headline.

    Simple idea:
    - AeroAPI returns many legs (past/future).
    - We want the SAME flight we put in `headline` for weather enrichment.
    - Rank: prefer query match (operating or codeshare), then prefer non-arrived.
    - Return the raw FlightAware flight dict (has origin.code_icao, delays, etc.).
    """
    raw_flights = payload.get("flights")
    if not isinstance(raw_flights, list) or not raw_flights:
        return None

    # Pair each summary with its raw flight so we can return the raw object later
    summaries = [
        (_summarize_flight(f, query_ident), f)
        for f in raw_flights
        if isinstance(f, dict)
    ]
    if not summaries:
        return None

    q = query_ident.upper()

    def _rank_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int]:
        """Lower tuple sorts first: (matched?, stale?)."""
        s, _raw = item
        # 0 = this leg matches the user's flight/codeshare; 1 = unrelated leg
        matched = (
            0
            if q
            in {
                str(s.get("ident") or "").upper(),
                str(s.get("ident_iata") or "").upper(),
                str(s.get("ident_icao") or "").upper(),
            }
            or q in (s.get("codeshares_iata") or [])
            or q in (s.get("codeshares_icao") or [])
            else 1
        )
        # Prefer active/upcoming over already-arrived when ranking
        stale = 0 if (s.get("status") or "").lower() not in {"arrived", "result unknown"} else 1
        return (matched, stale)

    summaries.sort(key=_rank_key)
    # [0] = best (summary, raw); [1] = raw FlightAware flight object
    return summaries[0][1]


def _enrich_formatted_with_noaa(
    formatted: dict[str, Any],
    raw_flight: dict[str, Any],
) -> dict[str, Any]:
    """Conditionally attach NOAA weather_context when a delay/hold/divert anomaly exists.

    Simple end-to-end flow:
    1) If no anomaly → return FlightAware payload unchanged (no NOAA call).
    2) Read origin/dest ICAO from FlightAware airport objects.
    3) One NOAA call for both stations.
    4) Decide which airport weather is PRIMARY based on phase.
    5) Attach weather_context + headline.weather_summary for the synthesizer.

    ICAO extraction happens from the FlightAware nested airport objects on the
    raw flight (origin.code_icao / destination.code_icao), not from guesswork.
    """
    # Gate: only enrich when ops look abnormal
    if not _has_weather_anomaly(raw_flight):
        return formatted

    # Nested airport objects from AeroAPI (not top-level payload keys)
    origin_obj = raw_flight.get("origin") if isinstance(raw_flight.get("origin"), dict) else {}
    dest_obj = (
        raw_flight.get("destination")
        if isinstance(raw_flight.get("destination"), dict)
        else {}
    )

    # Extract ICAO directly from FlightAware payload airport nodes.
    # Equivalent to: origin_icao = payload_flight["origin"].get("code_icao")
    # (with IATA→ICAO local fallback inside _extract_airport_icao)
    origin_icao = _extract_airport_icao(origin_obj)
    destination_icao = _extract_airport_icao(dest_obj)

    # One AWC GET retrieves BOTH stations: ids=ORIG,DEST (single ~200ms RTT).
    metars = fetch_noaa_metars([origin_icao or "", destination_icao or ""])

    phase = _ops_phase(raw_flight)
    # Phase-aware primary: ground/pre-dep → origin; enroute/hold/divert → destination
    if phase == "predeparture_or_ground":
        primary_icao = origin_icao
        primary_reason = "pre-departure / ground delay → origin weather is primary context"
    else:
        primary_icao = destination_icao
        primary_reason = "en-route / holding / diverted → destination weather is primary context"

    # Human-readable lines for headline / turn_notes
    summary_lines = [
        _metar_summary_line(
            origin_icao or "ORIGIN?",
            metars.get(origin_icao or ""),
            primary=(origin_icao is not None and origin_icao == primary_icao),
        ),
        _metar_summary_line(
            destination_icao or "DEST?",
            metars.get(destination_icao or ""),
            primary=(destination_icao is not None and destination_icao == primary_icao),
        ),
    ]

    # Structured block the synthesizer can trust
    weather_context = {
        "anomaly": True,
        "phase": phase,
        "primary_icao": primary_icao,
        "primary_reason": primary_reason,
        "origin_icao": origin_icao,
        "destination_icao": destination_icao,
        # Station -> parsed METAR fields from the single NOAA call
        "metars": metars,
        "summary_lines": summary_lines,
        "note": (
            "NOAA METARs fetched in one request via comma-separated ids "
            f"({origin_icao},{destination_icao})."
        ),
    }
    formatted["weather_context"] = weather_context

    # Append weather summaries onto headline so synthesizer / turn_notes see them.
    headline = formatted.get("headline")
    if isinstance(headline, dict):
        headline["weather_primary_icao"] = primary_icao
        headline["weather_phase"] = phase
        headline["weather_summary"] = summary_lines
        formatted["headline"] = headline

    # Keep a human-readable note for turn_notes consumers.
    prior_note = str(formatted.get("note") or "")
    weather_note = " | ".join(summary_lines)
    formatted["note"] = (
        f"{prior_note} Weather enrichment: {primary_reason}. {weather_note}"
    ).strip()
    return formatted


def _mock_flightaware_response(ident: str) -> dict[str, Any]:
    """Offline fallback shaped like our normalized FlightAware summary."""
    return {
        "source": "mock_flightaware",
        "ident": ident,
        "flight_count_returned": 1,
        "flights_shown": 1,
        "flights": [
            {
                "ident": ident.upper(),
                "ident_iata": None,
                "flight_number": None,
                "operator": "AAL",
                "registration": "N12345",
                "aircraft_type": "B738",
                "status": "En Route / On Time",
                "cancelled": False,
                "diverted": False,
                "origin": {
                    "code": "DFW",
                    "name": "Dallas/Fort Worth Intl",
                    "city": "Dallas",
                    "gate": "A20",
                    "terminal": None,
                },
                "destination": {
                    "code": "ORD",
                    "name": "Chicago O'Hare Intl",
                    "city": "Chicago",
                    "gate": "B12",
                    "terminal": None,
                    "baggage_claim": "5",
                },
                "departure_delay_sec": 720,
                "arrival_delay_sec": 900,
                "scheduled_out": "2024-01-15T18:00:00Z",
                "estimated_out": "2024-01-15T18:12:00Z",
                "actual_out": None,
                "scheduled_in": "2024-01-15T20:30:00Z",
                "estimated_in": "2024-01-15T20:45:00Z",
                "actual_in": None,
                "progress_percent": 55,
                "route_distance_nm": 802,
                "codeshares_iata": [],
            }
        ],
        "note": "Mock telemetry. Set FLIGHTAWARE_API_KEY for live AeroAPI data.",
    }


@tool("flightaware_live_status", args_schema=FlightAwareInput)
def flightaware_live_status(ident: str) -> str:
    """Fetch live flight operational telemetry from FlightAware AeroAPI.

    Uses GET /flights/{ident} for status, gates, ETAs, registration/tail
    context, and delay-related fields. Not for general web news research.

    REVIEW: loops IATA then ICAO candidates; returns flightaware_empty
    instead of silently pretending there is no flight when SSL/API fails mid-way.
    """
    key = _usable_key("FLIGHTAWARE_API_KEY")
    clean_ident = ident.strip().upper()

    # No key -> return structured mock so the graph still runs
    if not key:
        return json.dumps(_mock_flightaware_response(clean_ident), default=str)

    # Collect failure reasons across candidates so we can return them if nothing works
    errors: list[str] = []
    try:
        # Build lookup list, e.g. AI8303 -> ["AI8303", "AIC8303"] (IATA first, then ICAO)
        # REVIEW: try as-given (IATA), then ICAO rewrite if needed
        for candidate in _icao_ident_candidates(clean_ident):
            try:
                # Live GET /flights/{candidate} via requests+certifi; returns parsed JSON dict
                payload = _aeroapi_get_flights(candidate, key)
            except requests.HTTPError as exc:
                # AeroAPI answered with a non-2xx status (401/404/429/etc.)
                body = ""  # default empty body if response object is missing
                if exc.response is not None:
                    # Keep only first 500 chars of error body so turn_notes stay readable
                    body = exc.response.text[:500]
                # Record which candidate failed and why, then try the next candidate
                errors.append(
                    f"{candidate}: HTTP {exc.response.status_code if exc.response else '?'}: {body}"
                )
                continue  # skip to next IATA/ICAO candidate
            except requests.RequestException as exc:
                # Network/SSL/timeout/connection errors (not an HTTP status from AeroAPI)
                errors.append(f"{candidate}: {exc}")  # store the exception message
                continue  # skip to next candidate

            # AeroAPI success shape is {"flights": [ ... ]}; coerce bad shapes to []
            flights = payload.get("flights") if isinstance(payload.get("flights"), list) else []
            if not flights:
                # 200 OK but no legs for this ident — try next candidate (e.g. ICAO form)
                errors.append(f"{candidate}: empty flights list")
                continue  # skip formatting; try the next candidate

            formatted = _format_flightaware_payload(
                clean_ident,
                payload,
                source="flightaware_aeroapi",
            )
            formatted["resolved_ident"] = candidate  # REVIEW: which candidate succeeded

            # NOAA enrichment: parse origin/dest ICAO from this FlightAware payload,
            # and if delay/hold/divert anomaly → one AWC call for both stations.
            primary_raw = _select_primary_raw_flight(payload, clean_ident)
            if primary_raw is not None:
                formatted = _enrich_formatted_with_noaa(formatted, primary_raw)

            return json.dumps(formatted, default=str)

        # REVIEW: honest empty result (not a fake AAL mock) when all candidates fail
        return json.dumps(
            {
                "source": "flightaware_empty",
                "ident": clean_ident,
                "flight_count_returned": 0,
                "flights": [],
                "headline": None,
                "tried": _icao_ident_candidates(clean_ident),
                "errors": errors,
                "note": "AeroAPI returned no flight legs for this ident.",
            },
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "source": "flightaware_error_fallback",
                "ident": clean_ident,
                "error": str(exc),
                "errors": errors,
                "mock": _mock_flightaware_response(clean_ident),
            },
            default=str,
        )

# ---------------------------------------------------------------------------
# Tavily — web research (news / fleet / advisories). NOT live radar.
#
# UI note (metro_progress.format_tavily_card):
#   - live source "tavily"  → show up to 3 title + blurb + link
#   - source "mock_tavily" / errors → UI only shows "Web search not connected"
#     (mock articles below still exist for the synthesizer / offline tests)
# ---------------------------------------------------------------------------

def _mock_tavily_response(query: str) -> dict[str, Any]:
    """Offline fallback payload when TAVILY_API_KEY is missing or the call fails.

    The Streamlit metro UI intentionally does NOT render these mock articles;
    it only shows "Web search not connected." The synthesizer may still see them.
    """
    return {
        "source": "mock_tavily",  # UI treats this as "not connected"
        "query": query,
        "results": [
            {
                "title": "Airline fleet utilization update (mock)",
                "url": "https://example.com/mock-fleet-news",
                "content": (
                    "Mock press note: carrier adjusting narrowbody utilization "
                    "ahead of seasonal demand; no official grounding order."
                ),
            },
            {
                "title": "Engine / airworthiness advisory digest (mock)",
                "url": "https://example.com/mock-ad-alert",
                "content": (
                    "Mock advisory: operators reminded to track service bulletins "
                    "for next-gen engines; not a live flight status feed."
                ),
            },
            {
                "title": "Regional weather operations advisory (mock)",
                "url": "https://example.com/mock-weather",
                "content": (
                    "Mock weather desk: convective activity may affect Midwest "
                    "hub banks this evening; plan for gate holds."
                ),
            },
        ],
        # Plain flag string; UI looks at source=mock_tavily more than this note
        "note": "web search not connected",
    }


@tool("tavily_aviation_web_search", args_schema=TavilyAviationInput)
def tavily_aviation_web_search(query: str) -> str:
    """Search the web for unstructured aviation research via Tavily.

    Use for airline press, fleet acquisition news, engine recall alerts,
    and general weather advisories. Do NOT use for live radar/telemetry.
    """
    key = _usable_key("TAVILY_API_KEY")

    if not key:
        return json.dumps(_mock_tavily_response(query), default=str)

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        result = client.search(
            query=f"aviation airline fleet operations news: {query}",
            search_depth="basic",
            max_results=5,
        )
        return json.dumps(
            {"source": "tavily", "query": query, "data": result},
            default=str,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "source": "tavily_error_fallback",
                "query": query,
                "error": str(exc),
                "mock": _mock_tavily_response(query),
            },
            default=str,
        )
