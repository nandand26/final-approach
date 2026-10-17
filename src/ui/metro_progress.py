"""Metro-board progress UI for one agent turn.

Plain-English idea:
  While the agent runs, show a fixed "train line" of stations.
  Lights change as work moves along: waiting → running → done / skipped / warn.
  When a spoke finishes, expanders show a SHORT human-readable card — not raw JSON.

Files that call this:
  app.py streams LangGraph updates → advance_on_node_update() → render_progress()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Friendly node & spoke display names (graph ids → operational labels)
# ---------------------------------------------------------------------------
NODE_DISPLAY_MAP = {
    "supervisor": "🧭 Flight Ops Router",
    "spoke_1_node": "🛰️ Physical Airspace Radar Engine",
    "spoke_2_node": "🛫 Commercial Flight Engine",
    "spoke_3_node": "🔎 Web Research Engine",
    "synthesizer": "📋 Intelligence Synthesizer",
    # Real LangGraph node / worker ids
    "spoke_1_opensky": "🛰️ Physical Airspace Radar Engine",
    "spoke_2_liveops": "🛫 Commercial Flight Engine",
    "spoke_3_websearch": "🔎 Web Research Engine",
    "Spoke_1_OpenSky": "🛰️ Physical Airspace Radar Engine",
    "Spoke_2_LiveOps": "🛫 Commercial Flight Engine",
    "Spoke_3_WebSearch": "🔎 Web Research Engine",
    "FINISH": "📋 Intelligence Synthesizer",
}

_VALID_PLAN_WORKERS = {
    "Spoke_1_OpenSky",
    "Spoke_2_LiveOps",
    "Spoke_3_WebSearch",
}

# Canonical metro track (visual order). Skipped spokes stay muted gray.
_CANONICAL_TRACK: list[tuple[str, str, str | None]] = [
    ("supervisor", "supervisor", None),
    ("spoke_3", "spoke", "Spoke_3_WebSearch"),
    ("spoke_2", "spoke", "Spoke_2_LiveOps"),
    ("spoke_1", "spoke", "Spoke_1_OpenSky"),
    ("finish", "finish", None),
]

# Metro light icons (emoji fallback; HTML pills add color/pulse)
_STATUS_ICON = {
    "pending": "⚪",
    "active": "🔵",
    "done": "🟢",
    "skipped": "⚫",
    "warn": "🟠",
}


@dataclass
class Station:
    """One stop on the metro line for this user question."""

    key: str
    kind: str  # "supervisor" | "spoke" | "finish"
    label: str
    worker: str | None = None
    status: str = "pending"  # pending | active | done | skipped | warn
    detail_md: str | None = None
    summary: str | None = None  # one-line payload under the station
    metrics: dict[str, str] | None = None  # Spoke 1 altitude / gs / etc.


@dataclass
class TurnProgress:
    """All UI state for ONE user message (one turn)."""

    stations: list[Station] = field(default_factory=list)
    plan_built: bool = False
    status_line: str = "Planning route…"
    final_report: str | None = None
    is_partial: bool = False
    # Sub-steps for st.status() live log (router decisions, spoke summaries)
    status_log: list[str] = field(default_factory=list)
    # Planned spoke workers for this turn (execution order)
    plan_workers: list[str] = field(default_factory=list)


def human_worker_label(worker: str) -> str:
    """Turn Spoke_1_OpenSky into the friendly operational name."""
    return NODE_DISPLAY_MAP.get(worker, worker.replace("_", " "))


def node_display_name(node_name: str) -> str:
    """Friendly label for a LangGraph node id."""
    return NODE_DISPLAY_MAP.get(node_name, node_name.replace("_", " "))


def _plan_worker_names(plan: list[Any]) -> list[str]:
    """Normalize plan items (strings or {worker, task} dicts) to worker names."""
    names: list[str] = []
    for item in plan or []:
        if isinstance(item, dict):
            worker = str(item.get("worker") or "").strip()
        else:
            worker = str(item or "").strip()
        if worker in _VALID_PLAN_WORKERS:
            names.append(worker)
    return names


def build_canonical_route(plan: list[Any]) -> list[Station]:
    """Fixed metro line; spokes not in `plan` start as skipped.

    Track sequence:
      Flight Ops Router → Web Research → Commercial Flight → Physical Airspace → Synthesizer
    """
    planned = set(_plan_worker_names(plan))
    stations: list[Station] = []
    for key, kind, worker in _CANONICAL_TRACK:
        if kind == "supervisor":
            label = NODE_DISPLAY_MAP["supervisor"]
        elif kind == "finish":
            label = NODE_DISPLAY_MAP["synthesizer"]
        else:
            label = human_worker_label(worker or key)

        status = "pending"
        if kind == "spoke" and worker and worker not in planned:
            status = "skipped"

        stations.append(
            Station(
                key=key,
                kind=kind,
                label=label,
                worker=worker,
                status=status,
            )
        )
    return stations


def build_route_from_plan(plan: list[Any]) -> list[Station]:
    """Alias used by app / tests — builds the fixed canonical metro track."""
    return build_canonical_route(plan)


def parse_plan_workers_from_note(note: str) -> list[str] | None:
    """Read the planned spokes out of a supervisor turn_note string.

    Returns:
      - list of spoke names, or [] for empty plan / chat only
      - None if this note is not a plan-generation note (e.g. a later hop)
    """
    match = re.search(
        r"Plan generated[^:]*:\s*(\[[^\]]*\])",
        note,
        flags=re.IGNORECASE,
    )
    if not match:
        if "Empty plan" in note or "conversational/direct bypass" in note:
            return []
        return None

    raw_list = match.group(1)
    workers: list[str] = []
    try:
        parsed = json.loads(raw_list.replace("'", '"'))
        if isinstance(parsed, list):
            workers = _plan_worker_names(parsed)
    except json.JSONDecodeError:
        workers = []

    if not workers:
        found = re.findall(
            r"Spoke_[123]_(?:OpenSky|LiveOps|WebSearch)",
            raw_list,
        )
        seen: set[str] = set()
        for w in found:
            if w in _VALID_PLAN_WORKERS and w not in seen:
                workers.append(w)
                seen.add(w)

    return [w for w in workers if w in _VALID_PLAN_WORKERS]


# ---------------------------------------------------------------------------
# Small text helpers for the human-readable cards
# ---------------------------------------------------------------------------

def _delay_minutes(seconds: Any) -> str | None:
    if not isinstance(seconds, (int, float)):
        return None
    mins = int(round(float(seconds) / 60.0))
    if mins <= 0:
        return None
    return f"{mins} min"


def _trim(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _extract_json_blobs(text: str) -> list[Any]:
    blobs: list[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(text, i)
            blobs.append(value)
            i = end
        except json.JSONDecodeError:
            i += 1
    return blobs


def _first_dict_blob(text: str) -> dict[str, Any] | None:
    for blob in _extract_json_blobs(text):
        if isinstance(blob, dict):
            return blob
    return None


def _looks_like_api_error(payload: dict[str, Any] | None, note: str = "") -> bool:
    """True when a spoke returned an API / rate-limit style failure.

    Prefer structured payload fields. Avoid bare ``"429" in note`` — Tavily
    relevance scores like ``0.57064295`` contain that digit run and were
    false-flagging successful Web Research as a warning.
    """
    if isinstance(payload, dict):
        if payload.get("ok") is False:
            return True
        source = str(payload.get("source") or "").lower()
        if "error" in source or source.endswith("_fallback"):
            return True
        if payload.get("error") or payload.get("detail"):
            return True
        # Known live/mock success wrappers: trust source, skip fragile text scans.
        if source in {
            "tavily",
            "flightaware",
            "opensky",
            "mock_tavily",
            "mock_flightaware",
            "skipped",
        }:
            return False

    joined = note.lower()
    if "rate limit" in joined or "rate-limit" in joined or "too many requests" in joined:
        return True
    # HTTP 429 as its own token — not digits inside floats/ids (…64295…).
    if re.search(r"(?<![\d.])429(?![\d.])", joined):
        return True
    if not isinstance(payload, dict):
        return "error" in joined and (
            "opensky" in joined or "flightaware" in joined or "tavily" in joined
        )
    return False


# ---------------------------------------------------------------------------
# Card formatters — turn tool JSON into simple markdown for expanders
# ---------------------------------------------------------------------------

def format_flightaware_card(payload: dict[str, Any]) -> str:
    if payload.get("source") == "skipped":
        return "_No flight number found in this chat yet._"

    lines: list[str] = ["#### Live flight"]
    ident = payload.get("ident") or payload.get("resolved_ident")
    if ident:
        lines.append(f"**Flight:** `{ident}`")

    headline = payload.get("headline") if isinstance(payload.get("headline"), dict) else None
    if not headline:
        note = payload.get("note")
        if note:
            lines.append(str(note))
        else:
            lines.append("_No live flight details available._")
        return "\n\n".join(lines)

    status = headline.get("status")
    if status:
        lines.append(f"**Status:** {status}")

    route = headline.get("route") if isinstance(headline.get("route"), dict) else {}
    origin = route.get("origin")
    dest = route.get("destination")
    if origin or dest:
        lines.append(f"**Route:** {origin or '?'} → {dest or '?'}")

    aircraft = headline.get("aircraft_type")
    registration = headline.get("registration")
    craft_bits = []
    if aircraft:
        craft_bits.append(str(aircraft))
    if registration:
        craft_bits.append(f"tail {registration}")
    if craft_bits:
        lines.append(f"**Aircraft:** {' · '.join(craft_bits)}")

    if headline.get("queried_is_codeshare"):
        op = headline.get("operating_flight_iata") or headline.get("operating_flight")
        if op:
            lines.append(f"**Operated as:** `{op}` (your number is a codeshare)")

    if headline.get("scheduled_out"):
        lines.append(f"**Scheduled departure:** {headline['scheduled_out']}")
    if headline.get("estimated_in"):
        lines.append(f"**Estimated arrival:** {headline['estimated_in']}")

    flights = payload.get("flights") if isinstance(payload.get("flights"), list) else []
    primary = flights[0] if flights and isinstance(flights[0], dict) else {}
    dep = _delay_minutes(primary.get("departure_delay_sec"))
    arr = _delay_minutes(primary.get("arrival_delay_sec"))
    delay_bits = []
    if dep:
        delay_bits.append(f"{dep} departure")
    if arr:
        delay_bits.append(f"{arr} arrival")
    if delay_bits:
        lines.append(f"**Delay:** {', '.join(delay_bits)}")

    weather = payload.get("weather_context")
    if isinstance(weather, dict) and weather.get("anomaly"):
        lines.append("")
        lines.append("#### Weather nearby")
        phase = str(weather.get("phase") or "")
        if "predeparture" in phase:
            why = "Focusing on departure-airport weather"
        else:
            why = "Focusing on arrival-airport weather"
        primary_icao = weather.get("primary_icao")
        if primary_icao:
            lines.append(f"**{why}** (`{primary_icao}`)")

        metars = weather.get("metars") if isinstance(weather.get("metars"), dict) else {}
        for icao, metar in metars.items():
            if not isinstance(metar, dict):
                continue
            bits = []
            if metar.get("fltcat"):
                bits.append(str(metar["fltcat"]))
            if metar.get("temp") is not None:
                bits.append(f"{metar['temp']}°C")
            if metar.get("wspd") is not None:
                bits.append(f"wind {metar['wspd']} kt")
            if metar.get("wxString"):
                bits.append(str(metar["wxString"]))
            tag = "main" if icao == primary_icao else "other"
            if bits:
                lines.append(f"- **{icao}** ({tag}): {', '.join(bits)}")
            else:
                lines.append(f"- **{icao}** ({tag}): weather unavailable")

    return "\n\n".join(lines)


def format_tavily_card(payload: dict[str, Any]) -> str:
    lines: list[str] = ["#### Related reading"]
    source = str(payload.get("source") or "")

    if source in {"mock_tavily", "tavily_error_fallback"} or not _usable_live_tavily(payload):
        lines.append("_Web search not connected._")
        return "\n\n".join(lines)

    results: list[dict[str, Any]] = []
    if source == "tavily":
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            results = [r for r in data["results"] if isinstance(r, dict)]
        elif isinstance(payload.get("results"), list):
            results = [r for r in payload["results"] if isinstance(r, dict)]
    elif isinstance(payload.get("results"), list):
        results = [r for r in payload["results"] if isinstance(r, dict)]

    if not results:
        lines.append("_No related articles found._")
        return "\n\n".join(lines)

    for idx, item in enumerate(results[:3], start=1):
        title = str(item.get("title") or "Article").strip()
        content = _trim(str(item.get("content") or item.get("snippet") or ""))
        url = str(item.get("url") or "").strip()
        block = f"**{idx}. {title}**"
        if content:
            block += f"\n\n{content}"
        if url:
            block += f"\n\n[Read more]({url})"
        lines.append(block)

    return "\n\n".join(lines)


def _usable_live_tavily(payload: dict[str, Any]) -> bool:
    source = str(payload.get("source") or "")
    if source in {"mock_tavily", "tavily_error_fallback"}:
        return False
    return source == "tavily"


def extract_spoke1_metrics(note: str) -> dict[str, str] | None:
    """Pull Altitude / Ground Speed / Vertical Rate / Transponder Hex for columns."""
    payload = _first_dict_blob(note)
    if not isinstance(payload, dict) or payload.get("ok") is False:
        return None

    data = payload.get("data")
    state: dict[str, Any] | None = None
    if isinstance(data, dict) and isinstance(data.get("states"), list) and data["states"]:
        first = data["states"][0]
        if isinstance(first, dict):
            state = first
    if state is None:
        # Fallback: show icao24 from track / args when no live vector
        icao = None
        if isinstance(data, dict) and data.get("icao24"):
            icao = data.get("icao24")
        else:
            m = re.search(r"['\"]icao24['\"]\s*:\s*['\"]([0-9a-fA-F]{6})['\"]", note)
            if m:
                icao = m.group(1)
        if not icao:
            return None
        return {
            "Altitude": "—",
            "Ground Speed": "—",
            "Vertical Rate": "—",
            "Transponder Hex": str(icao),
        }

    def _fmt(val: Any, unit: str) -> str:
        if val is None:
            return "—"
        try:
            num = float(val)
            if unit == "ft":
                return f"{int(round(num * 3.28084)):,} ft"
            if unit == "kt":
                return f"{int(round(num * 1.94384))} kt"
            if unit == "fpm":
                return f"{int(round(num * 196.85))} fpm"
            return str(val)
        except (TypeError, ValueError):
            return str(val)

    return {
        "Altitude": _fmt(state.get("baro_altitude"), "ft"),
        "Ground Speed": _fmt(state.get("velocity"), "kt"),
        "Vertical Rate": _fmt(state.get("vertical_rate"), "fpm"),
        "Transponder Hex": str(state.get("icao24") or "—"),
    }


def spoke1_summary(note: str, metrics: dict[str, str] | None) -> str:
    if metrics and metrics.get("Altitude") not in {None, "—"}:
        return f"Fetched {metrics['Altitude']} vector"
    if metrics and metrics.get("Transponder Hex") not in {None, "—"}:
        return f"Queried `{metrics['Transponder Hex']}`"
    tool_match = re.search(r"tool=([a-z_]+)", note)
    if tool_match:
        return f"Ran `{tool_match.group(1)}`"
    return "Airspace query complete"


def spoke2_summary(note: str, fa_payload: dict[str, Any] | None) -> str:
    # Codeshare resolve: AI8303 → SIA403 style
    if isinstance(fa_payload, dict):
        headline = fa_payload.get("headline") if isinstance(fa_payload.get("headline"), dict) else {}
        queried = fa_payload.get("ident") or fa_payload.get("resolved_ident")
        if headline.get("queried_is_codeshare"):
            op = headline.get("operating_flight_iata") or headline.get("operating_flight")
            if queried and op:
                return f"Resolved {queried} to {op}"
        status = headline.get("status")
        callsign = queried or headline.get("registration")
        if callsign and status:
            return f"{callsign}: {status}"
        if callsign:
            return f"Fetched commercial status for {callsign}"
    ident_match = re.search(r"Resolved ident from conversation:\s*(\S+)", note)
    if ident_match and ident_match.group(1) != "NONE":
        return f"Commercial lookup for {ident_match.group(1)}"
    return "Commercial flight pass complete"


def spoke3_summary(note: str) -> str:
    query_match = re.search(r"\[Spoke_3_WebSearch\]\s*Query:\s*(.+)", note)
    if query_match:
        return f"Researched: {_trim(query_match.group(1), 80)}"
    if "tavily_error" in note.lower():
        return "Web research warning"
    return "Web research complete"


def format_spoke1_card(note: str) -> str:
    lines: list[str] = ["#### Physical airspace (OpenSky)"]

    tool_match = re.search(r"tool=([a-z_]+)", note)
    if tool_match:
        lines.append(f"**Tool:** `{tool_match.group(1)}`")

    payload = _first_dict_blob(note)
    if not isinstance(payload, dict):
        lines.append("_No OpenSky payload parsed._")
        return "\n\n".join(lines)

    if payload.get("ok") is False:
        lines.append(f"⚠️ **OpenSky error:** {payload.get('error')}")
        return "\n\n".join(lines)

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("states"), list):
        states = [s for s in data["states"] if isinstance(s, dict)]
        lines.append(f"**Live vectors:** {len(states)} aircraft")
        for s in states[:5]:
            bits = []
            if s.get("icao24"):
                bits.append(f"`{s['icao24']}`")
            if s.get("callsign"):
                bits.append(str(s["callsign"]).strip())
            if s.get("baro_altitude") is not None:
                bits.append(f"alt {s['baro_altitude']} m")
            if s.get("velocity") is not None:
                bits.append(f"gs {s['velocity']} m/s")
            if s.get("true_track") is not None:
                bits.append(f"trk {s['true_track']}°")
            if s.get("on_ground") is True:
                bits.append("on ground")
            if bits:
                lines.append("- " + " · ".join(bits))
        return "\n\n".join(lines)

    if isinstance(data, list):
        lines.append(f"**Movements / legs:** {len(data)} (showing up to 5)")
        for row in data[:5]:
            if not isinstance(row, dict):
                continue
            bits = []
            for key, label in (
                ("icao24", "icao24"),
                ("callsign", "callsign"),
                ("estDepartureAirport", "from"),
                ("estArrivalAirport", "to"),
                ("firstSeen", "firstSeen"),
                ("lastSeen", "lastSeen"),
            ):
                if row.get(key) is not None and str(row.get(key)).strip():
                    bits.append(f"{label}: {row[key]}")
            if bits:
                lines.append("- " + " · ".join(bits))
        return "\n\n".join(lines)

    if isinstance(data, dict) and data.get("path") is not None:
        path = data.get("path") or []
        lines.append(f"**Track points:** {len(path) if isinstance(path, list) else 'n/a'}")
        if data.get("icao24"):
            lines.append(f"**icao24:** `{data.get('icao24')}`")
        return "\n\n".join(lines)

    lines.append("_OpenSky returned data (see synthesizer for details)._")
    return "\n\n".join(lines)


def format_spoke2_card(note: str) -> str:
    """Commercial Flight Engine card — FlightAware (+ NOAA if present). No Tavily."""
    fa_payload: dict[str, Any] | None = None

    for blob in _extract_json_blobs(note):
        if isinstance(blob, dict) and (
            "headline" in blob
            or "flights" in blob
            or str(blob.get("source", "")).startswith("flightaware")
            or blob.get("source") in {"skipped", "mock_flightaware"}
        ):
            fa_payload = blob
            break
    if fa_payload is None:
        for blob in _extract_json_blobs(note):
            if isinstance(blob, dict):
                fa_payload = blob
                break

    if fa_payload is not None:
        return format_flightaware_card(fa_payload)
    return "#### Live flight\n\n_Live flight details unavailable._"


def format_spoke3_card(note: str) -> str:
    """Web Research Engine card — Tavily results."""
    query_match = re.search(r"\[Spoke_3_WebSearch\]\s*Query:\s*(.+)", note)
    query = query_match.group(1).strip() if query_match else ""

    tv_payload: dict[str, Any] | None = None
    for blob in _extract_json_blobs(note):
        if isinstance(blob, dict) and (
            str(blob.get("source", "")).startswith("tavily")
            or str(blob.get("source", "")).startswith("mock_tavily")
            or "results" in blob
            or isinstance(blob.get("data"), dict)
        ):
            tv_payload = blob
            break

    parts: list[str] = ["#### Web research (Tavily)"]
    if query:
        parts.append(f"**Query:** {_trim(query, 160)}")
    if tv_payload is not None:
        parts.append(format_tavily_card(tv_payload))
    else:
        parts.append("_Web search payload unavailable._")
    return "\n\n".join(parts)


def _parse_spoke2_fa(note: str) -> dict[str, Any] | None:
    for blob in _extract_json_blobs(note):
        if isinstance(blob, dict) and (
            "headline" in blob
            or "flights" in blob
            or str(blob.get("source", "")).startswith("flightaware")
            or blob.get("source") in {"skipped", "mock_flightaware", "flightaware_error_fallback"}
        ):
            return blob
    return _first_dict_blob(note)


def format_supervisor_card(note: str, plan: list[str] | None) -> str:
    if plan is not None:
        if not plan:
            return "No tools needed — routing straight to the Intelligence Synthesizer."
        labels = [human_worker_label(w) for w in plan]
        return "**Plan:** " + " → ".join(labels)
    clean = re.sub(r"^\[supervisor\]\s*", "", note).strip()
    return _trim(clean, 240)


def metro_strip_markdown(stations: list[Station]) -> str:
    """Plain markdown fallback strip (also used in snapshots)."""
    if not stations:
        return "**Route:** Planning…"
    parts: list[str] = []
    for station in stations:
        icon = _STATUS_ICON.get(station.status, "⚪")
        label = station.label
        if station.status == "active":
            label = f"**{label}**"
        elif station.status == "skipped":
            label = f"~~{label}~~"
        parts.append(f"{icon} {label}")
    return " → ".join(parts)


def metro_strip_html(stations: list[Station]) -> str:
    """HTML/CSS pill track with pulse on the active station."""
    if not stations:
        return "<p><strong>Route:</strong> Planning…</p>"

    styles = {
        "pending": ("#e8ecf1", "#5b6b7c", "1px solid #c5ced8"),
        "active": ("#dff0ff", "#0b5cab", "2px solid #1a73e8"),
        "done": ("#e6f6ea", "#1b7a3d", "1px solid #34a853"),
        "skipped": ("#f1f2f4", "#9aa3ad", "1px dashed #c0c6cc"),
        "warn": ("#fff4e5", "#a15c00", "1px solid #f5a623"),
    }
    icons = {
        "pending": "○",
        "active": "●",
        "done": "✓",
        "skipped": "–",
        "warn": "!",
    }

    css = """
    <style>
    @keyframes metro-pulse {
      0% { box-shadow: 0 0 0 0 rgba(26,115,232,0.45); }
      70% { box-shadow: 0 0 0 10px rgba(26,115,232,0); }
      100% { box-shadow: 0 0 0 0 rgba(26,115,232,0); }
    }
    .metro-track {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem;
      margin: 0.25rem 0 0.75rem 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    .metro-pill {
      display: inline-flex; flex-direction: column; gap: 0.15rem;
      padding: 0.45rem 0.7rem; border-radius: 10px; min-width: 7.5rem;
      max-width: 14rem;
    }
    .metro-pill .title { font-size: 0.82rem; font-weight: 600; line-height: 1.25; }
    .metro-pill .meta { font-size: 0.72rem; opacity: 0.9; line-height: 1.2; }
    .metro-pill.active { animation: metro-pulse 1.4s ease-out infinite; }
    .metro-pill.skipped .title { opacity: 0.55; }
    .metro-arrow { color: #8a94a0; font-weight: 700; padding: 0 0.1rem; }
    </style>
    """

    chunks = [css, '<div class="metro-track">']
    for i, station in enumerate(stations):
        bg, fg, border = styles.get(station.status, styles["pending"])
        icon = icons.get(station.status, "○")
        cls = f"metro-pill {station.status}"
        meta = ""
        if station.status == "active":
            meta = '<div class="meta">In Progress…</div>'
        elif station.status == "warn" and station.summary:
            meta = f'<div class="meta">⚠ {station.summary}</div>'
        elif station.status == "done" and station.summary:
            meta = f'<div class="meta">{station.summary}</div>'
        elif station.status == "skipped":
            meta = '<div class="meta">Skipped</div>'

        chunks.append(
            f'<div class="{cls}" style="background:{bg};color:{fg};border:{border};">'
            f'<div class="title">{icon} {station.label}</div>'
            f"{meta}</div>"
        )
        if i < len(stations) - 1:
            chunks.append('<span class="metro-arrow">➔</span>')
    chunks.append("</div>")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# State machine: each streamed LangGraph node update advances the lights
# ---------------------------------------------------------------------------

def _log(progress: TurnProgress, message: str) -> None:
    progress.status_log.append(message)


def _station_by_worker(progress: TurnProgress, worker: str) -> Station | None:
    for station in progress.stations:
        if station.worker == worker:
            return station
    return None


def _station_by_kind(progress: TurnProgress, kind: str) -> Station | None:
    for station in progress.stations:
        if station.kind == kind:
            return station
    return None


def _activate_next_from_plan(progress: TurnProgress) -> None:
    """Light the next planned spoke that is still pending, else the synthesizer.

    If a station is already active (e.g. spoke just activated after completion,
    and supervisor emits a later hop), leave the train where it is.
    """
    if any(s.status == "active" for s in progress.stations):
        active = next(s for s in progress.stations if s.status == "active")
        progress.status_line = f"Now: {active.label}"
        return

    for worker in progress.plan_workers:
        station = _station_by_worker(progress, worker)
        if station and station.status == "pending":
            station.status = "active"
            progress.status_line = f"Now: {station.label}"
            _log(progress, f"In progress: {station.label}")
            return
    finish = _station_by_kind(progress, "finish")
    if finish and finish.status == "pending":
        finish.status = "active"
        progress.status_line = f"Now: {finish.label}"
        _log(progress, f"In progress: {finish.label}")
        return
    progress.status_line = "Finishing…"


def advance_on_node_update(progress: TurnProgress, node_name: str, update: dict[str, Any]) -> None:
    """Handle one streamed update from app_graph.stream(...)."""
    notes = [str(n) for n in (update.get("turn_notes") or [])]

    if update.get("final_report"):
        progress.final_report = str(update["final_report"])

    # --- First supervisor visit: learn the plan and draw the whole route ---
    if not progress.plan_built and node_name == "supervisor":
        for note in notes:
            plan = parse_plan_workers_from_note(note)
            if plan is None:
                continue
            progress.plan_workers = list(plan)
            progress.stations = build_canonical_route(plan)
            progress.plan_built = True
            router = _station_by_kind(progress, "supervisor")
            if router:
                router.status = "done"
                router.detail_md = format_supervisor_card(note, plan)
                if plan:
                    router.summary = " → ".join(human_worker_label(w) for w in plan)
                    _log(
                        progress,
                        f"{NODE_DISPLAY_MAP['supervisor']}: routing to "
                        + " → ".join(human_worker_label(w) for w in plan),
                    )
                else:
                    router.summary = "Direct to synthesizer"
                    _log(
                        progress,
                        f"{NODE_DISPLAY_MAP['supervisor']}: empty plan → "
                        f"{NODE_DISPLAY_MAP['synthesizer']}",
                    )
            _activate_next_from_plan(progress)
            return
        progress.status_line = "Planning route…"
        _log(progress, f"{NODE_DISPLAY_MAP['supervisor']}: planning…")
        return

    # --- Later supervisor visits (after a spoke returns) ---
    if node_name == "supervisor":
        for note in notes:
            clean = re.sub(r"^\[supervisor\]\s*", "", note).strip()
            if "Continuing plan" in note or "→" in note:
                _log(progress, f"{NODE_DISPLAY_MAP['supervisor']}: {_trim(clean, 140)}")
            elif "FINISH" in note or "Plan queue empty" in note:
                _log(
                    progress,
                    f"{NODE_DISPLAY_MAP['supervisor']}: handoff → "
                    f"{NODE_DISPLAY_MAP['synthesizer']}",
                )
        _activate_next_from_plan(progress)
        return

    if node_name == "spoke_1_opensky":
        _complete_spoke(progress, "Spoke_1_OpenSky", notes)
        return

    if node_name == "spoke_2_liveops":
        _complete_spoke(progress, "Spoke_2_LiveOps", notes)
        return

    if node_name == "spoke_3_websearch":
        _complete_spoke(progress, "Spoke_3_WebSearch", notes)
        return

    if node_name == "synthesizer":
        for station in progress.stations:
            if station.status == "active":
                station.status = "done"
        finish = _station_by_kind(progress, "finish")
        if finish:
            finish.status = "done"
            finish.detail_md = "Answer ready."
            finish.summary = "Report synthesized"
        # Any still-pending planned stations → skipped (should already be marked)
        for station in progress.stations:
            if station.status == "pending":
                station.status = "skipped"
        progress.status_line = "Answer ready"
        _log(progress, f"{NODE_DISPLAY_MAP['synthesizer']}: complete")
        return


def _complete_spoke(progress: TurnProgress, worker: str, notes: list[str]) -> None:
    joined = "\n".join(notes)
    station = _station_by_worker(progress, worker)
    if station is None:
        _activate_next_from_plan(progress)
        return

    if worker == "Spoke_1_OpenSky":
        detail = format_spoke1_card(joined)
        metrics = extract_spoke1_metrics(joined)
        payload = _first_dict_blob(joined)
        warn = _looks_like_api_error(payload, joined)
        summary = spoke1_summary(joined, metrics)
        station.metrics = metrics
        station.detail_md = detail
        station.summary = summary if not warn else _trim(
            str((payload or {}).get("error") or "API / rate-limit warning"),
            100,
        )
        station.status = "warn" if warn else "done"
        _log(
            progress,
            f"{station.label}: {'⚠️ ' if warn else ''}{station.summary}",
        )
    elif worker == "Spoke_3_WebSearch":
        detail = format_spoke3_card(joined)
        payload = _first_dict_blob(joined)
        warn = _looks_like_api_error(payload, joined) or (
            "tavily_error" in joined.lower()
        )
        summary = spoke3_summary(joined)
        station.detail_md = detail
        station.summary = summary if not warn else _trim(
            str((payload or {}).get("error") or "Web research warning"),
            100,
        )
        station.status = "warn" if warn else "done"
        _log(
            progress,
            f"{station.label}: {'⚠️ ' if warn else ''}{station.summary}",
        )
    else:
        detail = format_spoke2_card(joined)
        fa = _parse_spoke2_fa(joined)
        warn = _looks_like_api_error(fa, joined)
        if "flightaware_error" in joined.lower():
            warn = True
        summary = spoke2_summary(joined, fa)
        station.detail_md = detail
        station.summary = summary if not warn else _trim(
            str((fa or {}).get("error") or (fa or {}).get("note") or "API / rate-limit warning"),
            100,
        )
        station.status = "warn" if warn else "done"
        _log(
            progress,
            f"{station.label}: {'⚠️ ' if warn else ''}{station.summary}",
        )

    _activate_next_from_plan(progress)


def mark_remaining_skipped(progress: TurnProgress) -> None:
    """Recursion limit hit: grey out stations we never reached."""
    for station in progress.stations:
        if station.status in {"pending", "active"}:
            station.status = "skipped"
    finish = _station_by_kind(progress, "finish")
    if finish:
        finish.status = "done"
        finish.detail_md = "Partial answer (run stopped early)."
        finish.summary = "Partial synthesis"
    progress.is_partial = True
    progress.status_line = "Partial answer"
    _log(progress, "Run stopped early — unfinished stations skipped.")


def progress_snapshot(progress: TurnProgress) -> dict[str, Any]:
    """Freeze the metro board so history can show it collapsed after rerun."""
    details: list[dict[str, Any]] = []
    for station in progress.stations:
        if station.status in {"done", "warn"} and station.detail_md and station.kind != "finish":
            details.append(
                {
                    "label": station.label,
                    "markdown": station.detail_md,
                    "status": station.status,
                    "summary": station.summary or "",
                    "metrics": station.metrics or {},
                }
            )
    return {
        "strip": metro_strip_markdown(progress.stations),
        "strip_html": metro_strip_html(progress.stations),
        "status_line": progress.status_line,
        "status_log": list(progress.status_log),
        "details": details,
        "is_partial": progress.is_partial,
        "stations": [
            {
                "key": s.key,
                "label": s.label,
                "status": s.status,
                "summary": s.summary,
                "kind": s.kind,
                "metrics": s.metrics,
            }
            for s in progress.stations
        ],
    }


def render_spoke1_metrics(metrics: dict[str, str] | None) -> None:
    """st.columns metrics: Altitude, Ground Speed, Vertical Rate, Transponder Hex."""
    if not metrics:
        return
    import streamlit as st

    keys = ("Altitude", "Ground Speed", "Vertical Rate", "Transponder Hex")
    cols = st.columns(4)
    for col, key in zip(cols, keys):
        with col:
            st.metric(key, metrics.get(key, "—"))


def render_agent_steps_panel(
    *,
    strip: str,
    status_line: str,
    details: list[dict[str, Any]],
    expanded: bool,
    strip_html: str | None = None,
) -> None:
    """Metro strip + per-station cards (used live or inside a collapsed expander)."""
    import streamlit as st

    if strip_html:
        st.markdown(strip_html, unsafe_allow_html=True)
    else:
        st.markdown(strip)
    if status_line:
        st.caption(status_line)
    for item in details:
        label = str(item.get("label") or "Step")
        body = str(item.get("markdown") or "")
        status = str(item.get("status") or "done")
        prefix = "⚠️ " if status == "warn" else ""
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else None
        with st.expander(f"{prefix}{label} details", expanded=False):
            if metrics:
                render_spoke1_metrics(metrics)
            st.markdown(body)


def render_progress(progress: TurnProgress) -> None:
    """Draw the live turn UI.

    While running: metro board is fully visible (hero progress).
    When the answer is ready: answer first, board tucked into a collapsed
    "Agent steps" expander so demos can still open it.
    """
    import streamlit as st

    details = [
        {
            "label": s.label,
            "markdown": s.detail_md or "",
            "status": s.status,
            "summary": s.summary or "",
            "metrics": s.metrics or {},
        }
        for s in progress.stations
        if s.status in {"done", "warn"} and s.detail_md and s.kind != "finish"
    ]
    strip = metro_strip_markdown(progress.stations)
    strip_html = metro_strip_html(progress.stations)

    # Live Spoke 1 metrics while that station just completed / is showing
    spoke1 = _station_by_worker(progress, "Spoke_1_OpenSky")
    show_live_metrics = bool(
        spoke1
        and spoke1.metrics
        and spoke1.status in {"done", "warn", "active"}
        and not progress.final_report
    )

    if progress.final_report:
        label = "Partial answer" if progress.is_partial else "Answer"
        st.markdown(f"### {label}")
        st.markdown(progress.final_report)
        with st.expander("Agent steps", expanded=False):
            render_agent_steps_panel(
                strip=strip,
                strip_html=strip_html,
                status_line=progress.status_line,
                details=details,
                expanded=False,
            )
        return

    st.markdown("### Metro route")
    render_agent_steps_panel(
        strip=strip,
        strip_html=strip_html,
        status_line=progress.status_line,
        details=details,
        expanded=True,
    )
    if show_live_metrics and spoke1 and spoke1.metrics:
        st.markdown("##### Airspace vector")
        render_spoke1_metrics(spoke1.metrics)
