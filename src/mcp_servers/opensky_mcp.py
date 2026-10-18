"""OpenSky Network MCP server — Physical Airspace Engine (Spoke 1).

Standalone FastMCP server wrapping the OpenSky REST API (same surface as the
official opensky_api Python client: get_states, airport arrivals/departures,
flights-by-aircraft, tracks).

Run as stdio MCP server:
  python -m src.mcp_servers.opensky_mcp

Spoke 1 imports the same tool functions in-process (Streamlit-friendly), so the
graph does not need a separate MCP subprocess for demos.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# OpenSky REST root (official Network API)
OPENSKY_API_ROOT = "https://opensky-network.org/api"
# OAuth2 token endpoint (client credentials grant)
OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
# Refresh a bit early so we never send an almost-expired Bearer token
_TOKEN_REFRESH_MARGIN_SEC = 30

# Create MCP server instance (name visible to MCP clients)
mcp = FastMCP("OpenSky-Physical-Airspace")

# Cached OAuth access token: (access_token, expires_at_epoch)
_oauth_cache: dict[str, Any] = {"token": None, "expires_at": 0.0} # check if the token is still valid, if not, refresh the token


# ---------------------------------------------------------------------------
# Shared helpers — normalize inputs + call OpenSky safely
# ---------------------------------------------------------------------------

def _normalize_icao24(icao24: str | None) -> str | None:
    """OpenSky expects lowercase hex icao24 (e.g. 3C675A → 3c675a)."""
    if icao24 is None:
        return None
    cleaned = str(icao24).strip().lower()
    return cleaned or None


def _to_epoch(value: Any, *, default: int | None = None) -> int | None:
    """Accept Unix int/float/str or ISO-8601 string → epoch seconds (UTC)."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return default
    # Pure digits → already epoch
    if text.isdigit():
        return int(text)
    try:
        # Allow trailing Z
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        logger.warning("Could not parse time value %r; using default=%s", value, default)
        return default


def _error_json(message: str, **extra: Any) -> str:
    """Structured error payload (never raise out of tools)."""
    payload = {"ok": False, "error": message, **extra}
    return json.dumps(payload, default=str)


def _ok_json(data: Any, **meta: Any) -> str:
    payload = {"ok": True, "data": data, **meta}
    return json.dumps(payload, default=str)


def _usable_env(name: str) -> str:
    """Return env value if it looks real (not a .env.example placeholder)."""
    value = os.getenv(name, "").strip()
    if not value or "your_" in value.lower() or value.endswith("_here"):
        return ""
    return value


def _get_oauth_bearer() -> str | None:
    """Fetch/cache OpenSky OAuth2 client-credentials access token.

    Uses OPENSKY_CLIENT_ID + OPENSKY_CLIENT_SECRET from .env
    (same flow as the official opensky_api TokenManager).
    """
    client_id = _usable_env("OPENSKY_CLIENT_ID")
    client_secret = _usable_env("OPENSKY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    now = time.time()
    cached = _oauth_cache.get("token")
    expires_at = float(_oauth_cache.get("expires_at") or 0.0)
    if cached and now < expires_at - _TOKEN_REFRESH_MARGIN_SEC:
        return str(cached)

    response = requests.post(
        OPENSKY_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("OpenSky token response missing access_token")
    expires_in = int(data.get("expires_in") or 1800)
    _oauth_cache["token"] = token
    _oauth_cache["expires_at"] = now + expires_in
    logger.debug("OpenSky OAuth token refreshed (valid ~%ss).", expires_in)
    return str(token)


def _opensky_basic_auth() -> tuple[str, str] | None:
    """Legacy username/password basic auth (optional fallback)."""
    user = _usable_env("OPENSKY_USERNAME")
    password = _usable_env("OPENSKY_PASSWORD")
    if user and password:
        return (user, password)
    return None


def _opensky_get(path: str, params: dict[str, Any] | None = None, *, timeout: float = 10.0) -> Any:
    """GET OpenSky REST endpoint; raises requests errors for the caller to catch.

    Auth preference:
      1) OAuth Bearer from client_id/client_secret
      2) Basic auth username/password
      3) Anonymous (lowest rate limits)
    """
    url = f"{OPENSKY_API_ROOT}{path}"
    headers: dict[str, str] = {}
    auth = None

    bearer = _get_oauth_bearer()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    else:
        auth = _opensky_basic_auth()

    response = requests.get(
        url,
        params=params or {},
        headers=headers,
        auth=auth,
        timeout=timeout,
    )
    # 404 often means "no data in window" for flight/airport endpoints
    if response.status_code == 404:
        return None
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()


def _parse_bbox(bbox: Any) -> tuple[float, float, float, float] | None:
    """Normalize bbox to (min_lat, max_lat, min_lon, max_lon)."""
    if bbox is None or bbox == "":
        return None
    if isinstance(bbox, str):
        parts = [p.strip() for p in bbox.split(",")]
        if len(parts) != 4:
            return None
        bbox = parts
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return None


def _clamp_window(begin: int, end: int, *, max_seconds: int) -> tuple[int, int]:
    """Ensure begin < end and window length ≤ max_seconds."""
    if begin >= end:
        end = begin + min(3600, max_seconds)
    if end - begin > max_seconds:
        begin = end - max_seconds
    return begin, end


# ---------------------------------------------------------------------------
# Core tool implementations (also imported by Spoke 1)
# ---------------------------------------------------------------------------

def get_live_aircraft_telemetry(icao24: str | None = None, bbox: Any = None) -> str:
    """Fetches real-time physical flight vectors (altitude, ground speed, true track, vertical rate, squawk code, on_ground status). DO NOT use for published flight schedules, commercial flight numbers (e.g. SQ403), or airline equipment lookup (e.g. A380)."""
    try:
        hex_id = _normalize_icao24(icao24)
        if not hex_id:
            return _error_json(
                "icao24 is required (24-bit transponder hex). "
                "Commercial flight numbers (e.g. AI8303) belong to Spoke 2.",
                tool="get_live_aircraft_telemetry",
            )

        params: dict[str, Any] = {"icao24": hex_id}
        box = _parse_bbox(bbox)
        if box is not None:
            # OpenSky bbox query params
            min_lat, max_lat, min_lon, max_lon = box
            params.update(
                {
                    "lamin": min_lat,
                    "lamax": max_lat,
                    "lomin": min_lon,
                    "lomax": max_lon,
                }
            )

        raw = _opensky_get("/states/all", params)
        states = []
        time_epoch = None
        if isinstance(raw, dict):
            time_epoch = raw.get("time")
            for row in raw.get("states") or []:
                if not isinstance(row, (list, tuple)) or len(row) < 17:
                    continue
                # Official state-vector column order
                states.append(
                    {
                        "icao24": row[0],
                        "callsign": (row[1] or "").strip() or None,
                        "origin_country": row[2],
                        "time_position": row[3],
                        "last_contact": row[4],
                        "longitude": row[5],
                        "latitude": row[6],
                        "baro_altitude": row[7],
                        "on_ground": row[8],
                        "velocity": row[9],
                        "true_track": row[10],
                        "vertical_rate": row[11],
                        "sensors": row[12],
                        "geo_altitude": row[13],
                        "squawk": row[14],
                        "spi": row[15],
                        "position_source": row[16],
                    }
                )

        return _ok_json(
            {"time": time_epoch, "states": states},
            tool="get_live_aircraft_telemetry",
            icao24=hex_id,
        )
    except requests.Timeout:
        return _error_json("OpenSky get_states timed out.", tool="get_live_aircraft_telemetry")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_live_aircraft_telemetry failed")
        return _error_json(str(exc), tool="get_live_aircraft_telemetry")


def get_airport_arrivals(
    airport: str | None = None,
    begin: Any = None,
    end: Any = None,
) -> str:
    """Retrieves actual/estimated arrival movements for a specific airport within a time window (max 7 days). DO NOT use for commercial flight route searches or scheduled flight availability."""
    try:
        apt = str(airport or "").strip().upper()
        if len(apt) != 4:
            return _error_json(
                "airport must be a 4-letter ICAO code (e.g. VIDP, WSSS, EDDF).",
                tool="get_airport_arrivals",
            )

        now = int(time.time())
        begin_e = _to_epoch(begin, default=now - 2 * 3600)
        end_e = _to_epoch(end, default=now)
        assert begin_e is not None and end_e is not None
        begin_e, end_e = _clamp_window(begin_e, end_e, max_seconds=7 * 24 * 3600)

        raw = _opensky_get(
            "/flights/arrival",
            {"airport": apt, "begin": begin_e, "end": end_e},
        )
        return _ok_json(
            raw or [],
            tool="get_airport_arrivals",
            airport=apt,
            begin=begin_e,
            end=end_e,
        )
    except requests.Timeout:
        return _error_json("OpenSky arrivals timed out.", tool="get_airport_arrivals")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_airport_arrivals failed")
        return _error_json(str(exc), tool="get_airport_arrivals")


def get_airport_departures(
    airport: str | None = None,
    begin: Any = None,
    end: Any = None,
) -> str:
    """Retrieves actual/estimated departure movements for a specific airport within a time window (max 7 days). DO NOT use for scheduled airline timetables."""
    try:
        apt = str(airport or "").strip().upper()
        if len(apt) != 4:
            return _error_json(
                "airport must be a 4-letter ICAO code (e.g. VIDP, WSSS, EDDF).",
                tool="get_airport_departures",
            )

        now = int(time.time())
        begin_e = _to_epoch(begin, default=now - 2 * 3600)
        end_e = _to_epoch(end, default=now)
        assert begin_e is not None and end_e is not None
        begin_e, end_e = _clamp_window(begin_e, end_e, max_seconds=7 * 24 * 3600)

        raw = _opensky_get(
            "/flights/departure",
            {"airport": apt, "begin": begin_e, "end": end_e},
        )
        return _ok_json(
            raw or [],
            tool="get_airport_departures",
            airport=apt,
            begin=begin_e,
            end=end_e,
        )
    except requests.Timeout:
        return _error_json("OpenSky departures timed out.", tool="get_airport_departures")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_airport_departures failed")
        return _error_json(str(exc), tool="get_airport_departures")


def get_aircraft_flight_history(
    icao24: str | None = None,
    begin: Any = None,
    end: Any = None,
) -> str:
    """Retrieves historical flight legs operated by a specific 24-bit transponder address (icao24) over the past month."""
    try:
        hex_id = _normalize_icao24(icao24)
        if not hex_id:
            return _error_json(
                "icao24 is required (24-bit transponder hex).",
                tool="get_aircraft_flight_history",
            )

        now = int(time.time())
        begin_e = _to_epoch(begin, default=now - 24 * 3600)
        end_e = _to_epoch(end, default=now)
        assert begin_e is not None and end_e is not None
        begin_e, end_e = _clamp_window(begin_e, end_e, max_seconds=30 * 24 * 3600)

        raw = _opensky_get(
            "/flights/aircraft",
            {"icao24": hex_id, "begin": begin_e, "end": end_e},
        )
        return _ok_json(
            raw or [],
            tool="get_aircraft_flight_history",
            icao24=hex_id,
            begin=begin_e,
            end=end_e,
        )
    except requests.Timeout:
        return _error_json("OpenSky aircraft history timed out.", tool="get_aircraft_flight_history")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_aircraft_flight_history failed")
        return _error_json(str(exc), tool="get_aircraft_flight_history")


def get_aircraft_trajectory_track(icao24: str | None = None, t: Any = 0) -> str:
    """Retrieves 3D spatial waypoints (latitude, longitude, barometric altitude, true track) along a flight path."""
    try:
        hex_id = _normalize_icao24(icao24)
        if not hex_id:
            return _error_json(
                "icao24 is required (24-bit transponder hex).",
                tool="get_aircraft_trajectory_track",
            )

        t_epoch = _to_epoch(t, default=0) or 0
        raw = _opensky_get("/tracks/all", {"icao24": hex_id, "time": t_epoch})
        return _ok_json(raw or {}, tool="get_aircraft_trajectory_track", icao24=hex_id, t=t_epoch)
    except requests.Timeout:
        return _error_json("OpenSky track timed out.", tool="get_aircraft_trajectory_track")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_aircraft_trajectory_track failed")
        return _error_json(str(exc), tool="get_aircraft_trajectory_track")


# Registry used by Spoke 1 to dispatch by tool name
OPENSKY_TOOLS = {
    "get_live_aircraft_telemetry": get_live_aircraft_telemetry,
    "get_airport_arrivals": get_airport_arrivals,
    "get_airport_departures": get_airport_departures,
    "get_aircraft_flight_history": get_aircraft_flight_history,
    "get_aircraft_trajectory_track": get_aircraft_trajectory_track,
}


def invoke_opensky_tool(tool_name: str, **kwargs: Any) -> str:
    """Dispatch a single OpenSky tool by name (Spoke 1 client entrypoint)."""
    import inspect

    fn = OPENSKY_TOOLS.get(tool_name)
    if fn is None:
        return _error_json(
            f"Unknown OpenSky tool: {tool_name}",
            allowed=list(OPENSKY_TOOLS),
        )
    # Only pass kwargs the tool actually accepts (drop None + unknown keys)
    sig = inspect.signature(fn)
    clean = {
        k: v
        for k, v in kwargs.items()
        if v is not None and k in sig.parameters
    }
    # Required params missing (e.g. Gemini chose telemetry but left icao24=None
    # on a commercial-flight-only question) → structured error, never TypeError.
    missing = [
        name
        for name, param in sig.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in clean
    ]
    if missing:
        return _error_json(
            f"{tool_name} missing required argument(s): {', '.join(missing)}. "
            "Provide an icao24 hex / ICAO airport, or use Spoke 2 for commercial flight numbers.",
            tool=tool_name,
            provided=list(clean),
        )
    try:
        return fn(**clean)
    except TypeError as exc:
        return _error_json(str(exc), tool=tool_name, provided=list(clean))


# ---------------------------------------------------------------------------
# FastMCP tool registrations (stdio MCP surface)
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_get_live_aircraft_telemetry(icao24: str, bbox: str | None = None) -> str:
    """Fetches real-time physical flight vectors (altitude, ground speed, true track, vertical rate, squawk code, on_ground status). DO NOT use for published flight schedules, commercial flight numbers (e.g. SQ403), or airline equipment lookup (e.g. A380)."""
    return get_live_aircraft_telemetry(icao24=icao24, bbox=bbox)


@mcp.tool()
def tool_get_airport_arrivals(airport: str, begin: str | int, end: str | int) -> str:
    """Retrieves actual/estimated arrival movements for a specific airport within a time window (max 7 days). DO NOT use for commercial flight route searches or scheduled flight availability."""
    return get_airport_arrivals(airport=airport, begin=begin, end=end)


@mcp.tool()
def tool_get_airport_departures(airport: str, begin: str | int, end: str | int) -> str:
    """Retrieves actual/estimated departure movements for a specific airport within a time window (max 7 days). DO NOT use for scheduled airline timetables."""
    return get_airport_departures(airport=airport, begin=begin, end=end)


@mcp.tool()
def tool_get_aircraft_flight_history(icao24: str, begin: str | int, end: str | int) -> str:
    """Retrieves historical flight legs operated by a specific 24-bit transponder address (icao24) over the past month."""
    return get_aircraft_flight_history(icao24=icao24, begin=begin, end=end)


@mcp.tool()
def tool_get_aircraft_trajectory_track(icao24: str, t: str | int = 0) -> str:
    """Retrieves 3D spatial waypoints (latitude, longitude, barometric altitude, true track) along a flight path."""
    return get_aircraft_trajectory_track(icao24=icao24, t=t)


if __name__ == "__main__":
    # stdio MCP server entrypoint
    mcp.run()
