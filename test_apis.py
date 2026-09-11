"""Quick interactive smoke tests for OpenAI, Gemini, and FlightAware.

Usage (from aviation_agent/):
  source .venv/bin/activate
  python test_apis.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

AEROAPI_BASE = "https://aeroapi.flightaware.com/aeroapi"


def _usable(name: str) -> bool:
    value = os.getenv(name, "")
    return bool(value) and "your_" not in value and "api_key_here" not in value


def test_openai() -> None:
    """One prompt → one ChatGPT response."""
    if not _usable("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY missing in .env")
        return

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    model_name = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
    prompt = input("Enter one prompt for ChatGPT: ").strip()
    if not prompt:
        print("No prompt entered.")
        return

    print(f"Calling OpenAI model={model_name} ...")
    llm = ChatOpenAI(model=model_name, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    print("\n--- OpenAI response ---")
    print(response.content)
    print("--- OK ---")


def test_gemini() -> None:
    """One prompt → one Gemini Flash response."""
    if not _usable("GOOGLE_API_KEY"):
        print("FAIL: GOOGLE_API_KEY missing in .env")
        return

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    model_name = os.getenv("SPOKE_MODEL", "gemini-2.5-flash")
    prompt = input("Enter one prompt for Gemini: ").strip()
    if not prompt:
        print("No prompt entered.")
        return

    print(f"Calling Gemini model={model_name} ...")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    # Newer Gemini models may return a list of content blocks, not a plain string
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(text_parts)
    print("\n--- Gemini response ---")
    print(content)
    print("--- OK ---")


def test_flightaware() -> None:
    """One flight ident / tail → AeroAPI GET /flights/{ident}."""
    if not _usable("FLIGHTAWARE_API_KEY"):
        print("FAIL: FLIGHTAWARE_API_KEY missing in .env")
        return

    # Yes: one ident is enough (flight code like UAL123 or tail like N12345)
    ident = input("Enter one flight ident or tail (e.g. UAL123 or N12345): ").strip().upper()
    if not ident:
        print("No ident entered.")
        return

    path_ident = urllib.parse.quote(ident, safe="")
    url = f"{AEROAPI_BASE}/flights/{path_ident}"
    request = urllib.request.Request(
        url,
        headers={
            "x-apikey": os.environ["FLIGHTAWARE_API_KEY"],
            "Accept": "application/json",
            "User-Agent": "aviation-agent-test/1.0",
        },
        method="GET",
    )

    print(f"Calling FlightAware {url} ...")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        print("\n--- FlightAware response (trimmed) ---")
        print(json.dumps(payload, indent=2, default=str)[:2500])
        if len(json.dumps(payload, default=str)) > 2500:
            print("\n... (truncated)")
        print("--- OK ---")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {exc.code}\n{body[:800]}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")


def main() -> None:
    print(
        """
API smoke test menu
  1) ChatGPT / OpenAI
  2) Gemini Flash
  3) FlightAware AeroAPI
  4) Run all three
  0) Exit
"""
    )
    choice = input("Select option: ").strip()

    # Switch-case style dispatch
    match choice:
        case "1":
            test_openai()
        case "2":
            test_gemini()
        case "3":
            test_flightaware()
        case "4":
            print("\n=== 1) OpenAI ===")
            test_openai()
            print("\n=== 2) Gemini ===")
            test_gemini()
            print("\n=== 3) FlightAware ===")
            test_flightaware()
        case "0":
            print("Bye.")
            sys.exit(0)
        case _:
            print(f"Unknown option: {choice}")


if __name__ == "__main__":
    main()
