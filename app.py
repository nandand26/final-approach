"""Streamlit front door for the Aviation Multi-Agent System.

Easy overview of this file:
  1) Make sure DBs exist, show the page title.
  2) Force Google login (no chat until signed in).
  3) Sidebar: this user's chat threads (from auth_sessions.db).
  4) Main pane: load that chat's messages from LangGraph memory.
  5) On a new prompt: run the agent for THIS thread_id, show metro progress,
     then reload the saved final answer.

Security idea in one line:
  user owns thread_ids in SQLite → every graph call stamps that thread_id
  → LangGraph memory stays private per chat.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

# ---------------------------------------------------------------------------
# Bootstrapping: path + env BEFORE importing our packages
# ---------------------------------------------------------------------------
# Streamlit runs this file as a script, so `src` must be on sys.path.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load API keys from .env (OpenAI, Gemini, FlightAware, Tavily, LangSmith, …)
load_dotenv(ROOT / ".env", override=True)

# Mirror LangSmith env aliases for newer SDKs (not UI).
# Regional SaaS matters: APAC/EU keys 403 on the default US endpoint.
if (os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower() in {"1", "true", "yes"}:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGSMITH_TRACING", "true")
if os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
if os.getenv("LANGCHAIN_PROJECT") and not os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]
_ls_endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
if _ls_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = _ls_endpoint
    os.environ["LANGSMITH_ENDPOINT"] = _ls_endpoint

# Imports after path/env so models pick up keys at import time.
from src.db.auth_db import (  # noqa: E402
    create_new_thread,
    get_user_threads,
    init_auth_db,
    register_or_get_user,
)
from src.graph.nodes import synthesizer_node  # noqa: E402
from src.graph.workflow import app_graph  # noqa: E402
from src.ui.metro_progress import (  # noqa: E402
    NODE_DISPLAY_MAP,
    TurnProgress,
    advance_on_node_update,
    mark_remaining_skipped,
    progress_snapshot,
    render_agent_steps_panel,
    render_progress,
)


def _thread_config(
    thread_id: str,
    user_id: str | None = None,
    *,
    entrypoint: str = "streamlit",
) -> dict[str, Any]:
    """Config passed into every LangGraph call for this chat.

    - thread_id  → which memory bucket to use (this chat only)
    - recursion_limit → how many supervisor↔spoke hops we allow before abort
    - run_name / tags / metadata → LangSmith run tagging for this turn

    Tags are built dynamically from entrypoint + optional user_id so a future
    API/notebook caller can pass entrypoint="api" instead of hardcoding UI labels.
    """
    tags = [f"entrypoint:{entrypoint}"]
    metadata: dict[str, Any] = {
        "thread_id": thread_id,
        "entrypoint": entrypoint,
    }
    if user_id:
        tags.append(f"user:{user_id}")
        metadata["user_id"] = user_id
    return {
        # Up to 9 spoke hops (2+2+5) plus supervisor visits → keep headroom.
        "recursion_limit": 22,
        "configurable": {"thread_id": thread_id},
        "run_name": "Aviation_Ops_Turn",
        "tags": tags,
        "metadata": metadata,
    }


def _message_to_chat_row(message: Any) -> tuple[str, str] | None:
    """Turn a LangChain message into a Streamlit chat bubble.

    Returns ("user", text) or ("assistant", text), or None to skip.
    """
    if isinstance(message, HumanMessage):
        return ("user", str(message.content))
    if isinstance(message, AIMessage):
        return ("assistant", str(message.content))

    # Some stored messages may be plain dicts — handle those too
    msg_type = getattr(message, "type", None)
    if msg_type is None and isinstance(message, dict):
        msg_type = message.get("role")
    content = (
        str(getattr(message, "content", ""))
        if not isinstance(message, dict)
        else str(message.get("content", ""))
    )
    if msg_type in {"human", "user"}:
        return ("user", content)
    if msg_type in {"ai", "assistant"}:
        return ("assistant", content)
    return None


def _load_thread_messages(thread_id: str) -> list[Any]:
    """Read saved chat bubbles for one thread from LangGraph's checkpointer.

    We do NOT store chat text in auth_sessions.db.
    auth DB = who owns which thread_id
    checkpointer = the actual messages for that thread_id
    """
    try:
        snapshot = app_graph.get_state(config={"configurable": {"thread_id": thread_id}})
        values = getattr(snapshot, "values", {}) or {}
        messages = values.get("messages") or []
        return list(messages)
    except Exception:
        # Brand-new thread: no checkpoint yet → empty chat
        return []


def _extract_authenticated_user() -> dict[str, str]:
    """Pull a stable user id out of Streamlit's Google login object.

    Prefer Google `sub` (never changes). Fall back to email if needed.
    """
    user_obj = st.user

    user_sub = getattr(user_obj, "sub", None)
    user_email = getattr(user_obj, "email", None)
    user_name = getattr(user_obj, "name", None)

    # Some Streamlit builds expose dict-like .get()
    if user_sub is None and hasattr(user_obj, "get"):
        user_sub = user_obj.get("sub")
    if user_email is None and hasattr(user_obj, "get"):
        user_email = user_obj.get("email")
    if user_name is None and hasattr(user_obj, "get"):
        user_name = user_obj.get("name")

    clean_email = str(user_email or "").strip()
    clean_name = str(user_name or "").strip()
    clean_sub = str(user_sub or "").strip()
    user_id = clean_sub or clean_email
    if not user_id:
        raise ValueError("Authenticated user missing both `sub` and `email`.")

    return {
        "user_id": user_id,
        "email": clean_email,
        "display_name": clean_name or clean_email or user_id,
    }


def _store_agent_steps(thread_id: str, progress: TurnProgress) -> None:
    """Remember this turn's metro board so history can show it collapsed after rerun."""
    log = st.session_state.setdefault("agent_steps_log", {})
    per_thread: list = log.setdefault(thread_id, [])
    per_thread.append(progress_snapshot(progress))


def _agent_steps_for_assistant_index(
    thread_id: str,
    assistant_index: int,
    total_assistants: int,
) -> dict | None:
    """Map the Nth assistant bubble to a saved progress snapshot.

    Snapshots are aligned to the *newest* assistant turns, so older answers
    from before this feature simply have no Agent steps expander.
    """
    per_thread = (st.session_state.get("agent_steps_log") or {}).get(thread_id) or []
    if not per_thread:
        return None
    offset = max(0, total_assistants - len(per_thread))
    snap_i = assistant_index - offset
    if 0 <= snap_i < len(per_thread):
        snap = per_thread[snap_i]
        return snap if isinstance(snap, dict) else None
    return None


# ---------------------------------------------------------------------------
# One-time setup (runs on every Streamlit script rerun, but DB init is cheap)
# ---------------------------------------------------------------------------
init_auth_db()    # users + user_threads ownership SQLite (auth only — not flight history)

st.set_page_config(
    page_title="Aviation Fleet & Operational Intelligence Agent",
    page_icon="✈️",
    layout="centered",
)
st.title("Aviation Fleet & Operational Intelligence Agent")
st.caption("Google-authenticated, thread-isolated LangGraph aviation assistant.")

# ---------------------------------------------------------------------------
# Login gate — stop here if not signed in
# ---------------------------------------------------------------------------
if not st.user.is_logged_in:
    st.markdown(
        """
### Welcome
Sign in with Google to access your private aviation intelligence workspace.

Each account gets isolated chat threads and memory checkpoints.
"""
    )
    st.button("Sign in with Google", on_click=st.login, type="primary")
    st.stop()

# Save / update this Google user in our local auth DB
current_user = register_or_get_user(_extract_authenticated_user())
current_user_id = current_user["user_id"]

# If a different Google account logs in on the same browser, drop old thread pointer
if st.session_state.get("auth_user_id") != current_user_id:
    st.session_state.auth_user_id = current_user_id
    st.session_state.current_thread_id = None

# ---------------------------------------------------------------------------
# Sidebar = profile + list of THIS user's chats only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### User")
    st.markdown(f"**{current_user['display_name']}**")
    if current_user["email"]:
        st.caption(current_user["email"])

    st.button("Log out", on_click=st.logout)

    st.markdown("---")
    st.markdown("### Chats")

    # New Chat → new UUID thread_id row owned by this user
    # (titles can all say "New Chat"; thread_id is what keeps them apart)
    if st.button("+ New Chat", type="primary"):
        created = create_new_thread(user_id=current_user_id, title="New Chat")
        st.session_state.current_thread_id = created["thread_id"]
        st.rerun()

    threads = get_user_threads(current_user_id)
    if not threads:
        # First visit: auto-make one empty chat so the page is usable
        created = create_new_thread(user_id=current_user_id, title="New Chat")
        st.session_state.current_thread_id = created["thread_id"]
        st.rerun()

    thread_ids = [t["thread_id"] for t in threads]
    thread_labels = {
        t["thread_id"]: f"{t['title']} · {t['created_at'][:19].replace('T', ' ')}"
        for t in threads
    }

    # If session points at a thread we don't own / that vanished, pick the newest
    current_thread_id = st.session_state.get("current_thread_id")
    if current_thread_id not in thread_ids:
        current_thread_id = thread_ids[0]
        st.session_state.current_thread_id = current_thread_id

    selected_thread = st.selectbox(
        "Select a thread",
        options=thread_ids,
        index=thread_ids.index(current_thread_id),
        format_func=lambda tid: thread_labels.get(tid, tid),
    )

    # Switching chats = switching LangGraph memory namespace
    if selected_thread != current_thread_id:
        st.session_state.current_thread_id = selected_thread
        st.rerun()

active_thread_id = str(st.session_state.current_thread_id)

# ---------------------------------------------------------------------------
# Main chat history (from checkpointer for active_thread_id)
# ---------------------------------------------------------------------------
st.caption(f"Active thread: `{active_thread_id}`")
history_messages = _load_thread_messages(active_thread_id)
total_assistants = sum(
    1
    for m in history_messages
    if (row := _message_to_chat_row(m)) is not None and row[0] == "assistant"
)
assistant_turn_index = 0
for stored_message in history_messages:
    row = _message_to_chat_row(stored_message)
    if row is None:
        continue
    role, content = row
    with st.chat_message(role):
        st.markdown(content)
        # After each saved answer, offer the metro board collapsed (demo-friendly)
        if role == "assistant":
            snap = _agent_steps_for_assistant_index(
                active_thread_id,
                assistant_turn_index,
                total_assistants,
            )
            assistant_turn_index += 1
            if snap:
                with st.expander("Agent steps", expanded=False):
                    render_agent_steps_panel(
                        strip=str(snap.get("strip") or ""),
                        strip_html=str(snap.get("strip_html") or "") or None,
                        status_line=str(snap.get("status_line") or ""),
                        details=list(snap.get("details") or []),
                        expanded=False,
                    )

# ---------------------------------------------------------------------------
# New user prompt → run the agent for this thread
# ---------------------------------------------------------------------------
prompt = st.chat_input("Enter flight query or tail number...")
if prompt:
    # Show the user bubble immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send ONLY the new message. Old history is already in the checkpointer
    # for this thread_id. Also clear turn scratch so last question's plan
    # and tool notes cannot leak into this question.
    graph_input = {
        "messages": [HumanMessage(content=prompt)],
        # Stamp Google OAuth identity so Mem0 can scope cross-session memory.
        "user_id": current_user_id,
        "turn_notes": [],  # empty list = "clear scratch" (see merge_turn_notes)
        "plan": [],        # empty list = wipe last turn's spoke queue
        "next_worker": "",
        "current_task": "",  # hop-local sub-query cleared each turn
        # Memory Gate default OFF; supervisor may flip True for explicit prefs only.
        "extract_memory": False,
        "final_report": "",
    }

    with st.chat_message("assistant"):
        # Live UI slot we redraw as the metro board advances
        placeholder = st.empty()
        # Keep raw notes in case recursion limit forces a manual synthesizer call
        collected_turn_notes: list[str] = []
        progress = TurnProgress()
        # Live track also mirrored in session_state so a mid-turn rerun can recover
        st.session_state["live_metro_progress"] = progress
        thread_cfg = _thread_config(
            active_thread_id,
            user_id=current_user_id,
            entrypoint="streamlit",
        )
        # Mutable counter (list) — `if prompt:` is module scope, so `nonlocal` is illegal.
        logged_steps = [0]

        def _paint() -> None:
            """Redraw metro strip + expanders + answer into the assistant bubble."""
            with placeholder.container():
                render_progress(progress)

        def _flush_status_log(status_box: Any) -> None:
            """Push new metro status_log lines into the open st.status panel."""
            while logged_steps[0] < len(progress.status_log):
                status_box.write(progress.status_log[logged_steps[0]])
                logged_steps[0] += 1

        try:
            with st.status(
                "Orchestrating Flight Intelligence Pipeline...",
                expanded=True,
            ) as status_box:
                # stream_mode="updates" → one event each time a graph node finishes
                for event in app_graph.stream(
                    graph_input,
                    stream_mode="updates",
                    config=thread_cfg,  # stamps thread_id so memory stays on this chat
                ):
                    # event looks like: {"spoke_2_liveops": {"turn_notes": [...]}}
                    for node_name, update in event.items():
                        if not isinstance(update, dict):
                            continue

                        # Save scratch text for fallback; do NOT dump it raw in the UI
                        for note in update.get("turn_notes") or []:
                            collected_turn_notes.append(str(note))

                        # Move metro lights from this node update
                        advance_on_node_update(progress, str(node_name), update)
                        st.session_state["live_metro_progress"] = progress
                        _flush_status_log(status_box)
                        # Keep status label pointing at the active station
                        active = next(
                            (s for s in progress.stations if s.status == "active"),
                            None,
                        )
                        if active:
                            status_box.update(
                                label=(
                                    f"Orchestrating… now at {active.label}"
                                ),
                                state="running",
                            )
                        elif progress.final_report:
                            status_box.update(
                                label=(
                                    f"{NODE_DISPLAY_MAP['synthesizer']} complete"
                                ),
                                state="complete",
                            )
                        _paint()

                if not progress.final_report:
                    progress.final_report = "No response generated."
                    _paint()

                status_box.update(
                    label="Flight intelligence pipeline complete",
                    state="complete",
                )
                _flush_status_log(status_box)

            # Keep metro board available (collapsed) after rerun, next to this answer
            _store_agent_steps(active_thread_id, progress)
            st.session_state.pop("live_metro_progress", None)

            # Reload page from checkpointer so history is the source of truth.
            st.rerun()

        except GraphRecursionError as exc:
            # Agent hopped too many times — don't leave the user with a blank bubble.
            # Grey unfinished stations, then force a synthesizer pass on what we have.
            closer = (
                "[system] Recursion limit hit before the plan fully completed. "
                "Synthesizing a partial aviation report from turn notes collected so far. "
                f"({exc})"
            )
            collected_turn_notes.append(closer)
            mark_remaining_skipped(progress)
            _paint()

            prior_messages = list(_load_thread_messages(active_thread_id))
            # Make sure this user prompt is in the message list for the synthesizer
            if not prior_messages or _message_to_chat_row(prior_messages[-1]) != (
                "user",
                prompt,
            ):
                prior_messages = prior_messages + [HumanMessage(content=prompt)]

            # Honor Memory Gate if supervisor already wrote it into the checkpoint.
            extract_memory_flag = False
            try:
                snap = app_graph.get_state(thread_cfg)
                extract_memory_flag = bool(
                    (snap.values or {}).get("extract_memory")
                )
            except Exception:  # noqa: BLE001
                pass

            partial_state = {
                "messages": prior_messages,
                "user_id": current_user_id,
                "turn_notes": collected_turn_notes,
                "plan": [],
                "next_worker": "FINISH",
                "current_task": "",
                "extract_memory": extract_memory_flag,
                "final_report": "",
            }
            try:
                synth_update = synthesizer_node(partial_state)
                reply = str(synth_update.get("final_report") or "").strip()
                if not reply:
                    reply = (
                        "Partial report unavailable. Recursion limit was hit and "
                        "no synthesizer output was produced."
                    )
                # Write the partial answer into this thread's checkpoint by hand
                app_graph.update_state(
                    thread_cfg,
                    {
                        "messages": [AIMessage(content=reply)],
                        "final_report": reply,
                        "plan": [],
                        "turn_notes": collected_turn_notes,
                        "next_worker": "FINISH",
                    },
                )
                progress.final_report = reply
                progress.is_partial = True
                _paint()
                _store_agent_steps(active_thread_id, progress)
                st.session_state.pop("live_metro_progress", None)
                st.rerun()
            except Exception as synth_exc:  # noqa: BLE001
                placeholder.error(
                    f"Graph hit recursion limit, and partial synthesis also failed: {synth_exc}"
                )

        except Exception as exc:  # noqa: BLE001
            placeholder.error(f"Graph execution failed: {exc}")
