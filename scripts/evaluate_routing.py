"""LangSmith Phase 1 routing eval: supervisor intent, spokes, and Memory Gate.

Where data lives
----------------
- Golden cases: local source in scripts/create_langsmith_dataset.py
- Dataset rows: LangSmith cloud (create_langsmith_dataset.py uploads via API)
- This script: reads that cloud dataset by name, writes experiment scores back

End-to-end flow
---------------
  create_langsmith_dataset.py          evaluate_routing.py (this file)
         |                                      |
         v                                      v
  LangSmith Dataset                   langsmith.evaluate()
  (inputs + outputs per row)               |
         |                                 |
         +-------- data=DATASET_NAME -------+
                                           |
         For each Example:                 |
           inputs: {prompt, history}  -->  predict_routing()
           outputs: {expected_*}      <--  compare via 3 evaluators
                                           |
                                           v
                         Experiment "Routing_Supervisor_Test-..." in LangSmith UI

LangSmith vocabulary (for review)
---------------------------------
- Example = one labeled row from the dataset (inputs = test case, outputs = answer key)
- Run     = one execution of predict_routing for that Example (run.outputs = model got)
- Evaluator = function(run, example) returning {key, score, comment} feedback on that Run

What is NOT run: OpenSky, FlightAware, Tavily, synthesizer, clarification interrupt.

Alignment with the real app
---------------------------
- Turn resets: _build_eval_state copies the same empty fields app.py sends on a new
  prompt (turn_notes=[], plan=[], extract_memory=False, etc.).
- Messages: In Streamlit, graph_input only adds the NEW HumanMessage(prompt); older
  user/assistant lines already live in checkpoints.db and are merged by the
  add_messages_window reducer. This eval script has no checkpoint thread, so it
  must reconstruct the full transcript: dataset history + latest prompt as messages.
  supervisor_node then reads that list the same way (history_blob + latest user text).
- history in the dataset is hand-authored fiction (see create_langsmith_dataset.py),
  not replay from checkpoints.db. Assistant lines in history are scripted context.
- We call supervisor_node directly, not app_graph.stream. That is intentional: Phase 1
  scores routing labels only, not post-supervisor graph edges (Send fan-out, HITL).

What can still happen inside supervisor_node during eval
--------------------------------------------------------
- Live OpenAI structured SupervisorPlan call (or _offline_supervisor if the model fails)
- Mem0 search at turn start (scoped to EVAL_USER_ID)
- Post-model rules: intent forces Tavily-only on discovery/general, caps, dedup

Those side effects are acceptable for routing eval; they are not mocked here.

Prerequisites (.env):
  - LANGCHAIN_API_KEY — LangSmith upload + dataset fetch
  - OPENAI_API_KEY — supervisor structured plan (gpt-4o-mini default)

Run from aviation_agent project root:

    python scripts/evaluate_routing.py

Mem0 is skipped in eval (EVAL_USER_ID is empty). Production uses the real Google user_id.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import evaluate
from langsmith.schemas import Example, Run

# App imports expect the project root on sys.path (same layout as Streamlit app.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from src.graph.nodes import supervisor_node  # noqa: E402

# Must match the name created by scripts/create_langsmith_dataset.py (cloud copy).
DATASET_NAME = "FinalApproach_Routing_Eval_v2"
# LangSmith names experiments "{EXPERIMENT_PREFIX}-{timestamp}" (or similar).
EXPERIMENT_PREFIX = "Routing_Supervisor_Test"
# Empty user_id skips Mem0 search in supervisor_node (_search_user_memories returns []).
# Routing eval should not spin up local Qdrant or optional Mem0 NLP deps.
EVAL_USER_ID = ""


def _history_to_messages(history: list[dict[str, Any]] | None) -> list:
    """Turn dataset history rows into LangChain messages, oldest → newest.

    Dataset schema (from create_langsmith_dataset.py):
      {"role": "user"|"assistant", "content": "..."}

    Matches roles supervisor_node already understands (_latest_user_text also
    accepts dict-shaped messages). Unknown roles are skipped silently.
    """
    messages = []
    for item in list(history or []):
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "")
        if role in {"user", "human"}:
            messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
        # System/tool roles are not used in Phase 1 dataset rows.
    return messages


def _build_eval_state(prompt: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """One-shot AgentState dict passed into supervisor_node.

    Field defaults mirror app.py graph_input on a normal (non-resume) user prompt.
    See app.py ~lines 551-565 for the production twin.

    messages: history (prior turns) + HumanMessage(prompt) as the latest user line.
              _latest_user_text reads the last human message → prompt.
              history_blob is built from the full messages list inside supervisor_node.

    Omitted on purpose: no checkpointer config, no thread_id — single isolated hop.
    """
    return {
        "messages": _history_to_messages(history) + [HumanMessage(content=prompt)],
        "user_id": EVAL_USER_ID,
        "turn_notes": [],
        "plan": [],
        "spokes_to_call": [],
        "context_priority_order": [],
        "intent": "",
        "flight_number": None,
        "search_query": None,
        "next_worker": "",
        "current_task": "",
        "extract_memory": False,
        "final_report": "",
    }


def _invoke_supervisor(state: dict[str, Any]) -> dict[str, Any]:
    """Call supervisor_node; tolerate a future async implementation.

    Today supervisor_node is sync. LangSmith evaluate calls predict_routing from a
    worker thread; asyncio.run is only needed if the node becomes async.
    """
    if inspect.iscoroutinefunction(supervisor_node):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(supervisor_node(state))
        raise RuntimeError(
            "supervisor_node is async and an event loop is already running. "
            "Run this script from __main__ (no nested loop), or switch to an "
            "async predict_routing target."
        )
    return supervisor_node(state)


def predict_routing(inputs: dict) -> dict:
    """LangSmith target: dataset inputs → supervisor routing labels.

    LangSmith passes example.inputs from the cloud dataset:
      prompt  — current user utterance for this turn
      history — optional list of prior {role, content} dicts (defaults to [])

    We return run.outputs (what evaluators read). Only three fields are scored;
    plan rows, flight_number, and context_priority_order are ignored in Phase 1.

    Mapping from supervisor_node return patch:
      patch["intent"], patch["spokes_to_call"], patch["extract_memory"]
    """
    prompt = str(inputs.get("prompt") or "").strip()
    history = list(inputs.get("history") or [])
    state = _build_eval_state(prompt, history)
    # supervisor_node returns a partial patch (LangGraph node style), not full AgentState.
    patch = _invoke_supervisor(state)
    return {
        "intent": patch.get("intent"),
        "spokes_to_call": list(patch.get("spokes_to_call") or []),
        "extract_memory": bool(patch.get("extract_memory", False)),
    }


def evaluate_intent(run: Run, example: Example) -> dict:
    """Binary score: run.outputs.intent vs example.outputs.expected_intent.

    example.outputs keys were uploaded by create_langsmith_dataset.py.
    """
    expected = (example.outputs or {}).get("expected_intent")
    actual = (run.outputs or {}).get("intent")
    ok = actual == expected
    return {
        "key": "intent",
        "score": 1 if ok else 0,
        "comment": None if ok else f"expected={expected!r}, got={actual!r}",
    }


def evaluate_memory(run: Run, example: Example) -> dict:
    """Binary score: Memory Gate (extract_memory) on vs off.

    Golden field in dataset: outputs.extract_memory (bool).
    """
    expected = bool((example.outputs or {}).get("extract_memory"))
    actual = bool((run.outputs or {}).get("extract_memory"))
    ok = actual == expected
    return {
        "key": "memory",
        "score": 1 if ok else 0,
        "comment": None if ok else f"expected extract_memory={expected}, got={actual}",
    }


def evaluate_spokes(run: Run, example: Example) -> dict:
    """Subset check on spokes_to_call (not plan row count or per-task caps).

    Golden fields: must_include_spokes, must_exclude_spokes (lists of spoke ids).

    must_include: every listed spoke must appear in spokes_to_call (subset test).
    must_exclude: no listed spoke may appear.
    Extra spokes not in either list do not fail this scorer (e.g. tavily allowed
    when must_include is only flightaware unless tavily is in must_exclude).

    Phase 1 does not assert two FlightAware rows for BA117+BA118 — only spoke names.
    """
    actual = list((run.outputs or {}).get("spokes_to_call") or [])
    must_include = list((example.outputs or {}).get("must_include_spokes") or [])
    must_exclude = list((example.outputs or {}).get("must_exclude_spokes") or [])

    missing_spokes = [spoke for spoke in must_include if spoke not in actual]
    forbidden_spokes = [spoke for spoke in must_exclude if spoke in actual]
    ok = not missing_spokes and not forbidden_spokes

    comment_parts: list[str] = []
    if missing_spokes:
        comment_parts.append(f"missing={missing_spokes}")
    if forbidden_spokes:
        comment_parts.append(f"forbidden={forbidden_spokes}")
    if not ok:
        comment_parts.append(f"got spokes_to_call={actual}")

    return {
        "key": "spokes",
        "score": 1 if ok else 0,
        "comment": "; ".join(comment_parts) if comment_parts else None,
    }


def main() -> int:
    """Drive langsmith.evaluate and print the experiment handle.

    Each dataset row gets three feedback scores (intent, memory, spokes) attached
    to the same Run in LangSmith. Aggregate pass rate is visible in the experiment UI.
    Filter examples by metadata.split == "dev" vs "test" in the UI when comparing.
    """
    print(
        f"Starting LangSmith evaluate on dataset {DATASET_NAME!r} "
        f"(experiment_prefix={EXPERIMENT_PREFIX!r})..."
    )
    # max_concurrency=2: 18 OpenAI supervisor calls; lower if you hit rate limits.
    results = evaluate(
        predict_routing, # our main fountion which evaluates the supervisor routing labels.
        data=DATASET_NAME, # the name of the dataset we want to evaluate.
        evaluators=[evaluate_intent, evaluate_memory, evaluate_spokes], # the evaluators we want to use, all evaluators must return a dict with key, score, and comment.
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=2,
    )
    print("Evaluation finished. Open LangSmith → Datasets → Experiments for details.")
    print(results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"Routing evaluation failed. Check LANGCHAIN_API_KEY, OPENAI_API_KEY, "
            f"and that dataset {DATASET_NAME!r} exists (run create_langsmith_dataset.py): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
