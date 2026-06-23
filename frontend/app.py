"""Streamlit UI for the AI Research Assistant.

Talks to the FastAPI backend: a research run streams live progress over the
WebSocket (consumed synchronously via websocket-client), and history/replay use the
REST endpoints. Set BACKEND_URL to point at a non-default backend host.
"""

import json
import os

import requests
import streamlit as st
import websocket  # websocket-client (sync)

def _backend_url() -> str:
    """Resolve the backend URL from (in priority order) env var, Streamlit secrets,
    then localhost. Secrets let a Streamlit Cloud deployment point at a tunnelled
    local backend without code changes."""
    if os.getenv("BACKEND_URL"):
        return os.environ["BACKEND_URL"]
    try:
        if "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
    except Exception:  # noqa: BLE001 - no secrets file configured
        pass
    return "http://localhost:8000"


BACKEND_URL = _backend_url()
WS_URL = BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws/research"

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")


# --------------------------------------------------------------------------- helpers
def fetch_history() -> list[dict]:
    try:
        return requests.get(f"{BACKEND_URL}/api/history", timeout=5).json()
    except requests.RequestException:
        return []


def fetch_run(session_id: str) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/api/research/{session_id}", timeout=5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def run_research(query: str, max_iterations: int, progress_box) -> dict | None:
    """Open the WebSocket, render progress as it streams, return the final result."""
    try:
        ws = websocket.create_connection(WS_URL, max_size=16_000_000, timeout=600)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not connect to backend at {WS_URL}: {e}")
        return None

    ws.send(json.dumps({"query": query, "max_iterations": max_iterations}))
    steps: list[str] = []
    result = None
    try:
        while True:
            msg = json.loads(ws.recv())
            kind = msg.get("type")
            if kind == "started":
                steps.append(f"🆔 Session `{msg['session_id']}` started")
            elif kind == "progress":
                steps.append(f"**{msg['label']}** — {msg['detail']}")
            elif kind == "complete":
                result = msg["result"]
                steps.append("✅ Done")
            elif kind == "error":
                st.error(msg["message"])
                break
            progress_box.markdown("\n\n".join(f"- {s}" for s in steps))
            if kind in ("complete", "error"):
                break
    finally:
        ws.close()
    return result


def render_result(result: dict) -> None:
    scores = result.get("ragas_scores") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Faithfulness", _fmt(scores.get("faithfulness")),
        help="How well the answer is grounded in the retrieved sources "
             "(0–1; higher = fewer unsupported claims).",
    )
    c2.metric(
        "Answer Relevancy", _fmt(scores.get("answer_relevancy")),
        help="How directly the answer addresses the question "
             "(0–1; higher = more on-topic).",
    )
    c3.metric(
        "Context Precision", _fmt(scores.get("context_precision")),
        help="How relevant the retrieved context is to the question "
             "(0–1; higher = less off-topic retrieval).",
    )

    st.markdown("### 📋 Answer")
    st.markdown(result.get("final_answer") or "_No answer generated._")

    citations = result.get("citations", [])
    if citations:
        with st.expander(f"📚 Sources & Citations ({len(citations)})", expanded=False):
            for c in citations:
                title = c.get("title") or c.get("url") or "source"
                url = c.get("url", "")
                line = f"**[{c.get('number')}]** {title}"
                st.markdown(f"{line}  \n{url}" if url else line)

    focus = result.get("focus_areas", [])
    if focus:
        with st.expander(f"🧭 Research Plan ({len(focus)} focus areas)"):
            for f in sorted(focus, key=lambda x: x.get("priority", 0), reverse=True):
                st.markdown(
                    f"**{f.get('title')}** (priority {f.get('priority')})  \n"
                    f"{f.get('description', '')}"
                )

    errors = result.get("errors", [])
    if errors:
        with st.expander(f"⚠️ Non-fatal notes ({len(errors)})"):
            for e in errors:
                st.text(f"• {e}")


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------- layout
if "result" not in st.session_state:
    st.session_state.result = None

st.title("🔬 AI Research Assistant")
st.caption(
    "A multi-agent pipeline (Plan → Research → Fact-check → Synthesize → Evaluate) "
    "built on LangGraph, with RAGAS quality scoring."
)

with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("Max research iterations", 1, 6, 3)
    backend_ok = False
    try:
        backend_ok = requests.get(f"{BACKEND_URL}/api/health", timeout=3).ok
    except requests.RequestException:
        backend_ok = False
    st.markdown(f"Backend: {'🟢 online' if backend_ok else '🔴 offline'}  \n`{BACKEND_URL}`")

    st.divider()
    st.header("🕘 History")
    for run in fetch_history():
        label = run["query"][:40] + ("…" if len(run["query"]) > 40 else "")
        if st.button(label, key=run["session_id"], use_container_width=True):
            loaded = fetch_run(run["session_id"])
            if loaded:
                st.session_state.result = loaded
                st.rerun()

query = st.text_area(
    "Research question",
    placeholder="e.g. What are the main causes of climate change?",
    height=90,
)
go = st.button("Research", type="primary", disabled=not backend_ok)

if go and query.strip():
    st.markdown("#### ⏳ Progress")
    progress_box = st.empty()
    with st.spinner("Researching… this runs 5 agents and can take a couple of minutes."):
        result = run_research(query.strip(), max_iterations, progress_box)
    if result:
        st.session_state.result = result

if st.session_state.result:
    st.divider()
    render_result(st.session_state.result)
