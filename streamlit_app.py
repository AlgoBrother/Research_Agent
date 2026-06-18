"""
Streamlit chat interface for the research agent.

Run with:
    streamlit run streamlit_app.py

Layout:
  - Sidebar: session controls (reset), recent queries
  - Main: chat history with expandable paper cards under each answer
  - Bottom: chat input box
"""

import streamlit as st
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from main_chat import ResearchAgent


st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide",
)

# ── Session state setup ──────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent(top_k=6)

if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str, "papers": list}
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Agent")
    st.caption("Ask about ML papers, architectures, and techniques.")

    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.agent.reset_history()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Recent questions")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        for m in reversed(user_msgs[-8:]):
            st.caption(f"• {m['content'][:60]}")
    else:
        st.caption("No questions yet.")

    st.divider()
    st.caption("Modes detected automatically:")
    st.caption("📌 **known_paper** — exact paper by arXiv ID")
    st.caption("🔍 **search** — topic exploration")
    st.caption("💾 **recall** — prior session lookup")


# ── Main chat area ───────────────────────────────────────────────────
st.title("Chat with your Research Agent")

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        papers = msg.get("papers", [])
        if papers:
            with st.expander(f"📄 {len(papers)} source paper(s)"):
                for p in papers:
                    authors = ", ".join(p.authors[:2])
                    if len(p.authors) > 2:
                        authors += " et al."
                    st.markdown(
                        f"**[{p.title}]({p.pdf_url})**  \n"
                        f"{authors} · {p.published.strftime('%Y-%m-%d')}  \n"
                        f"_{p.summary[:200]}..._"
                    )
                    st.divider()

# Chat input
query = st.chat_input("Ask about a paper, architecture, or technique...")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query, "papers": []})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent with live status updates
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()

        def on_step(msg: str):
            status_placeholder.caption(f"⏳ {msg}")

        result = st.session_state.agent.ask(query, on_step=on_step)
        status_placeholder.empty()

        answer = result["answer"]
        papers = result["papers"]

        answer_placeholder.markdown(answer)

        if papers:
            with st.expander(f"📄 {len(papers)} source paper(s)"):
                for p in papers:
                    authors = ", ".join(p.authors[:2])
                    if len(p.authors) > 2:
                        authors += " et al."
                    st.markdown(
                        f"**[{p.title}]({p.pdf_url})**  \n"
                        f"{authors} · {p.published.strftime('%Y-%m-%d')}  \n"
                        f"_{p.summary[:200]}..._"
                    )
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "papers": papers,
    })