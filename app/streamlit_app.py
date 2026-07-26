import time
import uuid

import streamlit as st

from lvlup.monitoring.db import log_conversation_start, log_feedback, log_message, log_retrieved_chunks
from lvlup.rag import answer

st.set_page_config(page_title="Lvl Up Coach", page_icon="\U0001F9E0")
st.title("\U0001F9E0 Lvl Up Coach")
st.caption("Coach de bienestar digital y habitos, basado en evidencia cientifica (OpenAlex).")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = log_conversation_start(str(uuid.uuid4()))
if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("message_id"):
            col_up, col_down = st.columns(2)
            if col_up.button("\U0001F44D", key=f"up-{entry['message_id']}"):
                log_feedback(entry["message_id"], 1)
                st.toast("Gracias por el feedback!")
            if col_down.button("\U0001F44E", key=f"down-{entry['message_id']}"):
                log_feedback(entry["message_id"], -1)
                st.toast("Gracias, lo tendremos en cuenta.")

question = st.chat_input("Pregunta sobre foco, habitos, HRV, zona 2...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    log_message(st.session_state.conversation_id, "user", question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando evidencia..."):
            start = time.time()
            result = answer(question)
            latency_ms = int((time.time() - start) * 1000)
        st.markdown(result["answer"])
        with st.expander("Fuentes"):
            for chunk in result["chunks"]:
                citation = chunk.get("doi") or chunk.get("landing_page_url") or ""
                st.markdown(f"- **{chunk['title']}** ({chunk.get('publication_year', 'n/d')}) - {citation}")

    message_id = log_message(
        st.session_state.conversation_id,
        "assistant",
        result["answer"],
        latency_ms=latency_ms,
    )
    log_retrieved_chunks(message_id, result["chunks"])
    st.session_state.history.append({"role": "assistant", "content": result["answer"], "message_id": message_id})
