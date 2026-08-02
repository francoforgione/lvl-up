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
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0

with st.sidebar:
    st.subheader("Uso de la sesion")
    st.metric("Costo acumulado", f"${st.session_state.session_cost:.4f}")
    st.metric("Tokens acumulados", f"{st.session_state.session_tokens:,}")
    if st.button("Nueva conversacion"):
        st.session_state.conversation_id = log_conversation_start(str(uuid.uuid4()))
        st.session_state.history = []
        st.rerun()


def api_history() -> list[dict]:
    """Strip UI-only keys: the Messages API rejects anything but role/content."""
    return [{"role": e["role"], "content": e["content"]} for e in st.session_state.history]


for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("caption"):
            st.caption(entry["caption"])
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
    history = api_history()
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})
    log_message(st.session_state.conversation_id, "user", question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando evidencia..."):
            start = time.time()
            result = answer(question, history=history)
            latency_ms = int((time.time() - start) * 1000)

        st.markdown(result["answer"])

        usage = result["usage"]
        caption = (
            f"{result['iterations']} paso(s) del agente - "
            f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens - "
            f"${result['cost_usd']:.5f} - {latency_ms} ms"
        )
        if result["hit_iteration_limit"]:
            caption += " - limite de iteraciones alcanzado"
        st.caption(caption)

        if result["chunks"]:
            with st.expander(f"Fuentes ({len(result['chunks'])})"):
                for chunk in result["chunks"]:
                    citation = chunk.get("doi") or chunk.get("landing_page_url") or ""
                    st.markdown(
                        f"- **{chunk['title']}** ({chunk.get('publication_year', 'n/d')}) - {citation}"
                    )
        elif result["guardrail_triggered"]:
            st.info(
                "Fuera de alcance: no hay papers en el corpus que respalden esta pregunta, "
                "asi que el coach no responde de memoria."
            )
        else:
            st.caption("El agente respondio sin buscar (contexto previo suficiente).")

    message_id = log_message(
        st.session_state.conversation_id,
        "assistant",
        result["answer"],
        model=result["model"],
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=result["cost_usd"],
        tool_iterations=result["iterations"],
        guardrail_triggered=result["guardrail_triggered"],
    )
    log_retrieved_chunks(message_id, result["chunks"])

    st.session_state.session_cost += result["cost_usd"]
    st.session_state.session_tokens += usage.total_tokens
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "message_id": message_id,
            "caption": caption,
        }
    )
    st.rerun()
