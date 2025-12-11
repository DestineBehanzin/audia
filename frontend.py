import streamlit as st
from backend import ask_audia, read_file
import time

st.set_page_config(page_title="AudIA", page_icon="🕶️")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": read_file("./comportement.txt")}
    ]

st.title("🕶️ AudIA - Assistant Acousticien")

# Afficher l’historique (hors system)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("Pose ta question :")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        stream = ask_audia(st.session_state.messages)

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full += token
                placeholder.write(full)
                time.sleep(0.01)

        st.session_state.messages.append({"role": "assistant", "content": full})