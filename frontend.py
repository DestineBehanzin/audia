import streamlit as st
from backend import ask_pablo, read_file
import time

st.set_page_config(page_title="Pablo – Le Parrain du Chatbot", page_icon="🕶️")

# -------------------------
# INITIALISATION DE LA MEMOIRE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": read_file("./context.txt")}
    ]

st.title("🕶️ Pablo – Le Parrain du Chatbot")
st.write("Parle au Parrain… mais n’oublie pas qu’il ne se remet **jamais** en question.")

# -------------------------
# AFFICHAGE DE L’HISTORIQUE
# -------------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":  # ne pas afficher le system prompt
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# -------------------------
# ZONE DE SAISIE UTILISATEUR
# -------------------------
user_input = st.chat_input("Que veux-tu demander à Pablo ?")

if user_input:

    # Ajout du message utilisateur à la mémoire
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Affichage instantané côté utilisateur
    with st.chat_message("user"):
        st.write(user_input)

    # -------------------------
    # APPEL AU BACKEND GROQ
    # -------------------------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # Envoi au backend
        stream = ask_pablo(chat_history=st.session_state.messages)

        # Lecture du stream token par token
        for chunk in stream:
            if chunk.choices[0].delta.content is None:
                continue

            token = chunk.choices[0].delta.content
            full_response += token
            placeholder.write(full_response)

            time.sleep(0.01)

        # Ajout à la mémoire
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
