# app.py
import streamlit as st
import json
from backend import preprocess_data, generate_graphs, save_pdf, ask_audia

st.set_page_config(page_title="AudIA - Rapport Acoustique", layout="centered")

# --------------------------- Styles ---------------------------
st.markdown("""
<style>
h1,h2,h3{text-align:center;}
.stButton>button{
    background-color:#1F77B4;color:white;height:40px;width:250px;
    border-radius:8px;font-size:16px;font-weight:bold;
}
.stButton>button:hover{background-color:#145A86;}
</style>
""", unsafe_allow_html=True)

st.title("📊 AudIA - Rapport Acoustique Résidentiel")
st.markdown("Téléversez votre JSON pour générer automatiquement un rapport PDF complet (texte + visualisations).")

# --------------------------- Fonction LLM ---------------------------
def generate_llm_report(summary, prompt_file="comportement.txt"):
    """Génère le texte du rapport à partir du résumé et du prompt"""
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Résumé des données acoustiques :\n{json.dumps(summary, indent=2)}\nGénère un rapport complet, professionnel et lisible."}
    ]
    final_report = ""
    for chunk in ask_audia(chat_history):
        try:
            part = chunk.choices[0].delta.content
            if part:
                final_report += part
        except Exception:
            continue
    return final_report.strip()

# --------------------------- Upload JSON ---------------------------
uploaded_file = st.file_uploader("📁 Choisissez un fichier JSON", type="json")
if uploaded_file:
    try:
        mesures = json.load(uploaded_file)
        st.success("✅ Fichier JSON chargé !")
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        mesures = None

    if mesures:
        # Préprocessing
        processed_summary = preprocess_data(mesures)

        # Génération texte rapport
        with st.spinner("⏳ Génération du texte du rapport..."):
            report_text = generate_llm_report(processed_summary)
        st.success("✅ Texte du rapport généré !")
        st.text_area("Aperçu du rapport", report_text, height=300)

        # Génération graphiques
        image_paths = generate_graphs(processed_summary)
        st.subheader("📊 Visualisations")
        cols = st.columns(2)
        for i, img in enumerate(image_paths):
            cols[i % 2].image(img, use_column_width=True)

        # Génération PDF complet
        st.subheader("📄 Télécharger le rapport PDF complet")
        if st.button("Générer et télécharger le PDF"):
            pdf_path = "rapport_complet.pdf"
            save_pdf(report_text, output_path=pdf_path, image_paths=image_paths)
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Télécharger le PDF", f, file_name="rapport_complet.pdf", mime="application/pdf")
