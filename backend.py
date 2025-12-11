import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from config import MODEL

load_dotenv()

# =========================
# UTILITAIRES
# =========================
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# PREPROCESSING AVEC PANDAS
# =========================
def preprocess_data(mesures):
    df = pd.DataFrame(mesures)

    summary = {}

    # ---- Statistiques globales ----
    summary["stats_globales"] = {
        "LAeq_moyen": round(df["LAeq_segment_dB"].mean(), 2),
        "LAeq_min": round(df["LAeq_segment_dB"].min(), 2),
        "LAeq_max": round(df["LAeq_segment_dB"].max(), 2),
        "LAeq_ecart_type": round(df["LAeq_segment_dB"].std(), 2),
        "L10_moyen": round(df["L10_dB"].mean(), 2),
        "L50_moyen": round(df["L50_dB"].mean(), 2),
        "L90_moyen": round(df["L90_dB"].mean(), 2),
    }

    # ---- Agrégation jour / nuit ----
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["periode"] = df["timestamp"].dt.hour.apply(lambda h: "jour" if 7 <= h <= 22 else "nuit")

    summary["jour_nuit"] = (
        df.groupby("periode")["LAeq_segment_dB"]
        .agg(["mean", "min", "max"])
        .round(2)
        .to_dict()
    )

    # ---- Classification des bruits ----
    families = {
        "Circulation": ["vehicle", "car", "engine", "traffic", "skate"],
        "Electromenager": ["appliance", "washing", "fridge"],
        "Voisinage": ["music", "voice", "steps", "cacophony"],
        "Plomberie": ["pipe", "plumbing", "water"],
        "Autres": []
    }

    all_labels = df["top_5_labels"].explode()
    family_counts = {fam: 0 for fam in families}

    for label in all_labels:
        lname = label.lower()
        matched = False
        for fam, keys in families.items():
            if any(k in lname for k in keys):
                family_counts[fam] += 1
                matched = True
                break
        if not matched:
            family_counts["Autres"] += 1

    summary["bruits_par_famille"] = family_counts

    # ---- Typologie: normal vs exceptionnel ----
    freq = all_labels.value_counts()
    seuil = freq.mean()

    summary["typologie"] = {
        "normaux": freq[freq <= seuil].index.tolist(),
        "exceptionnels": freq[freq > seuil].index.tolist()
    }

    # ---- Top 20 bruits ----
    summary["top_20_bruits"] = freq.head(20).to_dict()

    # ---- Note sonore ----
    LAeq = summary["stats_globales"]["LAeq_moyen"]

    if LAeq < 30: note = "A"
    elif LAeq < 35: note = "B"
    elif LAeq < 40: note = "C"
    elif LAeq < 45: note = "D"
    elif LAeq < 50: note = "E"
    elif LAeq < 55: note = "F"
    else: note = "G"

    summary["note_sonore"] = note

    # ---- Faiblesses probables ----
    weaknesses = []
    if family_counts["Circulation"] > seuil:
        weaknesses.append("Faible isolation fenêtres / façade.")
    if family_counts["Voisinage"] > seuil:
        weaknesses.append("Faible isolation mur / sol / plafond.")
    if family_counts["Plomberie"] > seuil:
        weaknesses.append("Problème probable de tuyauterie / cloisons légères.")

    summary["faiblesses_probables"] = weaknesses

    return summary

# =========================
# INTERACTION AVEC AUDIA (LLM)
# =========================
def ask_audia(chat_history):
    client = Groq(api_key=os.environ["GROQ_KEY"])
    return client.chat.completions.create(
        messages=chat_history,
        stream=True,
        model=MODEL
    )

# =========================
# PDF
# =========================
def save_pdf(report_text, output_path="rapport_acoustique.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    # Découpe sur double saut de ligne
    sections = report_text.split("\n\n")
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Remplacement des <br> pour éviter les erreurs
        sec = sec.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

        # Détecte un tableau Markdown
        if sec.startswith("|"):
            rows = [row.strip().split("|")[1:-1] for row in sec.split("\n") if row.strip()]
            table_data = [[Paragraph(cell.strip(), styles["Normal"]) for cell in row] for row in rows]

            table = Table(table_data, hAlign="LEFT")
            table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ]))
            story.append(table)
        else:
            story.append(Paragraph(sec, styles["Normal"]))

        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"📄 PDF généré : {output_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    system_prompt_file = "/Users/destine/Projets/hackaton_audia/comportement.txt"
    json_file = "/Users/destine/Projets/hackaton_audia/dps_analysis_pi3_exemple.json"

    system_prompt = read_file(system_prompt_file)
    mesures = read_json(json_file)

    processed_summary = preprocess_data(mesures)

    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content":
         "Voici le résumé structuré des données acoustiques :\n"
         f"{json.dumps(processed_summary, indent=2)}\n"
         "Génère un rapport acoustique complet, professionnel et lisible."
        }
    ]

    # Réception du rapport
    final_report = ""
    for chunk in ask_audia(chat_history):
        part = chunk.choices[0].delta.content
        if part:
            final_report += part
            print(part, end="")

    # PDF automatique
    save_pdf(final_report)