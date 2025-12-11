import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage

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

    # ---- Valeurs brutes pour histogramme ----
    summary["raw_LAeq_values"] = df["LAeq_segment_dB"].tolist()

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

    # S'assurer que top_5_labels contient des listes
    df["top_5_labels"] = df["top_5_labels"].apply(lambda x: x if isinstance(x, list) else [])

    all_labels = df["top_5_labels"].explode()
    family_counts = {fam: 0 for fam in families}

    for label in all_labels:
        lname = str(label).lower()
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
    seuil = freq.quantile(0.75)  # 75% quantile pour les bruits exceptionnels

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
# GRAPHIQUES
# =========================
def generate_graphs(processed_summary):
    image_paths = []
    palette = {
        "primary": "#1F77B4",
        "secondary": "#2ECC71",
        "tertiary": "#E67E22",
        "background": "#F4F6F7"
    }

    sns.set_theme(style="whitegrid")

    # ---------- Top 5 bruits ----------
    top5 = dict(list(processed_summary["top_20_bruits"].items())[:5])
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(top5.keys()), y=list(top5.values()), palette=[palette["primary"]]*len(top5))
    plt.title("Top 5 des bruits détectés", fontsize=16, fontweight="bold")
    plt.ylabel("Occurrences")
    plt.xticks(rotation=25)
    for i, value in enumerate(top5.values()):
        plt.text(i, value + max(top5.values())*0.02, value, ha='center', fontsize=11)
    img1 = os.path.abspath("top5_bruits.png")
    plt.tight_layout()
    plt.savefig(img1, dpi=300)
    plt.close()
    image_paths.append(img1)

    # ---------- LAeq Jour vs Nuit ----------
    jn = processed_summary["jour_nuit"]
    labels = ["Jour", "Nuit"]
    values = [jn["mean"]["jour"], jn["mean"]["nuit"]]
    plt.figure(figsize=(8,5))
    sns.barplot(x=labels, y=values, palette=[palette["primary"], palette["secondary"]])
    plt.title("LAeq Moyen – Jour vs Nuit", fontsize=16, fontweight="bold")
    plt.ylabel("Niveau sonore (dB)")
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value} dB", ha='center', fontsize=11)
    img2 = os.path.abspath("jour_vs_nuit.png")
    plt.tight_layout()
    plt.savefig(img2, dpi=300)
    plt.close()
    image_paths.append(img2)

    # ---------- Min / Moyenne / Max LAeq ----------
    stats = processed_summary["stats_globales"]
    labels = ["Min", "Moyenne", "Max"]
    values = [stats["LAeq_min"], stats["LAeq_moyen"], stats["LAeq_max"]]
    plt.figure(figsize=(8,5))
    sns.barplot(x=labels, y=values, palette=[palette["secondary"], palette["primary"], palette["tertiary"]])
    plt.title("LAeq – Minimum / Moyenne / Maximum", fontsize=16, fontweight="bold")
    plt.ylabel("Décibels (dB)")
    for i, value in enumerate(values):
        plt.text(i, value + 0.5, f"{value} dB", ha='center', fontsize=11)
    img3 = os.path.abspath("stats_globales.png")
    plt.tight_layout()
    plt.savefig(img3, dpi=300)
    plt.close()
    image_paths.append(img3)

    # ---------- Histogramme LAeq ----------
    plt.figure(figsize=(8,5))
    sns.histplot(processed_summary["raw_LAeq_values"], kde=True, color=palette["primary"])
    plt.title("Distribution des niveaux LAeq", fontsize=16, fontweight="bold")
    plt.xlabel("Niveau sonore (dB)")
    plt.ylabel("Fréquence")
    img4 = os.path.abspath("distribution_LAeq.png")
    plt.tight_layout()
    plt.savefig(img4, dpi=300)
    plt.close()
    image_paths.append(img4)

    return image_paths

# =========================
# PDF
# =========================
def save_pdf(report_text, output_path="rapport_acoustique.pdf", image_paths=None):
    if image_paths is None:
        image_paths = []

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitleCustom", fontSize=28, leading=32, textColor=colors.HexColor("#1A5276"), alignment=1, spaceAfter=20, bold=True))
    styles.add(ParagraphStyle(name="CoverSubtitleCustom", fontSize=16, leading=20, textColor=colors.HexColor("#2471A3"), alignment=1, spaceAfter=30))
    styles.add(ParagraphStyle(name="CoverInfoCustom", fontSize=12, leading=18, textColor=colors.HexColor("#2C3E50"), alignment=0, spaceAfter=8))
    styles.add(ParagraphStyle(name="Title1Custom", fontSize=20, leading=26, textColor=colors.HexColor("#1A5276"), spaceAfter=14, bold=True))
    styles.add(ParagraphStyle(name="Title2Custom", fontSize=16, leading=22, textColor=colors.HexColor("#2471A3"), spaceAfter=10, bold=True))
    styles.add(ParagraphStyle(name="Title3Custom", fontSize=13, leading=18, textColor=colors.HexColor("#2E86C1"), spaceAfter=8, bold=True))
    styles.add(ParagraphStyle(name="BodyCustom", fontSize=11, leading=16, spaceAfter=10))

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []

    # Page de garde
    story.append(Paragraph("Rapport d’analyse acoustique résidentielle", styles["CoverTitleCustom"]))
    story.append(Paragraph("Analyse des niveaux sonores & recommandations techniques", styles["CoverSubtitleCustom"]))
    story.append(Paragraph("Préparé par AudIA – Expert en acoustique résidentielle", styles["CoverInfoCustom"]))
    story.append(Spacer(1, 40))

    # Contenu
    sections = report_text.split("\n\n")
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        sec = sec.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        if sec.startswith("## "):
            story.append(Paragraph(sec[3:], styles["Title1Custom"]))
        elif sec.startswith("### "):
            story.append(Paragraph(sec[4:], styles["Title2Custom"]))
        elif sec.startswith("#### "):
            story.append(Paragraph(sec[5:], styles["Title3Custom"]))
        elif sec.startswith("|"):
            rows = [row.split("|")[1:-1] for row in sec.split("\n") if "|" in row]
            table_data = [[Paragraph(cell.strip(), styles["BodyCustom"]) for cell in r] for r in rows]
            table = Table(table_data, hAlign="LEFT")
            table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
            story.append(table)
        else:
            story.append(Paragraph(sec, styles["BodyCustom"]))
        story.append(Spacer(1, 12))

    # Ajout des images
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                pil_img = PILImage.open(img_path)
                ratio = pil_img.height / pil_img.width
                story.append(Image(img_path, width=450, height=450*ratio))
                story.append(Spacer(1, 14))
            except Exception as e:
                print("Erreur image :", e)

    doc.build(story)
    print(f"✅ PDF généré : {output_path}")

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
        try:
            part = chunk.choices[0].delta.content
            if part:
                final_report += part
                print(part, end="")
        except Exception:
            continue

    # PDF automatique avec visualisations
    image_paths = generate_graphs(processed_summary)
    save_pdf(final_report, image_paths=image_paths)
