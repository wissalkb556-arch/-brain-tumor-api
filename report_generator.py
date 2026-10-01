from datetime import datetime


def generate_report(result: dict, patient_id: str = "ANONYME") -> str:
    pred, conf, bm = result['predicted_class'], result['confidence'] * 100, result['benign_malignant']
    confidence_note = (
        "\n⚠️ ATTENTION : confiance du modèle faible — examen par un radiologue fortement recommandé."
        if conf < 70 else ""
    )
    if pred == 'notumor':
        findings = "Aucune anomalie structurelle évocatrice d'une lésion tumorale n'a été détectée."
        impression = "Examen ne présentant pas de caractéristiques radiologiques associées à une tumeur cérébrale."
    else:
        findings = f"Une zone d'intérêt a été identifiée, compatible avec un(e) {pred}. La carte Grad-CAM localise la région ayant motivé cette classification."
        impression = f"Suspicion de {pred} ({bm}). Corrélation clinique et confirmation par un spécialiste indispensables."
    return f"""
{'='*60}
RAPPORT D'AIDE AU DIAGNOSTIC — SYSTÈME IA (NON DIAGNOSTIC FINAL)
{'='*60}
ID Patient        : {patient_id}
Date d'analyse    : {datetime.now().strftime('%Y-%m-%d %H:%M')}
Modalité          : IRM cérébrale, coupe axiale

FINDINGS
-------------------------
{findings}

IMPRESSION
-------------------------
{impression}

Classification    : {pred.upper()}
Confiance modèle  : {conf:.1f}%
Type              : {bm}
{confidence_note}

AVERTISSEMENT
-------------------------
Ce rapport est généré automatiquement par un système d'IA à titre
d'aide à la décision uniquement. Il doit être validé par un
professionnel de santé qualifié.
{'='*60}
"""
