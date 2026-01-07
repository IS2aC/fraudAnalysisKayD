# lmstudio_llm.py

import base64
import io
import json
import os
import tempfile
from typing import List, Dict

import requests
from PIL import Image
import pypdfium2 as pdfium
from datetime import datetime


LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL_ID = "qwen3-vl-8b-instruct"


# =========================
# PDF -> IMAGES
# =========================

def pdf_bytes_to_pil_images(pdf_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    images: List[Image.Image] = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        pdf = pdfium.PdfDocument(tmp_path)
        n_pages = len(pdf)

        for i in range(n_pages):
            page = pdf.get_page(i)
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image.convert("RGB"))
            bitmap.close()
            page.close()

        pdf.close()

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except PermissionError:
            pass

    return images


def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =========================
# PROMPT RCCM
# =========================

def build_rccm_extraction_prompt():
    return """
Tu es un assistant OCR spécialisé dans l'analyse de documents administratifs
ivoiriens, chargé d'extraire les informations d'un certificat d'immatriculation
au RCCM (Registre du Commerce et du Crédit Mobilier).

===================================================
STRUCTURE TYPE D'UN CERTIFICAT RCCM
===================================================

Un certificat RCCM authentique comporte généralement :

1) EN-TÊTE :
   - Logo/en-tête du Tribunal de Commerce d'Abidjan
   - Titre : "DECLARATION D'IMMATRICULATION DE PERSONNE MORALE"
   - Formulaire type : "RCCM 2014-M1" ou similaire
   - Numéro RCCM principal (ex : CI-ABJ-03-2023-B12-04308)

2) SECTION 1 - INFORMATIONS SUR LA PERSONNE MORALE :
   - Raison ou dénomination sociale
   - Nom commercial / Sigle / Enseigne
   - Forme juridique (SARL, SA, SAS, etc.)
   - Capital social (en chiffres et en lettres)
   - Détails du capital (numéraire, nature, industrie)
   - Adresse du siège social
   - Adresse de l'établissement créé
   - Durée de la société (généralement en années)

3) SECTION 2 - ACTIVITÉS ET ÉTABLISSEMENTS :
   - Activité(s) exercée(s) : description détaillée
   - Date de début d'activité
   - Origine du fonds (Création, Achat, Apport, etc.)
   - Nombre de salariés
   - Établissements secondaires (le cas échéant)

4) SECTION 3 - ASSOCIÉS :
   - Nom et prénoms / dénomination des associés
   - Genre (M/F)
   - Date et lieu de naissance
   - Adresse

5) SECTION 4 - DIRIGEANTS :
   - Nom et prénoms du/des dirigeant(s)
   - Genre (M/F)
   - Date et lieu de naissance
   - Adresse
   - Qualité (Gérant, Directeur Général, Président, etc.)

6) SECTION 5 - COMMISSAIRES AUX COMPTES (si applicable) :
   - Cabinet / Nom et Prénoms
   - Numéro d'agrément
   - Adresse

7) SECTION 6 - SIGNATURE ET VALIDATION :
   - Nom du mandataire/demandeur
   - Date de demande d'immatriculation
   - Date d'immatriculation effective
   - **TAMPON DU GREFFIER** avec :
     * Nom et titre du greffier
     * Mention "GREFFIER EN CHEF" ou "GREFFIER EN CHEF ADJOINT"
     * Mention "TRIBUNAL DE COMMERCE D'ABIDJAN"
     * Date et signature du greffier
   - Numéro RCCM attribué (confirmation)

8) BAS DE PAGE :
   - Mentions de vérification
   - Contact du greffe (FAX, site web)
   - Numéro de page (ex : "Page 1 sur 2")

===========================
ÉLÉMENTS D'AUTHENTICITÉ
===========================

Un certificat RCCM authentique DOIT contenir :

1) **TAMPON DU GREFFIER** :
   - Tampon officiel visible avec cachet circulaire ou rectangulaire
   - Nom du greffier (ex : KOUASSI KONAN MATHIAS)
   - Titre : "Administrateur des Greffes et Parquets" ou similaire
   - Mention "GREFFIER EN CHEF ADJOINT" ou "GREFFIER EN CHEF"
   - "TRIBUNAL DE COMMERCE D'ABIDJAN"
   - Signature manuscrite du greffier

2) **COHÉRENCE DES DATES** :
   - Date de demande ≤ Date d'immatriculation
   - Date d'immatriculation ≤ Date du document

3) **FORMAT DU NUMÉRO RCCM** :
   - Structure type : CI-ABJ-XX-YYYY-BXX-XXXXX
   - CI = Côte d'Ivoire
   - ABJ = Abidjan
   - Format cohérent dans tout le document

===========================
TÂCHE D'EXTRACTION
===========================

À partir de l'image fournie, tu dois extraire TOUTES les informations
disponibles en respectant la structure ci-dessus.

Pour chaque champ :
- Extrais la valeur exacte si elle est présente
- Mets null si l'information n'est pas visible ou illisible
- Indique "Non spécifié" si le champ existe mais est vide

===========================
FORMAT DE SORTIE (JSON)
===========================

Réponds STRICTEMENT avec un JSON de la forme :

{
  "type_document": "RCCM",
  "numero_rccm": "CI-ABJ-03-2023-B12-04308",
  "authenticite": {
    "tampon_greffier_present": true,
    "nom_greffier": "KOUASSI KONAN MATHIAS",
    "titre_greffier": "GREFFIER EN CHEF ADJOINT",
    "signature_presente": true,
    "date_signature": "20/07/2023",
    "tribunal": "TRIBUNAL DE COMMERCE D'ABIDJAN",
    "score_authenticite": 0.95
  },
  "personne_morale": {
    "denomination_sociale": "EXPERTS ASSOCIES POUR LA TRANSFORMATION DIGITALE ET SERVICES",
    "nom_commercial": null,
    "sigle": "EXAT CONSULTING",
    "enseigne": null,
    "forme_juridique": "SARL",
    "capital_social": {
      "montant_chiffres": 1000000,
      "devise": "F CFA",
    },
    "siege_social": {
      "adresse_complete": "ABIDJAN COCODY RIVIERA PALMERAIE SIPIM 6 LOT 136 SECTION EM PARCELLE 91",
      "ville": "ABIDJAN",
      "commune": "COCODY",
      "quartier": "RIVIERA PALMERAIE"
    },
    "activites": {
        "description": "SOCIETE DE PRESTATION DE SERVICES INFORMATIQUES, INTEGRATEUR DE SOLUTIONS INFORMATIQUES, FORMATIONS, VENTE DE MATERIELS, CONSEILS ET ACCOMPAGNEMENT DES PROJETS INFORMATIQUES, IMPORT-EXPORT, DIVERSES PRESTATIONS",
        "date_debut": "05/07/2023",
        "origine_fonds": "Création",
        "nombre_salaries": 0,
        "etablissements_secondaires": false
    },
    "dirigeants": [
        {
        "nom_prenoms": "TRAORE AHMED ERIC HERMANN",
        "genre": "M",
        "date_naissance": "15/11/1974",
        "lieu_naissance": "BOBO DIOULASSO",
        "pays_naissance": "Burkina Faso",
        "adresse": null,
        "qualite": "Gérant(e)"
    }],
    "qualite_extraction": {
        "document_lisible": true,
        "completude": 0.92,
        "confiance_globale": 0.88,
        "champs_illisibles": [],
        "observations": "Document complet et authentique. Toutes les informations principales ont été extraites avec succès."
    }
}

===========================
RÈGLES IMPORTANTES
===========================

1) **Authenticité** :
   - score_authenticite entre 0 et 1 (0 = probablement faux, 1 = très authentique)
   - Facteurs : présence du tampon, cohérence des dates, format RCCM, signatures

2) **Extraction des montants** :
   - Convertis toujours les montants en nombres (sans espaces ni séparateurs)
   - Garde la devise telle quelle (F CFA, FCFA, etc.)

3) **Dates** :
   - Format : "JJ/MM/AAAA"
   - Extrais exactement comme sur le document

4) **Listes** :
   - Pour les activités multiples, crée un array
   - Pour les dirigeants/associés multiples, crée un array d'objets

5) **Champs vides** :
   - null si l'information n'existe pas
   - [] (array vide) pour les listes sans éléments
   - "Non spécifié" si le champ existe mais est vide

6) **Qualité d'extraction** :
   - completude : pourcentage de champs remplis (0 à 1)
   - confiance_globale : confiance moyenne sur la lecture (0 à 1)
   - Liste les champs illisibles ou incertains

Ne renvoie AUCUN texte en dehors de ce JSON.
Sois précis et exhaustif dans l'extraction.
""".strip()


# =========================
# PARSING
# =========================

def parse_rccm_document(res_llm: dict) -> dict:
    return {
        "type_document": res_llm.get("type_document", "RCCM"),
        "numero_rccm": res_llm.get("numero_rccm"),

        "authenticite": res_llm.get("authenticite", {}),

        "personne_morale": res_llm.get("personne_morale", {}),

        "activites": res_llm.get("activites", {}),

        "dirigeants": res_llm.get("dirigeants", []),
        "associes": res_llm.get("associes", []),

        "dates_importantes": res_llm.get("dates_importantes", {}),

        "qualite_extraction": res_llm.get("qualite_extraction", {}),
    }


# =========================
# APPEL LLM
# =========================

def call_lmstudio_vision_analyse_rccm(
    pil_image: Image.Image,
    lm_studio_base_url: str
) -> dict:

    system_prompt = build_rccm_extraction_prompt()
    img_b64 = pil_to_base64_jpeg(pil_image)

    payload = {
        "model": LMSTUDIO_MODEL_ID,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    },
                    {
                        "type": "text",
                        "text": "Analyse ce certificat RCCM et renvoie STRICTEMENT le JSON demandé."
                    },
                ],
            },
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
    }

    url = f"{lm_studio_base_url}/v1/chat/completions"

    resp = requests.post(url, headers=headers, json=payload, timeout=200)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Tentative parsing JSON strict
    try:
        res_llm = json.loads(content)
    except json.JSONDecodeError as e:
        res_llm = {
            "type_document": "RCCM",
            "numero_rccm": None,
            "personne_morale": {},
            "activites": {},
            "dirigeants": [],
            "associes": [],
            "qualite_extraction": {
                "document_lisible": False,
                "observations": f"Erreur parsing JSON: {str(e)}",
            },
            "raw_response": content,
        }

    return parse_rccm_document(res_llm)


# =========================
# FONCTION FINALE UTILISÉE PAR L’API
# =========================

def analyse_rccm(
    file_bytes: bytes,
    filename: str,
    lm_studio_url: str,
    pdf_scale: float = 2.0,
    doc_type: str = "rccm"
) -> Dict:

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        pil_images = pdf_bytes_to_pil_images(file_bytes, scale=pdf_scale)
        if not pil_images:
            raise ValueError("Impossible de rendre le PDF en images.")
    else:
        pil_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    # une seule page analysée pour RCCM généralement
    result = call_lmstudio_vision_analyse_rccm(pil_images[0], lm_studio_base_url=lm_studio_url)

    # informations principales pour l'API
    info = {
        "nom": result.get("personne_morale", {}).get("denomination_sociale"),
        "prenoms": "Personne Morale",
        "date_naissance": None,
        "numero_doc": result.get("numero_rccm"),
        "date_expiration": None,
    }

    return {
        "rapport": None,
        "score": 25,
        "type_document": doc_type,
        "date_analyse": f"{datetime.now().strftime('%d/%m/%Y')} à {datetime.now().strftime('%H:%M')}",
        "info": info,
        "verification_number": 3,
        "justify": None,
        "extraction_detaillee": result
    }
