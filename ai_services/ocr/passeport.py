import base64
import io
import json
import os
import tempfile
from typing import List, Dict, Tuple
from datetime import datetime
import re

import requests
from PIL import Image
import pypdfium2 as pdfium
import pandas as pd


# =========================
# CONFIG LM STUDIO
# =========================

LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL_ID = "qwen3-vl-8b-instruct"   # adapte à ton modèle


# =========================
# CHAMPS PASSEPORT
# =========================

PASSPORT_FIELDS = [
    "type_document",
    "numero_passeport",
    "nom",
    "prenoms",
    "nationalite",
    "date_naissance",
    "lieu_naissance",
    "sexe",
    "date_delivrance",
    "date_expiration",
    "autorite_delivrance"
]


# =========================
# UTILITAIRES PDF -> IMAGES
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
# PROMPT PASSEPORT
# =========================

def build_passport_prompt() -> str:
    return """
CONSIGNE POUR L'ASSISTANT :
Tu es un expert en vérification de documents officiels, spécialisé dans les passeports de la République de Côte d'Ivoire. Ton rôle est d'analyser l'image pour extraire les données et évaluer l'authenticité du document.

1) ANALYSE DE STRUCTURE ET D'AUTHENTICITÉ :
Évalue les éléments suivants pour déterminer la validité du document :
- document_complet : Vérifie si la page de données (bas) ET la page d'informations (haut) sont présentes.
- mrz_valide : La zone de lecture optique (les caractères < en bas) doit correspondre au nom et au numéro de passeport.
- image_fantome : Présence de la version translucide de la photo en haut à droite.
- logos_officiels : Présence des logos CEDEAO/ECOWAS et des armoiries nationales.

2) EXTRACTION DES DONNÉES :
Extrais toutes les informations textuelles. Si une information est absente ou illisible, mets la valeur à null.

RAPPELS GÉNÉRAUX :
- Format des dates : "dd/mm/yyyy".
- Réponds STRICTEMENT en JSON.
- Ne renvoie aucun texte en dehors du bloc JSON.

STRUCTURE DE RÉPONSE ATTENDUE :

{
    "analyse_securite": {
        "document_complet": boolean,
        "authenticite_probable": "oui" | "non" | "douteux",
        "points_de_controle": {
            "mrz_presente": boolean,
            "image_fantome_visible": boolean,
            "logos_conformes": boolean,
            "dates_coherentes": boolean
        },
        "alertes": []
    },
    "donnees_titulaire": {
        "nom": "string",
        "prenoms": "string",
        "date_naissance": "dd/mm/yyyy",
        "lieu_naissance": "string",
        "sexe": "M" | "F",
        "nationalite": "string",
        "profession": "string",
    },
    "donnees_document": {
        "passeport_no": "string",
        "type": "P",
        "code_pays": "CIV",
        "date_emission": "dd/mm/yyyy",
        "date_expiration": "dd/mm/yyyy",
    }
}
""".strip()


# =========================
# APPEL LM STUDIO
# =========================

def call_lmstudio_vision_analyse_passport(pil_image: Image.Image, lm_studio_base_url: str) -> Dict:

    system_prompt = build_passport_prompt()
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyse ce passeport et renvoie les champs demandés."
                    }
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
        "Content-Type": "application/json"
    }

    url = f"{lm_studio_base_url}/v1/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"Erreur LM Studio {resp.status_code} : {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Parsing JSON
    try:
        parsed = json.loads(content)
    except:
        parsed = {f: None for f in PASSPORT_FIELDS}
        parsed["raw_response"] = content

    # champs manquants
    for f in PASSPORT_FIELDS:
        if f not in parsed:
            parsed[f] = None

    parsed["type_document"] = "Passeport"

    return parsed


# =========================
# SCORE D’AUTHENTICITÉ
# =========================

def compute_passport_score(result: dict):
    return None


# =========================
# PIPELINE PRINCIPAL
# =========================

def analyse_passeport_file(
    file_bytes: bytes,
    filename: str,
    lm_studio_url: str,
    pdf_scale: float = 2.0,
    seuil_score: int = 75, 
    doc_type: str = "Passeport"
):

    ext = os.path.splitext(filename)[1].lower()

    # PDF → images
    if ext == ".pdf":
        pil_images = pdf_bytes_to_pil_images(file_bytes, scale=pdf_scale)
        if not pil_images:
            raise ValueError("Impossible de lire le PDF.")
    else:
        pil_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    # On analyse la page biographique
    result = call_lmstudio_vision_analyse_passport(
        pil_images[0],
        lm_studio_base_url=lm_studio_url
    )

    
    info =  {
        "nom": result.get("donnees_titulaire").get("nom"), 
        "prenoms": result.get("donnees_titulaire").get("prenoms"),
        "date_naissance": result.get("donnees_titulaire").get("date_naissance"),
        "numero_doc": result.get("donnees_document").get("passeport_no"),
        "date_expiration": result.get("donnees_document").get("date_expiration")
    }

    return {
            "rapport":None, 
            "score":99, 
            "type_document":doc_type,
            "date_analyse": f"{datetime.now().strftime("%d/%m/%Y")} à {datetime.now().strftime("%H:%M")}",
            "info":info, 
            "verification_number": 3,
            "justify": None
    }
