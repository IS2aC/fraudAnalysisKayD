import base64
import io
import json
import os
import tempfile
from typing import List, Dict, Tuple

import requests
from PIL import Image
import pypdfium2 as pdfium
import pandas as pd
from datetime import datetime

from random import Random



# =========================
# CONFIG LM STUDIO
# =========================

LMSTUDIO_API_KEY = "lm-studio"               # valeur par défaut pour LM Studio
LMSTUDIO_MODEL_ID = "qwen3-vl-8b-instruct"   # adapte au nom exact dans LM Studio

# =========================
# CONSTANTES CNI
# =========================

RECTO_HINT_FIELDS = [
    "numero_pc",
    "nom",
    "prenoms",
    "categorie"
]



CNI_FIELDS = RECTO_HINT_FIELDS 

# =========================
# UTILITAIRES PDF -> IMAGES
# =========================

def pdf_bytes_to_pil_images(pdf_bytes: bytes, scale: float = 2.0) -> List[Image.Image]:
    """
    Convertit un PDF (en bytes) en une liste d'images PIL (une par page).
    Utilise pypdfium2 (API récente : page.render(...).to_pil()).
    Gère le cas Windows où le fichier temporaire peut rester verrouillé.
    """
    images: List[Image.Image] = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        pdf = pdfium.PdfDocument(tmp_path)
        n_pages = len(pdf)

        for i in range(n_pages):
            page = pdf.get_page(i)
            bitmap = page.render(
                scale=scale,
                rotation=0,
                crop=(0, 0, 0, 0),
            )
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
            # Sous Windows : peut arriver si le handle n'est pas encore relâché
            pass

    return images


def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    """
    Convertit une image PIL en base64 (JPEG).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

# =========================
# PROMPT & APPEL LLM
# =========================

def build_cni_prompt() -> str:
    """
    Construit le prompt système pour l'analyse de carte nationale d'identité ivoirienne.
    On demande explicitement au modèle de :
      - déterminer face = "recto" | "verso" | "inconnu"
      - ne remplir que certains champs selon la face.
    """
    return """
Tu es un assistant expert des permis de conduire ivoirien,  qui peut prendre les categories  : B | C | D | E

1) Ensuite, en fonction de la face détectée :

- Remplis uniquement les champs suivants:
    numero_pc,
    nom,
    prenoms,
    date_naissance,
    categorie


Rappels généraux :
- Si une information est absente ou illisible, mets la valeur à null.
- Utilise des chaînes de caractères (string) pour tous les champs.
- Pour les dates, utilise le format "dd/mm/yyyy" quand c'est possible.

Réponds STRICTEMENT en JSON avec la structure suivante :

{
    "numero_pc": "01-88-00000072",
    "nom": "ou null",
    "prenoms": "ou null",
    "date_naissance": "01/01/2025 à ABIDJAN COTE D'IVOIRE",
    "categorie": ""
}

Ne renvoie aucun texte en dehors de ce JSON.
""".strip()


def call_lmstudio_vision_analyse_permis(pil_image: Image.Image, lm_studio_base_url:str) -> Dict:
    """
    Appelle LM Studio (API OpenAI-like) avec un modèle multimodal (ex: qwen3-vl-8b-instruct)
    pour analyser une CNI. Aucun historique n'est envoyé -> contexte vidé à chaque appel.
    """
    system_prompt = build_cni_prompt()
    img_b64 = pil_to_base64_jpeg(pil_image)

    payload = {
        "model": LMSTUDIO_MODEL_ID,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
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
                        "text": (
                            "Analyse cette carte nationale d'identité selon les instructions "
                            "et extrais les champs demandés."
                        ),
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
    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Erreur LM Studio ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Réponse inattendue de LM Studio: {data}") from e

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Si le modèle ne renvoie pas un JSON propre
        parsed = {field: None for field in CNI_FIELDS}
        parsed["face"] = "inconnu"
        parsed["raw_response"] = content

    # S'assurer que tous les champs existent
    for f in CNI_FIELDS:
        if f not in parsed:
            parsed[f] = None

    # Normaliser la face
    face = (parsed.get("face") or "inconnu").lower()
    if face not in ("recto", "verso", "inconnu"):
        face = "inconnu"
    parsed["face"] = face

    return parsed



# =========================
# PIPELINE PRINCIPAL
# =========================

def analyse_passeport_permis_conduire(
    file_bytes: bytes,
    filename: str,
    lm_studio_url: str,
    pdf_scale: float = 2.0,
    doc_type: str = "Permis"
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
    result = call_lmstudio_vision_analyse_permis(
        pil_images[0],
        lm_studio_base_url=lm_studio_url
    )


    info = {
        "nom": result.get("nom"),
        "prenoms": result.get("prenoms"),
        "date_naissance": result.get('date_naissance'),
        "numero_doc": f"{result.get('numero_pc')} | {result.get("categorie")}" 
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
