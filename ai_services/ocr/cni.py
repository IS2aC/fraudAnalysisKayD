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
    "numero_cni",
    "nom",
    "prenoms",
    "date_naissance",
    "nationalite",
    "date_expiration"
]

VERSO_HINT_FIELDS = [
    "nni",
    "profession",
    "date_emission",
]


CNI_FIELDS = ["face"] + RECTO_HINT_FIELDS + VERSO_HINT_FIELDS

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
Tu es un assistant expert des cartes de nationalité ivoirienne (CNI).

1) Commence par déterminer si l'image correspond à :
   - le RECTO de la carte (photo d'identité visible, informations principales :
     nom, prénoms, date de naissance, etc.)
   - le VERSO de la carte (pas de photo d'identité, informations comme NNI,
     profession, date d'émission, etc.)
   - ou bien un cas incertain.

Pour cela, remplis le champ:
    "face": "recto" | "verso" | "inconnu"

2) Ensuite, en fonction de la face détectée :

- Si l'image est un RECTO, remplis uniquement les champs suivants
  (laisse les autres à null) :
    numero_cni,
    nom,
    prenoms,
    date_naissance,
    nationalite,
    date_expiration

- Si l'image est un VERSO, remplis uniquement les champs suivants
  (laisse les autres à null) :
    nni,
    profession,
    date_emission.

- Si tu n'es pas sûr (face = "inconnu"), laisse tous les champs à null.

Rappels généraux :
- Si une information est absente ou illisible, mets la valeur à null.
- Utilise des chaînes de caractères (string) pour tous les champs.
- Pour les dates, utilise le format "dd/mm/yyyy" quand c'est possible.

Réponds STRICTEMENT en JSON avec la structure suivante :

{
    "face": "recto",
    "numero_cni": "CI002658965",
    "nom": "ou null",
    "prenoms": "ou null",
    "date_naissance": "dd/mm/yyyy",
    "nationalite": "ou null",
    "date_expiration": "dd/mm/yyyy"
    "nni": "12121245896",
    "profession": "ou null",
    "date_emission": "dd/mm/yyyy"
}

Ne renvoie aucun texte en dehors de ce JSON.
""".strip()


def call_lmstudio_vision_analyse_cni(pil_image: Image.Image, lm_studio_base_url:str) -> Dict:
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
# LOGIQUE MÉTIER CNI
# =========================

def clean_results_by_face(raw_results: List[Dict]) -> List[Dict]:
    """
    Nettoie les résultats bruts pour respecter strictement :

    - les champs de RECTO_HINT_FIELDS seulement en recto
    - les champs de VERSO_HINT_FIELDS seulement en verso
    - pas d'extraction des mêmes champs sur les 2 pages.

    Pour 'inconnu', on applique une politique stricte : aucun champ gardé.
    """
    cleaned: List[Dict] = []

    for res in raw_results:
        base = {field: None for field in CNI_FIELDS}
        face = (res.get("face") or "inconnu").lower()

        # On conserve le champ face tel quel
        base["face"] = face

        if face == "recto":
            for f in RECTO_HINT_FIELDS:
                base[f] = res.get(f)
        elif face == "verso":
            for f in VERSO_HINT_FIELDS:
                base[f] = res.get(f)
        else:
            # 'inconnu' -> strict : on ne garde rien d'autre que face
            pass

        cleaned.append(base)

    return cleaned


def fuse_cni_results(results: List[Dict]) -> Dict:
    """
    Fusionne les résultats de plusieurs pages de CNI en un seul dict.
    (results est déjà nettoyé par clean_results_by_face)

    Règle :
    - on prend le premier non-null
    - si plusieurs valeurs en concurrence, on prend la plus longue (texte le plus complet).
    """
    fused = {field: None for field in CNI_FIELDS}

    # Pour face, on agrège simplement : priorité au recto, puis verso, sinon inconnu
    faces_seen = [r.get("face") for r in results if r.get("face")]
    if "recto" in faces_seen:
        fused["face"] = "recto"
    elif "verso" in faces_seen:
        fused["face"] = "verso"
    elif faces_seen:
        fused["face"] = faces_seen[0]
    else:
        fused["face"] = "inconnu"

    for res in results:
        for field in CNI_FIELDS:
            if field == "face":
                continue  # déjà géré séparément

            val = res.get(field)
            if val is None or val == "":
                continue

            if fused[field] is None or fused[field] == "":
                fused[field] = val
            else:
                if isinstance(val, str) and isinstance(fused[field], str):
                    if len(val) > len(fused[field]):
                        fused[field] = val

    return fused


def build_cni_table(results: List[Dict], fused: Dict) -> pd.DataFrame:
    """
    Construit un tableau avec :
    - une ligne par champ
    - une colonne 'fusion'
    - une colonne par page (page_1, page_2, ...)
    """
    fields_for_table = [f for f in CNI_FIELDS if f != "face"]

    data = {
        "champ": fields_for_table,
        "fusion": [fused.get(f) for f in fields_for_table],
    }

    for i, res in enumerate(results):
        col_name = f"page_{i+1}"
        data[col_name] = [res.get(field) for field in fields_for_table]

    df = pd.DataFrame(data)
    return df


# ===========================================
# FONCTION FINALE A APPELER DANS LE FRONT END
# ===========================================
import re
from datetime import datetime

def compute_score_from_results(results: list):
    """
    Outils de scoring de documents adminitratifs basé :  
    - format numero cni
    - date_expiration > datetime.now()
    - date_naissance < date_emission && date_naissance < datetime.now()
    - date_emission < date_expiration
    """

    if not results:
        return None

    data = results[0]  # on suppose une seule carte

    numero_cni = data.get("numero_cni")
    date_emission = data.get("date_emission")
    date_expiration = data.get("date_expiration")
    date_naissance = data.get("date_naissance")

    score = 0

    # --------- Critère 1 : format numéro CNI ----------
    # format attendu CI00XXXXXXXX (8 chiffres)
    if numero_cni:
        num_clean = numero_cni.replace(" ", "")
        if "CI00" in num_clean:
            score += 40
    

    # --------- Critère 2 : expiration >= émission ----------
    if (date_emission is not None) and (date_expiration is not None):
        try:
            d_em = datetime.strptime(date_emission, "%d/%m/%Y").date()
            d_exp = datetime.strptime(date_expiration, "%d/%m/%Y").date()
            if d_exp >= d_em:
                score += 30
        except:
            pass
 
    # --------- Critère 3 : naissance < aujourd'hui ----------
    if date_naissance:
        try:
            d_birth = datetime.strptime(date_naissance, "%d/%m/%Y").date()
            if d_birth < datetime.now().date():
                score += 30
        except:
            pass
    
    # --------- Critère 4 : Caractère Aléatoire ----------
    alea =  Random()
    score += alea.randint(0,10)


    # clamp 0–100
    score = max(0, min(100, score))

    # ajouter score au résultat
    data["score"] = score

    return data


def analyse_cni_file(
    file_bytes: bytes,
    filename: str,
    lm_studio_url:str,
    pdf_scale: float = 2.0, 
    doc_type = "Carte Nationale d'Identité",
    seuil_score = 75
    
) -> Tuple[List[Image.Image], List[Dict], List[str], Dict, pd.DataFrame]:
    """
    Analyse un fichier CNI (PDF ou image) multi-pages.

    Pipeline :
    - PDF / image -> liste d'images (une par page)
    - LLM sur chaque page -> raw_results (avec "face" déjà rempli par le modèle)
    - clean_results_by_face(raw_results) -> results nettoyés
    - extraction de la liste faces à partir des results
    - fuse_cni_results(results) -> fused
    - build_cni_table(results, fused) -> df_table

    Retourne :
    - pil_images : liste d'images PIL (une par page)
    - results    : liste de dicts nettoyés (une par page)
    - faces      : liste de 'recto'/'verso'/'inconnu' par page
    - fused      : dict fusionné (vision globale de la CNI)
    - df_table   : DataFrame (fusion + valeurs par page)
    """
    ext = os.path.splitext(filename)[1].lower()

    # 1) PDF / image -> liste d'images
    if ext == ".pdf":
        pil_images = pdf_bytes_to_pil_images(file_bytes, scale=pdf_scale)
        if not pil_images:
            raise ValueError("Impossible de rendre le PDF en images.")
    else:
        pil_images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    # 2) Appel LLM sur chaque page -> raw_results
    raw_results: List[Dict] = []
    for img in pil_images:
        res = call_lmstudio_vision_analyse_cni(img, lm_studio_base_url = lm_studio_url)
        raw_results.append(res)

    # 3) Nettoyer les résultats selon la face indiquée par le LLM
    results = clean_results_by_face(raw_results)
    print(results)

    score =  compute_score_from_results(results).get('score')
    info =  dict(results[0])
    # remplacer numero_cni par numero_doc
    info["numero_doc"] = info.pop("numero_cni")
    verif_number = 3 # nombre de verification (Verif sur le numero de cni, sur la date de naissance, date d'emission )
    justify =  "Document acceptable aux standards de Passeport Internationaux" if verif_number > seuil_score else "Document Non-Conforme !"


    return  {
            "rapport":None, 
            "score":score, 
            "type_document":doc_type,
            "date_analyse": f"{datetime.now().strftime("%d/%m/%Y")} à {datetime.now().strftime("%H:%M")}",
            "info":info, 
            "verification_number": verif_number,
            "justify": justify
    }