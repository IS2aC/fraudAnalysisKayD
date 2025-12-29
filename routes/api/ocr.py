""" Code des endpoint en charge des uses case OCR """
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from ai_services.ocr import OcrAuthFunctions as oaf

router = APIRouter(
    prefix="/ai-api",
    tags=["OCR"]
)

@router.post(
    "/ocr_document",
    summary="Endpoint en charge des opérations d'OCR ponctuel"
)
async def analyse(type_document: str = Form(...), file: UploadFile = File(...)):
    # doc_type =  str(type_document.strip())
    try:
        # Vérification du type de fichier
        if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Format non supporté"
            )

        # Lecture du fichier
        file_bytes = await file.read()

        # Appel OCR
        analyse_result = oaf.ocr_auth_cni(file=file_bytes)

        # Réponse OK
        return {
            "code": 200,
            "message": "Analyse effectuée avec succès",
            "model_response": analyse_result
        }

    except HTTPException:
        # Laisser passer les HTTPException déjà levées
        raise

    except Exception as e:
        # Erreur serveur inattendue
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de l'analyse OCR : {str(e)}"
        )
