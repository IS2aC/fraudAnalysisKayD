from fastapi import APIRouter, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from utils.models import DocPrisEnChargeParOcr

router = APIRouter(
    prefix="/ui",
    tags=["UI"]
)

router.mount("/static", StaticFiles(directory="routes/ui/static"), name="static")
templates = Jinja2Templates(directory="routes/ui/templates")

# UI ENPOINTS ############
@router.get("/ocr_document_ui", response_class=HTMLResponse)
async def ocr_document_ui(request: Request):
    # liste de documents pris en charge
    docs_list = list(DocPrisEnChargeParOcr) 

    return templates.TemplateResponse("ocr_document_ui.html",  {"request": request, "type_document": docs_list})
##########################