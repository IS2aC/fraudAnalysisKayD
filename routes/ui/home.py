from fastapi import APIRouter, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

router = APIRouter(
    prefix="/ui",
    tags=["UI"]
)

router.mount("/static", StaticFiles(directory="routes/ui/static"), name="static")
templates = Jinja2Templates(directory="routes/ui/templates")

# UI ENPOINTS ############
@router.get("/home", response_class=HTMLResponse)
async def ocr_document_ui(request: Request):
    return templates.TemplateResponse("home.html",  {"request": request})
##########################