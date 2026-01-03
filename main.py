from fastapi import FastAPI, Request
from routes.ui import ocr_document_ui, home
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import HTMLResponse
from routes.api import ocr


app = FastAPI(
    title="API - IA Endpoint Creallia",
    description = """
        API-IA solution support pour une application d'analyse et de detections de transactions fauduleuses.
    """,
    version="0.0.0",
)


# --------------------------
#  404 CUSTOM
# --------------------------
@app.exception_handler(404)
async def not_found_page(request: Request, exc: StarletteHTTPException):
    return home.templates.TemplateResponse(
        "404.html",
        {"request": request},
        status_code=404
    )



# UI   ENDPOINTS ############
app.include_router(ocr_document_ui.router)
app.include_router(home.router)
#############################

# API ENDPOINTS ############
app.include_router(ocr.router)
#############################




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


