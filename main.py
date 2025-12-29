from fastapi import FastAPI
from routes.ui import ocr_document_ui
from routes.api import ocr

app = FastAPI(
    title="API - IA Endpoint Creallia",
    description = """
        API-IA solution support pour une application d'analyse et de detections de transactions fauduleuses.
    """,
    version="0.0.0",
)



# UI   ENDPOINTS ############
app.include_router(ocr_document_ui.router)
#############################

# API ENDPOINTS ############
app.include_router(ocr.router)
#############################




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


