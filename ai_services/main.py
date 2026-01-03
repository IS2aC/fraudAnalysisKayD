from ai_services.utils import execution_timer

from ai_services.ocr import cni, passeport, permis_conduire, rccm


class OcrProcessing:

    def __init__(self, doc_name, file_bytes):
        self.doc_name =  doc_name
        self.file_bytes =  file_bytes


    @execution_timer
    def make_ocr():
        pass