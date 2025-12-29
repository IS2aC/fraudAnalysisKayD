from datetime import datetime

class OcrAuthFunctions:
    
    @staticmethod
    def ocr_auth_cni(file):
        return {
            "rapport":None, 
            "score":88, 
            "type_document":"Carte National d'Identité",
            "date_analyse": f"{datetime.now().strftime("%d/%m/%Y")} à {datetime.now().strftime("%H:%M")}",
            "verification_number": 36,
            "justify": "Document répondant aux standards de Carte Nationale d'identité en Cote d'Ivoire."
        }
    
    @staticmethod
    def ocr_auth_visa(file):
        return {
            "rapport":None, 
            "score":51, 
            "type_document":"Passeport",
            "date_analyse": f"{datetime.now().strftime("%d/%m/%Y")} à {datetime.now().strftime("%H:%M")}",
            "verification_number": 7,
            "justify": "Document répondant aux standards de Passeport Internationaux."
        }
    


    @staticmethod
    def ocr_auth(doctype, file):
        if doctype == "CNI":
            OcrAuthFunctions.ocr_auth_cni(file)
        else: 
            OcrAuthFunctions.ocr_auth_visa(file)