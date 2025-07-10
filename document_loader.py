# document_loader.py
from PyPDF2 import PdfReader

def load_pdf(file):
   
    pdf = PdfReader(file)
    full_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text
