import os
import requests
import fitz  # PyMuPDF
from docx import Document
# import mailparser
from urllib.parse import urlparse

# === Base Abstract Class ===

class DocumentLoader:
    def __init__(self, source: str):
        self.source = source

    def extract(self):
        raise NotImplementedError("This method should be implemented in child classes.")

# === PDF Loader (Handles Local + URL) ===

class PDFLoader(DocumentLoader):
    def __init__(self, source: str):
        super().__init__(source)
        self.is_url = self.source.startswith("http")

    def _download_pdf(self):
        local_path = "temp_blob.pdf"
        response = requests.get(self.source)
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return local_path
        else:
            raise Exception(f"Failed to download PDF. Status: {response.status_code}")

    def extract(self):
        pdf_path = self._download_pdf() if self.is_url else self.source
        doc = fitz.open(pdf_path)
        clauses = []

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # Structured blocks
            for i, block in enumerate(blocks):
                text = block[4].strip()
                if len(text) > 50:  # Basic filter
                    clauses.append({
                        "text": text,
                        "page": page_num,
                        "position": i + 1
                    })
        doc.close()

        if self.is_url and os.path.exists(pdf_path):
            os.remove(pdf_path)  # Cleanup temp file

        return clauses

# === DOCX Loader ===

class DOCXLoader(DocumentLoader):
    def extract(self):
        doc = Document(self.source)
        clauses = []

        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                clauses.append({
                    "text": text,
                    "style": para.style.name,
                    "position": i + 1
                })

        return clauses

# === Email Loader (.eml files) ===

# class EmailLoader(DocumentLoader):
#     def extract(self):
#         mail = mailparser.parse_from_file(self.source)
#         return [{
#             "subject": mail.subject,
#             "from": mail.from_[0][1] if mail.from_ else "",
#             "to": mail.to[0][1] if mail.to else "",
#             "text": mail.body,
#             "date": str(mail.date)
#         }]

# === Main Wrapper Function ===

def load_document(source: str):
    parsed = urlparse(source)

    if source.endswith(".pdf") or parsed.scheme.startswith("http"):
        loader = PDFLoader(source)
    elif source.endswith(".docx"):
        loader = DOCXLoader(source)
    # elif source.endswith(".eml"):
    #     loader = EmailLoader(source)
    else:
        raise ValueError("Unsupported file format or source type.")

    content = loader.extract()
    return {
        "source": source,
        "clauses": content
    }


# if __name__ == '__main__':

#     output = load_document('https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D')
#     print("hello")
#     print(output['clauses'][4])