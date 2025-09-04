import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageFile
from io import BytesIO
from pyzbar.pyzbar import decode as qr_decode
import pytesseract
import re
from typing import List, Dict, Any, Tuple
from config import AppConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DocumentProcessor:
    """Handles all file reading, text extraction (OCR), and data parsing."""

    def __init__(self, files: List[Dict[str, Any]]):
        self.files = files

    def _classify_document(self, filename: str) -> str:
        """Classifies a document based on keywords in its filename."""
        fn_lower = filename.lower()
        for doc_type, keywords in AppConfig.DOC_TYPE_MAP.items():
            if any(keyword in fn_lower for keyword in keywords):
                return doc_type
        return "uncategorized"

    def _extract_from_pdf_ocr(self, file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from a PDF using OCR for maximum accuracy."""
        text, qr_data = "", []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=300)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        page_text = pytesseract.image_to_string(img)
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                        qr_data.extend(obj.data.decode('utf-8') for obj in qr_decode(img))
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _find_sku_in_dataframe(self, df: pd.DataFrame) -> str | None:
        """Finds the Product SKU Code in a given DataFrame."""
        np_array = df.to_numpy()
        for row_idx, row in enumerate(np_array):
            for col_idx, cell in enumerate(row):
                if "PRODUCT SKU CODE" in str(cell).upper().strip() and (col_idx + 2) < len(row):
                    return np_array[row_idx, col_idx + 2]
        return None

    def process_files(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Processes all files, classifies them, extracts text, and finds SKUs."""
        processed_docs = []
        all_skus = set()

        for file in self.files:
            file_name = file['name']
            file_bytes = file['bytes']
            doc_type = self._classify_document(file_name)
            file_content = ""
            
            if file_name.lower().endswith('.pdf'):
                result = self._extract_from_pdf_ocr(file_bytes)
                if result.get('success'):
                    file_content = result.get('text', '')
            
            elif file_name.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(BytesIO(file_bytes), header=None).fillna('')
                    file_content = df.to_string()
                    sku = self._find_sku_in_dataframe(df)
                    if sku: all_skus.add(sku)
                except Exception as e:
                    file_content = f"Error reading CSV file: {e}"

            elif file_name.lower().endswith('.xlsx'):
                try:
                    xls = pd.read_excel(BytesIO(file_bytes), sheet_name=None, header=None)
                    all_sheets_text = []
                    # Assume SKU is on the first sheet for QC purposes
                    first_sheet_name = list(xls.keys())[0]
                    sku = self._find_sku_in_dataframe(xls[first_sheet_name].fillna(''))
                    if sku: all_skus.add(sku)
                    
                    for sheet_name, df in xls.items():
                        all_sheets_text.append(f"--- Sheet: {sheet_name} ---\n{df.to_string()}")
                    file_content = "\n\n".join(all_sheets_text)
                except Exception as e:
                    file_content = f"Error reading XLSX file: {e}"

            processed_docs.append({'filename': file_name, 'text': file_content, 'doc_type': doc_type})
            
            skus_found = re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', file_content, re.IGNORECASE)
            for sku in skus_found:
                all_skus.add(sku.upper())
        
        return processed_docs, list(filter(None, all_skus))
