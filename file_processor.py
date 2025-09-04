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

    def _classify_document(self, filename: str) -> Tuple[str, str]:
        """Classifies a document and determines if it's shared or unique."""
        fn_lower = filename.lower()
        
        doc_type = "uncategorized"
        for dt, keywords in AppConfig.DOC_TYPE_MAP.items():
            if any(keyword in fn_lower for keyword in keywords):
                doc_type = dt
                break
        
        file_nature = "Unique Artwork"
        if any(keyword in fn_lower for keyword in AppConfig.SHARED_FILE_KEYWORDS):
            file_nature = "Shared File"
            
        return doc_type, file_nature

    def _extract_from_pdf_ocr(self, file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text and QR codes from a PDF using OCR."""
        text, qr_data = "", []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=300)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        page_text = pytesseract.image_to_string(img)
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                        # Decode QR codes from the high-res image
                        decoded_qrs = qr_decode(img)
                        qr_data.extend(obj.data.decode('utf-8') for obj in decoded_qrs)
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
        """Processes all files, classifies them, extracts data, and finds SKUs."""
        processed_docs = []
        all_skus = set()

        for file in self.files:
            file_name = file['name']
            file_bytes = file['bytes']
            doc_type, file_nature = self._classify_document(file_name)
            file_content = ""
            qr_codes = []
            
            if file_name.lower().endswith('.pdf'):
                result = self._extract_from_pdf_ocr(file_bytes)
                if result.get('success'):
                    file_content = result.get('text', '')
                    qr_codes = result.get('qr_data', [])
            
            elif file_name.lower().endswith(('.csv', '.xlsx')):
                try:
                    df = pd.read_excel(BytesIO(file_bytes), header=None).fillna('') if file_name.lower().endswith('.xlsx') else pd.read_csv(BytesIO(file_bytes), header=None).fillna('')
                    file_content = df.to_string()
                    sku = self._find_sku_in_dataframe(df)
                    if sku: all_skus.add(sku)
                except Exception as e:
                    file_content = f"Error reading spreadsheet file: {e}"

            # Extract dimensions using a regex pattern
            dims = re.findall(r'(\d+\.?\d*mm\s?\(?[W|H|D]\)?\s?x\s?\d+\.?\d*mm\s?\(?[W|H|D]\)?)', file_content)
            
            processed_docs.append({
                'filename': file_name, 
                'text': file_content, 
                'doc_type': doc_type,
                'file_nature': file_nature,
                'qr_codes': qr_codes,
                'dimensions': dims
            })
            
            skus_found = re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', file_content, re.IGNORECASE)
            for sku in skus_found:
                all_skus.add(sku.upper())
        
        return processed_docs, list(filter(None, all_skus))
