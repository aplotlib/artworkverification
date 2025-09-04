import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageFile
from io import BytesIO
from pyzbar.pyzbar import decode as qr_decode
import re
from typing import List, Dict, Any, Tuple

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DocumentProcessor:
    """Handles all file reading, text extraction (OCR), and data parsing."""

    def __init__(self, files: List[Dict[str, Any]]):
        """
        Args:
            files: A list of dictionaries, where each dict has 'name' and 'bytes'.
        """
        self.files = files

    def _extract_from_pdf(self, file_bytes: bytes) -> Dict[str, Any]:
        """Extracts text and QR codes from a PDF file's bytes."""
        text, qr_data = "", []
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
                    # Increase DPI for better OCR of small text
                    pix = page.get_pixmap(dpi=200)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        qr_data.extend(obj.data.decode('utf-8') for obj in qr_decode(img))
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _extract_from_qc_sheet(self, file_bytes: bytes) -> Dict[str, Any]:
        """Robustly extracts SKU and Color from a QC sheet (CSV)."""
        try:
            df = pd.read_csv(BytesIO(file_bytes), dtype=str, header=None).fillna('')
            specs = {}
            # Convert dataframe to numpy array for faster iteration
            np_array = df.to_numpy()
            
            # Find the exact cell locations of headers, then get adjacent values
            for row_idx, row in enumerate(np_array):
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell).upper().strip()
                    if "PRODUCT SKU CODE" in cell_str and (col_idx + 2) < len(row):
                        specs['sku'] = np_array[row_idx, col_idx + 2]
                    if "PRODUCT COLOR" in cell_str and (col_idx + 2) < len(row):
                        specs['color'] = np_array[row_idx, col_idx + 2]
            
            if 'sku' not in specs:
                return {'success': False, 'error': "Could not find 'PRODUCT SKU CODE' in QC sheet."}
            
            return {'success': True, **specs, 'text': df.to_string()}
        except Exception as e:
            return {'success': False, 'error': f"Could not parse QC Sheet: {e}"}


    def process_files(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Processes all uploaded files and returns extracted data and detected SKUs."""
        processed_docs = []
        all_skus = set()

        for file in self.files:
            file_name = file['name']
            file_bytes = file['bytes']
            file_content = ""
            
            if file_name.lower().endswith('.pdf'):
                result = self._extract_from_pdf(file_bytes)
                if result.get('success'):
                    file_content = result.get('text', '')
                    processed_docs.append({'filename': file_name, 'text': file_content})
            
            elif file_name.lower().endswith(('.csv', '.xlsx')):
                # Even for QC sheets, we treat them as text for the unified analysis
                file_content = BytesIO(file_bytes).read().decode('utf-8', errors='ignore')
                processed_docs.append({'filename': file_name, 'text': file_content})
                
                # Also extract SKU for variant detection
                qc_data = self._extract_from_qc_sheet(file_bytes)
                if qc_data.get('success'):
                    all_skus.add(qc_data.get('sku'))

            # Also find SKUs from general text content
            skus_found = re.findall(r'([A-Z]{3,}\d{3,}[A-Z]*)', file_content, re.IGNORECASE)
            for sku in skus_found:
                all_skus.add(sku.upper())
        
        return processed_docs, list(filter(None, all_skus))
