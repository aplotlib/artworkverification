import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image, ImageFile
from io import BytesIO
from pyzbar.pyzbar import decode as qr_decode
import pytesseract
import re
from typing import List, Dict, Any, Tuple
from config import AppConfig

# Allow processing of truncated image files which can sometimes occur in PDFs
ImageFile.LOAD_TRUNCATED_IMAGES = True

# UPGRADE: Cache the entire file processing function to speed up subsequent runs.
@st.cache_data
def process_files_cached(files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A cached wrapper around the main file processing logic."""
    # We instantiate the processor inside the cached function.
    processor = DocumentProcessor(files)
    return processor.process_files()


class DocumentProcessor:
    """Handles all file reading, text extraction (OCR), and data parsing."""

    def __init__(self, files: List[Dict[str, Any]]):
        self.files = [self._freeze_file(f) for f in files]

    def _freeze_file(self, file_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts file bytes to a hashable tuple for caching."""
        return {
            "name": file_dict["name"],
            "bytes": tuple(file_dict["bytes"])
        }

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

    def _extract_from_pdf_ocr(self, file_bytes: tuple) -> Dict[str, Any]:
        """Extracts text and QR codes from a PDF using OCR with enhanced error handling."""
        text, qr_data = "", []
        try:
            # Convert tuple back to bytes for processing
            file_bytes_io = BytesIO(bytes(file_bytes))
            with fitz.open(stream=file_bytes_io, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=300)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        processed_img = img.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
                        page_text = pytesseract.image_to_string(processed_img)
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        decoded_qrs = qr_decode(img)
                        qr_data.extend(obj.data.decode('utf-8') for obj in decoded_qrs)
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except fitz.errors.FitzError as e:
            return {'success': False, 'error': f"Failed to read PDF (file may be corrupted): {e}"}
        except Exception as e:
            return {'success': False, 'error': f"An unexpected error occurred during PDF processing: {e}"}

    def _find_sku_in_dataframe(self, df: pd.DataFrame) -> str | None:
        """Finds the Product SKU Code in a given DataFrame."""
        np_array = df.to_numpy()
        for row_idx, row in enumerate(np_array):
            for col_idx, cell in enumerate(row):
                if "PRODUCT SKU CODE" in str(cell).upper().strip() and (col_idx + 2) < len(row):
                    sku_candidate = np_array[row_idx, col_idx + 2]
                    if sku_candidate and isinstance(sku_candidate, str):
                        return sku_candidate.strip()
        return None

    def process_files(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Processes all files with enhanced error handling and classification."""
        processed_docs = []
        all_skus = set()

        for file in self.files:
            try:
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
                    else:
                        st.warning(f"Could not fully process '{file_name}': {result.get('error')}")

                elif file_name.lower().endswith(('.csv', '.xlsx')):
                    try:
                        # Convert tuple back to bytes for processing
                        file_bytes_io = BytesIO(bytes(file_bytes))
                        df = pd.read_excel(file_bytes_io, header=None).fillna('') if file_name.lower().endswith('.xlsx') else pd.read_csv(file_bytes_io, header=None).fillna('')
                        file_content = df.to_string()
                        sku = self._find_sku_in_dataframe(df)
                        if sku: all_skus.add(sku)
                    except Exception as e:
                        st.warning(f"Could not read spreadsheet '{file_name}': {e}")

                dims = re.findall(r'(\d+\.?\d*mm\s?\(?[W|H|D]\)?\s?x\s?\d+\.?\d*mm\s?\(?[W|H|D]\)?)', file_content)
                
                processed_docs.append({
                    'filename': file_name, 'text': file_content, 'doc_type': doc_type,
                    'file_nature': file_nature, 'qr_codes': qr_codes, 'dimensions': dims
                })
                
                skus_found = re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', file_content, re.IGNORECASE)
                for sku in skus_found:
                    all_skus.add(sku.upper())
            
            except Exception as e:
                st.error(f"Critical error while processing '{file.get('name', 'Unknown')}'. Skipping file. Error: {e}")
                continue

        return processed_docs, list(filter(None, all_skus))
