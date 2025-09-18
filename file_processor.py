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
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from collections import defaultdict
from ai_analyzer import AIReviewer, check_api_keys

ImageFile.LOAD_TRUNCATED_IMAGES = True

@st.cache_data
def process_files_cached(files: Tuple[Dict[str, Any], ...]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A cached wrapper around the main file processing logic."""
    processor = DocumentProcessor(list(files))
    return processor.process_files()

class DocumentProcessor:
    """Handles all file reading, text extraction, and brand compliance analysis."""

    def __init__(self, files: List[Dict[str, Any]]):
        self.files = files
        self._setup_brand_colors()
        # Initialize AIReviewer once for OCR corrections
        api_keys = check_api_keys()
        self.ai_reviewer = AIReviewer(api_keys) if api_keys.get('openai') else None


    def _setup_brand_colors(self):
        """Pre-calculates LabColor objects for brand colors for efficient comparison."""
        self.brand_colors_lab = []
        brand_color_data = AppConfig.BRAND_GUIDE['colors']['brand_color']
        main_srgb = sRGBColor(rgb_r=brand_color_data['rgb'][0], rgb_g=brand_color_data['rgb'][1], rgb_b=brand_color_data['rgb'][2], is_upscaled=True)
        self.brand_colors_lab.append({"name": brand_color_data['name'], "lab": convert_color(main_srgb, LabColor)})
        
        for color in AppConfig.BRAND_GUIDE['colors']['complementary_colors']:
            rgb = color['rgb']
            srgb = sRGBColor(rgb_r=rgb[0], rgb_g=rgb[1], rgb_b=rgb[2], is_upscaled=True)
            self.brand_colors_lab.append({"name": color['name'], "lab": convert_color(srgb, LabColor)})

    def _get_closest_brand_color(self, rgb_tuple: Tuple[int, int, int]) -> Tuple[str, float]:
        """Finds the closest brand color to a given RGB value using Delta E."""
        if not rgb_tuple or len(rgb_tuple) < 3: return "N/A", float('inf')
        color_to_check_srgb = sRGBColor(rgb_r=rgb_tuple[0], rgb_g=rgb_tuple[1], rgb_b=rgb_tuple[2], is_upscaled=True)
        color_to_check_lab = convert_color(color_to_check_srgb, LabColor)
        
        min_delta_e, closest_color_name = min(
            ((delta_e_cie2000(color_to_check_lab, bc["lab"]), bc["name"]) for bc in self.brand_colors_lab),
            key=lambda x: x[0]
        )
        return closest_color_name, min_delta_e

    def _classify_document(self, filename: str) -> Tuple[str, str]:
        """Classifies a document and determines if it's shared or unique."""
        fn_lower = filename.lower()
        doc_type = "uncategorized"
        for dt, keywords in AppConfig.DOC_TYPE_MAP.items():
            if any(keyword in fn_lower for keyword in keywords):
                doc_type = dt
                break
        file_nature = "Shared File" if any(keyword in fn_lower for keyword in AppConfig.SHARED_FILE_KEYWORDS) else "Unique Artwork"
        return doc_type, file_nature

    def _analyze_pdf_brand_compliance(self, file_bytes: tuple) -> Dict[str, Any]:
        """Extracts fonts and colors from a PDF for brand compliance checking."""
        fonts, colors = set(), defaultdict(int)
        try:
            doc = fitz.open(stream=BytesIO(bytes(file_bytes)), filetype="pdf")
            for page in doc:
                fonts.update(font[3] for font in page.get_fonts())
                for drawing in page.get_drawings():
                    rgb = tuple(int(c * 255) for c in drawing.get("color", [])[:3])
                    if rgb and rgb not in AppConfig.BRAND_GUIDE["COLOR_IGNORE_LIST"]:
                        colors[rgb] += 1
            doc.close()
            
            color_analysis = [
                {
                    "rgb": rgb,
                    "closest_brand_color": (closest_name := self._get_closest_brand_color(rgb))[0],
                    "delta_e": (delta_e := closest_name[1]),
                    "compliant": delta_e <= AppConfig.BRAND_GUIDE["color_tolerance"]
                } for rgb in colors
            ]
            return {"success": True, "fonts": list(fonts), "colors": color_analysis}
        except Exception as e:
            return {"success": False, "error": f"Brand compliance analysis failed: {e}"}

    def _extract_from_pdf(self, file_bytes: tuple) -> Dict[str, Any]:
        """Extracts text and QR codes from a PDF using a hybrid approach."""
        text, qr_data = "", []
        try:
            doc = fitz.open(stream=BytesIO(bytes(file_bytes)), filetype="pdf")
            for page_num, page in enumerate(doc):
                # First, try direct text extraction
                page_text = page.get_text()
                
                # If direct extraction yields little text, fall back to OCR
                if len(page_text.strip()) < 50: # Heuristic threshold
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img)
                    if self.ai_reviewer:
                        page_text = self.ai_reviewer.run_ai_ocr_correction(ocr_text)
                    else:
                        page_text = ocr_text
                
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # QR Code decoding
                pix = page.get_pixmap(dpi=300)
                img = Image.open(BytesIO(pix.tobytes("png")))
                qr_data.extend(obj.data.decode('utf-8') for obj in qr_decode(img))
            doc.close()
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except Exception as e:
            return {'success': False, 'error': f"PDF processing failed: {e}"}

    def process_files(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Processes all files, including brand compliance checks."""
        processed_docs = []
        all_skus = set()

        for file in self.files:
            file_name, file_bytes = file['name'], file['bytes']
            doc_type, file_nature = self._classify_document(file_name)
            doc_data = {
                'filename': file_name, 'text': "", 'doc_type': doc_type, 'file_nature': file_nature,
                'qr_codes': [], 'dimensions': [], 'brand_compliance': {}
            }

            if file_name.lower().endswith('.pdf'):
                pdf_result = self._extract_from_pdf(file_bytes)
                if pdf_result['success']:
                    doc_data.update({'text': pdf_result['text'], 'qr_codes': pdf_result['qr_data']})
                else:
                    st.warning(f"Processing failed for '{file_name}': {pdf_result['error']}")
                doc_data['brand_compliance'] = self._analyze_pdf_brand_compliance(file_bytes)
            
            elif file_name.lower().endswith(('.csv', '.xlsx')):
                try:
                    df = pd.read_excel(BytesIO(bytes(file_bytes)), header=None) if file_name.lower().endswith('.xlsx') else pd.read_csv(BytesIO(bytes(file_bytes)), header=None)
                    doc_data['text'] = df.to_string(index=False)
                except Exception as e:
                    st.warning(f"Could not read spreadsheet '{file_name}': {e}")
            
            skus_found = re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', doc_data['text'], re.IGNORECASE)
            all_skus.update(sku.upper() for sku in skus_found)

            processed_docs.append(doc_data)

        return processed_docs, sorted(list(filter(None, all_skus)))
