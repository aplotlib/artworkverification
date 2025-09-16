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
from colormath.color_objects import sRGBColor
from colormath.color_diff import delta_e_cie2000
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True

@st.cache_data
def process_files_cached(files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """A cached wrapper around the main file processing logic."""
    processor = DocumentProcessor(files)
    return processor.process_files()

class DocumentProcessor:
    """Handles all file reading, text extraction, and brand compliance analysis."""

    def __init__(self, files: List[Dict[str, Any]]):
        self.files = [self._freeze_file(f) for f in files]
        self._setup_brand_colors()

    def _freeze_file(self, file_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts file bytes to a hashable tuple for caching."""
        return {"name": file_dict["name"], "bytes": tuple(file_dict["bytes"])}

    def _setup_brand_colors(self):
        """Pre-calculates LabColor objects for brand colors for efficient comparison."""
        self.brand_colors_lab = []
        main_color_rgb = AppConfig.BRAND_GUIDE['colors']['brand_color']['rgb']
        self.brand_colors_lab.append({
            "name": AppConfig.BRAND_GUIDE['colors']['brand_color']['name'],
            "lab": sRGBColor(rgb_r=main_color_rgb[0], rgb_g=main_color_rgb[1], rgb_b=main_color_rgb[2], is_upscaled=True).convert_to('lab')
        })
        for color in AppConfig.BRAND_GUIDE['colors']['complementary_colors']:
            rgb = color['rgb']
            self.brand_colors_lab.append({
                "name": color['name'],
                "lab": sRGBColor(rgb_r=rgb[0], rgb_g=rgb[1], rgb_b=rgb[2], is_upscaled=True).convert_to('lab')
            })

    def _get_closest_brand_color(self, rgb_tuple: Tuple[int, int, int]) -> Tuple[str, float]:
        """Finds the closest brand color to a given RGB value using Delta E."""
        if not rgb_tuple or len(rgb_tuple) < 3: return "N/A", float('inf')
        color_to_check_lab = sRGBColor(rgb_r=rgb_tuple[0], rgb_g=rgb_tuple[1], rgb_b=rgb_tuple[2], is_upscaled=True).convert_to('lab')
        
        min_delta_e = float('inf')
        closest_color_name = "Non-Brand Color"

        for brand_color in self.brand_colors_lab:
            delta_e = delta_e_cie2000(color_to_check_lab, brand_color["lab"])
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                closest_color_name = brand_color["name"]
        
        return closest_color_name, min_delta_e

    def _analyze_pdf_brand_compliance(self, file_bytes: tuple) -> Dict[str, Any]:
        """Extracts fonts and colors from a PDF for brand compliance checking."""
        fonts, colors = set(), defaultdict(int)
        try:
            with fitz.open(stream=BytesIO(bytes(file_bytes)), filetype="pdf") as doc:
                for page in doc:
                    fonts.update(font[3] for font in page.get_fonts())
                    for drawing in page.get_drawings():
                        rgb = tuple(int(c * 255) for c in drawing.get("color", [])[:3])
                        if rgb: colors[rgb] += 1
                
                color_analysis = []
                for rgb, count in colors.items():
                    closest_name, delta_e = self._get_closest_brand_color(rgb)
                    is_compliant = delta_e <= AppConfig.BRAND_GUIDE["color_tolerance"]
                    color_analysis.append({
                        "rgb": rgb, "closest_brand_color": closest_name,
                        "delta_e": delta_e, "compliant": is_compliant
                    })

            return {"success": True, "fonts": list(fonts), "colors": color_analysis}
        except Exception as e:
            return {"success": False, "error": f"Brand compliance analysis failed: {e}"}

    def _extract_from_pdf_ocr(self, file_bytes: tuple) -> Dict[str, Any]:
        """Extracts text and QR codes from a PDF using OCR."""
        text, qr_data = "", []
        try:
            with fitz.open(stream=BytesIO(bytes(file_bytes)), filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=300)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        text += f"\n--- Page {page_num + 1} ---\n{pytesseract.image_to_string(img)}"
                        qr_data.extend(obj.data.decode('utf-8') for obj in qr_decode(img))
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except Exception as e:
            return {'success': False, 'error': f"OCR extraction failed: {e}"}

    def process_files(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Processes all files, now including brand compliance checks."""
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
                ocr_result = self._extract_from_pdf_ocr(file_bytes)
                if ocr_result['success']:
                    doc_data['text'] = ocr_result['text']
                    doc_data['qr_codes'] = ocr_result['qr_data']
                else:
                    st.warning(f"OCR failed for '{file_name}': {ocr_result['error']}")

                brand_result = self._analyze_pdf_brand_compliance(file_bytes)
                doc_data['brand_compliance'] = brand_result
            
            elif file_name.lower().endswith(('.csv', '.xlsx')):
                try:
                    df = pd.read_excel(BytesIO(bytes(file_bytes)), header=None).fillna('') if file_name.lower().endswith('.xlsx') else pd.read_csv(BytesIO(bytes(file_bytes)), header=None).fillna('')
                    doc_data['text'] = df.to_string()
                    doc_data['brand_compliance'] = {"success": True, "fonts": ["Arial"], "colors": []} 
                except Exception as e:
                    st.warning(f"Could not read spreadsheet '{file_name}': {e}")
            
            doc_data['dimensions'] = re.findall(r'(\d+\.?\d*mm\s?\(?[W|H|D]\)?\s?x\s?\d+\.?\d*mm\s?\(?[W|H|D]\)?)', doc_data['text'])
            skus_found = re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', doc_data['text'], re.IGNORECASE)
            for sku in skus_found:
                all_skus.add(sku.upper())

            processed_docs.append(doc_data)

        return processed_docs, list(filter(None, all_skus))
