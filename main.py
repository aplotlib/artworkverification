import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image, ImageFile
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime
import json
from typing import Dict, List, Tuple
from contextlib import contextmanager
import time
import psutil
from pyzbar.pyzbar import decode as qr_decode
from collections import Counter
import gc
import os
import sys

# Allows loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Tesseract Configuration (Uncomment if needed for deployment) ---
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('artwork_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Vive Health Artwork Verification",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DEMO DATA LOADER ---
@st.cache_data
def load_demo_files():
    """Loads the provided demo files into memory."""
    demo_files = {}
    file_paths = [
        "wheelchair_bag_advanced_black_240625.pdf", "wheelchair_bag_advanced_purple_floral_240625.pdf",
        "wheelchair_bag_advanced_quickstart_020625.pdf", "wheelchair_bag_black_shipping_mark.pdf",
        "wheelchair_bag_logo_40mm_embroidered.pdf", "wheelchair_bag_purple_flower_shipping_mark.pdf",
        "wheelchair_bag_tag_black_250625.pdf", "wheelchair_bag_tag_purple_250625.pdf",
        "wheelchair_bag_washtag.pdf", "Wheelchair Bag Advanced 020625.xlsx - Black.csv",
        "Wheelchair Bag Advanced 020625.xlsx - Purple Floral.csv"
    ]
    for filename in file_paths:
        try:
            with open(filename, "rb") as f:
                file_bytes = f.read()
                mime_type = "application/pdf" if filename.endswith(".pdf") else "text/csv"
                demo_files[filename] = {"buffer": BytesIO(file_bytes), "name": filename, "type": mime_type}
        except FileNotFoundError:
            st.error(f"Demo file not found: {filename}. Please ensure it's in the same directory.")
            return None
    return list(demo_files.values())


# --- CORE CLASSES (MemoryManager, SessionStateManager, etc.) ---
class MemoryManager:
    """Manages memory usage and cleanup."""
    @staticmethod
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    @staticmethod
    def cleanup():
        gc.collect()

class SessionStateManager:
    """Manages session state."""
    @staticmethod
    def initialize():
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.validation_complete = False
            st.session_state.documents = {}
            st.session_state.validation_results = []
            st.session_state.review_states = {}

# --- CONSTANTS ---
DOC_TYPE_CONFIG = {
    'packaging_artwork': {'keywords': ['packaging', 'box', '_black_240625', '_purple_floral_240625'], 'required': True, 'description': 'Main product packaging artwork'},
    'manual': {'keywords': ['manual', 'instructions', 'guide', 'qsg', 'quickstart'], 'required': False, 'description': 'Product manual or quick start guide'},
    'washtag': {'keywords': ['washtag', 'wash tag', 'care'], 'required': False, 'description': 'Washtag with care instructions'},
    'shipping_mark': {'keywords': ['shipping', 'mark', 'carton'], 'required': True, 'description': 'Shipping mark with SKU and quantity'},
    'qc_sheet': {'keywords': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'], 'required': True, 'description': 'QC sheet with specifications'},
    'logo_tag': {'keywords': ['logo', 'tag'], 'required': False, 'description': 'Logo Tag'},
}

# --- CONTEXT MANAGERS for safe file handling ---
@contextmanager
def managed_pdf_document(file_buffer):
    pdf_document = None
    try:
        file_buffer.seek(0)
        pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
        yield pdf_document
    finally:
        if pdf_document:
            pdf_document.close()
        MemoryManager.cleanup()

@contextmanager
def managed_image(file_buffer):
    image = None
    try:
        image = Image.open(file_buffer)
        yield image
    finally:
        if image:
            image.close()
        MemoryManager.cleanup()

# --- FILE PROCESSING ---
class DocumentExtractor:
    """Handles text, QR code, and dimension extraction from files."""
    @staticmethod
    def _decode_qr_codes(image: Image.Image) -> List[str]:
        try:
            return [obj.data.decode('utf-8') for obj in qr_decode(image)]
        except Exception:
            return []

    @staticmethod
    def extract(file_buffer, file_type, filename):
        if file_type == "application/pdf":
            return DocumentExtractor._extract_from_pdf(file_buffer, filename)
        elif file_type in ["image/png", "image/jpeg"]:
            return DocumentExtractor._extract_from_image(file_buffer, filename)
        elif file_type in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            return DocumentExtractor._extract_from_excel(file_buffer, filename)
        return {'success': False, 'error': 'Unsupported file type'}

    @staticmethod
    def _extract_from_pdf(file_buffer, filename):
        text, images, qr_data, page_dims = "", [], [], "N/A"
        try:
            with managed_pdf_document(file_buffer) as pdf_doc:
                if len(pdf_doc) > 0:
                    rect = pdf_doc[0].rect
                    page_dims = f"{rect.width/72:.2f} x {rect.height/72:.2f} inches"
                for page in pdf_doc:
                    text += page.get_text()
                    pix = page.get_pixmap(dpi=150)
                    with managed_image(BytesIO(pix.tobytes("png"))) as img:
                        qr_data.extend(DocumentExtractor._decode_qr_codes(img))
                        if len(images) < 3: images.append(img.copy())
            return {'success': True, 'text': text, 'images': images, 'qr_data': qr_data, 'dimensions': page_dims}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _extract_from_image(file_buffer, filename):
        try:
            with managed_image(file_buffer) as img:
                dims = f"{img.width} x {img.height} pixels"
                text = pytesseract.image_to_string(img)
                qr_data = DocumentExtractor._decode_qr_codes(img)
            return {'success': True, 'text': text, 'images': [img.copy()], 'qr_data': qr_data, 'dimensions': dims}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _extract_from_excel(file_buffer, filename):
        try:
            df = pd.read_csv(file_buffer) if filename.endswith('.csv') else pd.read_excel(file_buffer)
            text = df.to_string()
            return {'success': True, 'text': text, 'images': [], 'qr_data': [], 'dimensions': f"{df.shape[1]} columns x {df.shape[0]} rows"}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class DocumentClassifier:
    """Classifies documents based on filename and content."""
    @staticmethod
    def classify(filename, text):
        for doc_type, config in DOC_TYPE_CONFIG.items():
            if any(kw in filename.lower() for kw in config['keywords']):
                return doc_type
        if 'distributed by' in text.lower(): return 'packaging_artwork'
        return 'unknown'

# --- VALIDATION LOGIC ---
class ArtworkValidator:
    """Contains all validation rules for artwork components."""

    def validate_packaging_artwork(self, text, filename):
        results = []
        if 'made in china' not in text.lower() and 'made in taiwan' not in text.lower():
            results.append(('failed', f'Missing country of origin on packaging: {filename}', 'origin_missing'))
        else:
            results.append(('passed', f'Country of origin present on packaging: {filename}', 'origin_ok'))
        return results

    def validate_washtag(self, text, filename):
        results = []
        if 'made in china' not in text.lower():
            results.append(('failed', f'Missing "Made in China" on washtag: {filename}', 'washtag_origin_missing'))
        if not any(kw in text.lower() for kw in ['machine wash', 'do not bleach']):
            results.append(('warning', f'Care instructions may be incomplete on washtag: {filename}', 'washtag_care_incomplete'))
        return results

    def cross_validate_serials(self, documents):
        """
        New and improved serial number validation logic.
        - Extracts all 12-digit UPCs and UDIs from all documents.
        - For each UPC, it checks for a partial match within the found UDIs.
        - It also validates that the UPC is present in the QR code data of the packaging artwork.
        """
        results = []
        all_text = " ".join([d.get('text', '') for d in documents.values()])
        all_qr_data = []
        for doc in documents.values():
            if doc.get('doc_type') == 'packaging_artwork':
                all_qr_data.extend(doc.get('qr_data', []))

        # Regex to find 12-digit UPCs and UDIs in the format (01)xxxxxxxxxxxxxx
        upcs = set(re.findall(r'\b(\d{12})\b', all_text))
        udis = set(re.findall(r'\(01\)(\d{14})', all_text))

        if not upcs:
            results.append(('failed', 'No 12-digit UPC serial numbers found in any document.', 'no_upcs_found', 'general'))
            return results

        for upc in upcs:
            # Check 1: UPC and UDI on box should have a partial EXACT match of 12 digit serials.
            udis_with_upc = [udi for udi in udis if upc in udi]
            if not udis_with_upc:
                results.append(('failed', f'UPC {upc} not found in any UDI.', 'upc_udi_mismatch', 'general'))
            else:
                results.append(('passed', f'UPC {upc} has a matching UDI.', 'upc_udi_match', 'general'))

            # Check 2: The Product QR code should have the 12digit UPC serial as well.
            if not any(upc in qr for qr in all_qr_data):
                results.append(('failed', f'UPC {upc} not found in any packaging QR code.', 'upc_missing_in_qr', 'general'))
            else:
                results.append(('passed', f'UPC {upc} found in packaging QR code.', 'upc_in_qr_ok', 'general'))

        return results

    def generate_serials_summary(self, documents):
        """
        Generates a DataFrame summarizing the found UPCs and UDIs.
        This is the foundation for the summary table you requested.
        We can expand this to include SKU and Size information later.
        """
        all_text = " ".join([d.get('text', '') for d in documents.values()])
        upcs = set(re.findall(r'\b(\d{12})\b', all_text))
        udis = set(re.findall(r'\(01\)(\d{14})', all_text))
        
        rows = []
        for upc in upcs:
            matching_udi = next((udi for udi in udis if upc in udi), "Not Found")
            rows.append({
                "UPC Digits": upc,
                "UDI Code": matching_udi
            })
        
        return pd.DataFrame(rows)


    def validate_all(self, documents, product_info):
        all_results = []
        for doc_type, data in documents.items():
            text, filename = data.get('text', ''), data.get('filename', '')
            if doc_type == 'packaging_artwork':
                all_results.extend(self.validate_packaging_artwork(text, filename))
            elif doc_type == 'washtag':
                all_results.extend(self.validate_washtag(text, filename))

        all_results.extend(self.cross_validate_serials(documents))
        return [(*r, doc_type) for r in all_results for doc_type in [r[2].split('_')[0]] if r]


# --- UI COMPONENTS ---
def display_header():
    """Displays the main application header."""
    st.markdown("""
    <style>
        .main-header { padding: 2rem; background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%); border-radius: 10px; text-align: center; margin-bottom: 2rem; color: white; }
        .success-box, .error-box, .warning-box, .info-box { padding: 1rem; border-left: 4px solid; margin: 0.5rem 0; }
        .success-box { background-color: #d1fae5; border-color: #10b981; }
        .error-box { background-color: #fee2e2; border-color: #ef4444; }
        .warning-box { background-color: #fef3c7; border-color: #f59e0b; }
        .info-box { background-color: #dbeafe; border-color: #3b82f6; }
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Artwork Verification System</h1>
        <p>Automated validation with QR code analysis, dimension checking, and interactive review.</p>
    </div>
    """, unsafe_allow_html=True)

def display_validation_result_with_review(result):
    """Displays a single validation result with interactive review widgets."""
    status, message, result_id, doc_type = result
    
    box_class = {'passed': 'success-box', 'failed': 'error-box', 'warning': 'warning-box'}.get(status, 'info-box')
    icon = {'passed': '✅', 'failed': '❌', 'warning': '⚠️'}.get(status, 'ℹ️')
    st.markdown(f'<div class="{box_class}">{icon} {message}</div>', unsafe_allow_html=True)
    
    if status in ['failed', 'warning']:
        if result_id not in st.session_state.review_states:
            st.session_state.review_states[result_id] = {'reviewed': False}
        
        state = st.session_state.review_states[result_id]
        cols = st.columns([1, 2, 3])
        state['reviewed'] = cols[0].checkbox("Reviewed", value=state['reviewed'], key=f"rev_{result_id}")
        if state['reviewed']:
            state['accurate'] = cols[1].radio("Finding accurate?", ["Yes", "No"], key=f"acc_{result_id}", horizontal=True, index=0 if state.get('accurate') == "Yes" else 1 if state.get('accurate') == "No" else None)
            if state.get('accurate') == 'Yes':
                state['action'] = cols[2].radio("Action Taken", ["Fixed", "Will Fix", "No Action"], key=f"act_{result_id}", horizontal=True, index=["Fixed", "Will Fix", "No Action"].index(state['action']) if state.get('action') in ["Fixed", "Will Fix", "No Action"] else None)

def generate_report_df(results, documents, product_info):
    """Generates a DataFrame for CSV/JSON export."""
    rows = []
    for status, message, result_id, doc_type in results:
        review_state = st.session_state.review_states.get(result_id, {})
        rows.append({
            'Status': status.upper(), 'Message': message, 'Document Type': doc_type,
            'Product': product_info.get('product_name'), 'SKU': product_info.get('sku'),
            'Reviewed': 'Yes' if review_state.get('reviewed') else 'No',
            'Finding Accurate': review_state.get('accurate'), 'Action Taken': review_state.get('action'),
            'Timestamp': datetime.now().isoformat()
        })
    return pd.DataFrame(rows)

# --- MAIN APPLICATION LOGIC ---
def main():
    SessionStateManager.initialize()
    display_header()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📋 Product Information")
        product_info = {
            'product_name': st.text_input("Product Name", "Wheelchair Bag Advanced", help="Product name for validation."),
            'sku': st.text_input("SKU", "LVA2037", help="Product SKU for validation.")
        }
        st.markdown("---")
        st.header("🚀 Actions")
        if st.button("Load Demo Data", help="Load pre-selected files to demonstrate functionality."):
            st.session_state.demo_files = load_demo_files()
            # Clear previous uploads if demo is loaded
            if 'uploaded_files' in st.session_state:
                del st.session_state['uploaded_files']
        
        if st.button("Clear All & Reset", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- FILE UPLOAD & PROCESSING ---
    files_to_process = []
    # Decide whether to use uploaded files or demo files
    if st.session_state.get('demo_files'):
        files_to_process = st.session_state.demo_files
        st.info("Showing Demo Data. To upload your own, click 'Clear All & Reset'.")
    else:
        uploaded_files = st.file_uploader(
            "Upload all artwork files (PDF, images, Excel/CSV)",
            type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls', 'csv'],
            accept_multiple_files=True
        )
        if uploaded_files:
            files_to_process = [{"buffer": file, "name": file.name, "type": file.type} for file in uploaded_files]

    if files_to_process and not st.session_state.documents:
        with st.spinner("Processing files... This may take a moment."):
            for file_data in files_to_process:
                extraction = DocumentExtractor.extract(file_data["buffer"], file_data["type"], file_data["name"])
                if extraction['success']:
                    doc_type = DocumentClassifier.classify(file_data["name"], extraction.get('text', ''))
                    st.session_state.documents[file_data["name"]] = {'doc_type': doc_type, 'filename': file_data["name"], **extraction}
                else:
                    st.error(f"Failed to process {file_data['name']}: {extraction.get('error')}")
        st.success(f"Processed {len(files_to_process)} files.")
        st.rerun()

    # --- VALIDATION & RESULTS DISPLAY ---
    if st.session_state.documents:
        if st.button("🔍 Run Validation", type="primary"):
            with st.spinner("Validating artwork..."):
                validator = ArtworkValidator()
                st.session_state.validation_results = validator.validate_all(st.session_state.documents, product_info)
                st.session_state.serials_summary = validator.generate_serials_summary(st.session_state.documents)
                st.session_state.validation_complete = True
        
        if st.session_state.validation_complete:
            st.header("📊 Summary Reports")
            with st.expander("Artwork Dimensions Summary", expanded=True):
                dims_data = [{'Filename': doc['filename'], 'Dimensions': doc.get('dimensions', 'N/A')} for doc in st.session_state.documents.values()]
                st.dataframe(pd.DataFrame(dims_data), use_container_width=True)
            
            # New Serials Summary Report
            if 'serials_summary' in st.session_state and not st.session_state.serials_summary.empty:
                with st.expander("Serials Summary", expanded=True):
                    st.dataframe(st.session_state.serials_summary, use_container_width=True)

            st.header("📋 Detailed Validation Results & Review")
            for result in sorted(st.session_state.validation_results, key=lambda x: {'failed': 0, 'warning': 1, 'info': 2, 'passed': 3}[x[0]]):
                display_validation_result_with_review(result)
            
            # --- EXPORT ---
            st.header("📥 Export Report")
            report_df = generate_report_df(st.session_state.validation_results, st.session_state.documents, product_info)
            st.download_button(
                label="Download Report as CSV",
                data=report_df.to_csv(index=False).encode('utf-8'),
                file_name=f"artwork_validation_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
            
    # --- DOCUMENT PREVIEWER ---
    if st.session_state.documents:
        st.header("📄 Document Preview")
        doc_name = st.selectbox("Select document to preview", list(st.session_state.documents.keys()))
        if doc_name:
            doc = st.session_state.documents[doc_name]
            with st.expander(f"View details for {doc_name}", expanded=False):
                st.text_area("Extracted Text", doc['text'], height=300)
                if doc.get('images'):
                    st.image(doc['images'], caption=[f"Image {i+1}" for i in range(len(doc['images']))], width=150)

if __name__ == "__main__":
    main()
