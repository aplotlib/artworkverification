import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image, ImageFile
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime
from typing import Dict, List, Any
from contextlib import contextmanager
import os
import sys
from pyzbar.pyzbar import decode as qr_decode
import openai
import anthropic

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="Vive Health Artwork Verification", page_icon="✅", layout="wide")

# --- Constants for Document Classification ---
SHARED_FILE_KEYWORDS = ['manual', 'instructions', 'guide', 'qsg', 'quickstart', 'washtag', 'wash tag', 'care', 'logo']
VARIANT_SPECIFIC_KEYWORDS = ['packaging', 'box', 'shipping', 'mark', 'carton', 'tag']
DOC_TYPE_MAP = {
    'packaging_artwork': ['packaging', 'box'],
    'manual': ['manual', 'instructions', 'guide', 'qsg', 'quickstart'],
    'washtag': ['washtag', 'wash tag', 'care'],
    'shipping_mark': ['shipping', 'mark', 'carton'],
    'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
    'logo_tag': ['tag'] 
}

# --- NEW: AI Integration ---
def check_api_keys():
    keys = {}
    if hasattr(st, 'secrets'):
        if 'openai_api_key' in st.secrets and st.secrets['openai_api_key']:
            keys['openai'] = st.secrets['openai_api_key']
        if 'anthropic_api_key' in st.secrets and st.secrets['anthropic_api_key']:
            keys['anthropic'] = st.secrets['anthropic_api_key']
    return keys

class AIReviewer:
    def __init__(self, provider, api_keys):
        self.provider = provider
        self.api_keys = api_keys

    def generate_summary(self, variant_sku, text_bundle):
        prompt = f"""
        You are a meticulous quality assurance specialist for a company named Vive Health. Your task is to review the extracted text from a set of artwork files for a specific product variant and provide a concise summary of your findings.

        Product Variant SKU: {variant_sku}

        Here is the combined text from all relevant documents (packaging, shipping marks, QC sheets, etc.):
        ---
        {text_bundle}
        ---

        Based on the text provided, please perform the following:
        1.  **Identify Key Information**: Find the primary Product Name, SKU, UPC (12-digit barcode number), and UDI (a longer number usually starting with '(01)').
        2.  **Check for Consistency**: State whether these key pieces of information appear to be consistent across the different documents.
        3.  **Flag Potential Issues**: Mention any other potential issues you see, such as missing "Made in China" text, conflicting product names, or formatting problems in the serial numbers.

        Present your findings as a brief, bulleted list. Be professional and objective. Start your response with "### AI Review Summary for {variant_sku}".
        """
        try:
            if self.provider == 'openai' and 'openai' in self.api_keys:
                client = openai.OpenAI(api_key=self.api_keys['openai'])
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return response.choices[0].message.content
            elif self.provider == 'anthropic' and 'anthropic' in self.api_keys:
                client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return response.content[0].text
            else:
                return f"Error: API key for {self.provider} not found or provider is not selected."
        except Exception as e:
            logger.error(f"AI review failed for {self.provider}: {e}")
            return f"An error occurred while generating the AI review: {str(e)}"

# --- File Processing and Extraction ---
@contextmanager
def managed_pdf_document(file_buffer):
    pdf_doc = None
    try:
        file_buffer.seek(0)
        pdf_doc = fitz.open(stream=file_buffer.read(), filetype="pdf")
        yield pdf_doc
    finally:
        if pdf_doc: pdf_doc.close()

class DocumentExtractor:
    @staticmethod
    def _decode_qr_codes(image: Image.Image) -> List[str]:
        try:
            return [obj.data.decode('utf-8') for obj in qr_decode(image)]
        except Exception:
            return []

    @staticmethod
    def _extract_from_pdf(file_buffer):
        text, qr_data, page_dims = "", [], "N/A"
        try:
            with managed_pdf_document(file_buffer) as pdf_doc:
                if len(pdf_doc) > 0:
                    rect = pdf_doc[0].rect
                    page_dims = f"{rect.width/72:.2f} x {rect.height/72:.2f} inches"
                for page in pdf_doc:
                    text += page.get_text()
                    pix = page.get_pixmap(dpi=200)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        qr_data.extend(DocumentExtractor._decode_qr_codes(img))
            return {'success': True, 'text': text, 'qr_data': qr_data, 'dimensions': page_dims}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _extract_from_qc_sheet(file_buffer, filename):
        try:
            df = pd.read_csv(file_buffer) if filename.endswith('.csv') else pd.read_excel(file_buffer)
            sku_row = df[df.iloc[:, 2] == 'Wheelchair Bag Advanced']
            sku = sku_row.iloc[0, 5] if not sku_row.empty else "SKU_NOT_FOUND"
            color = sku_row.iloc[0, 8] if not sku_row.empty else "COLOR_NOT_FOUND"
            return {'success': True, 'text': df.to_string(), 'sku': sku, 'color': color}
        except Exception as e:
            return {'success': False, 'error': f"Could not parse QC Sheet: {e}"}

    @staticmethod
    def extract(file_buffer, file_type, filename):
        if 'qc' in filename.lower() or '.csv' in filename.lower() or '.xlsx' in filename.lower():
            return DocumentExtractor._extract_from_qc_sheet(file_buffer, filename)
        elif file_type == "application/pdf":
            return DocumentExtractor._extract_from_pdf(file_buffer)
        return {'success': False, 'error': 'Unsupported file type'}

# --- Product Data Structure and Classifier ---
class ProductDataBuilder:
    def __init__(self, files):
        self.files = files
        self.product_data = {"variants": {}, "shared_files": {}}

    def _get_doc_type(self, filename):
        clean_name = filename.lower().replace('copy of ', '')
        for doc_type, keywords in DOC_TYPE_MAP.items():
            if any(kw in clean_name for kw in keywords):
                return doc_type
        return 'unknown'

    def build(self):
        for file in self.files:
            if self._get_doc_type(file['name']) == 'qc_sheet':
                extraction = DocumentExtractor.extract(file['buffer'], file['type'], file['name'])
                if extraction['success']:
                    sku = extraction['sku']
                    self.product_data['variants'][sku] = {"sku": sku, "color": extraction['color'], "files": {}}
        
        for file in self.files:
            doc_type = self._get_doc_type(file['name'])
            if doc_type == 'qc_sheet': continue
            extraction = DocumentExtractor.extract(file['buffer'], file['type'], file['name'])
            if not extraction['success']: continue
            file_data = {'filename': file['name'], 'doc_type': doc_type, **extraction}
            is_shared = any(kw in file['name'].lower() for kw in SHARED_FILE_KEYWORDS)
            if is_shared:
                self.product_data['shared_files'][doc_type] = file_data
                continue
            associated = False
            for sku, variant_data in self.product_data['variants'].items():
                if sku.lower() in file['name'].lower() or variant_data['color'].lower() in file['name'].lower():
                    variant_data['files'][doc_type] = file_data
                    associated = True
                    break
            if not associated:
                if "unassociated" not in self.product_data: self.product_data["unassociated"] = []
                self.product_data["unassociated"].append(file_data)
        
        return self.product_data

# --- Artwork Validator ---
class ArtworkValidator:
    def __init__(self, product_data):
        self.product_data = product_data

    def _validate_origin(self, text, filename, doc_type_name):
        if 'made in china' not in text.lower():
            return [('failed', f'Missing "Made in China" on {doc_type_name}: {filename}', f'{doc_type_name}_origin_missing')]
        return [('passed', f'"Made in China" present on {doc_type_name}: {filename}', f'{doc_type_name}_origin_ok')]

    def _validate_serials_for_variant(self, variant):
        results = []
        all_files = list(variant['files'].values()) + list(self.product_data['shared_files'].values())
        all_text = " ".join([d.get('text', '') for d in all_files])
        cleaned_text = all_text.replace(" ", "").replace("\n", "")
        packaging_file = variant['files'].get('packaging_artwork')
        qr_data = packaging_file.get('qr_data', []) if packaging_file else []
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))
        if not upcs:
            return [('failed', f"No 12-digit UPCs found for variant {variant['sku']}", f"no_upc_{variant['sku']}")]
        for upc in upcs:
            if not any(upc in udi for udi in udis):
                results.append(('failed', f"UPC {upc} has no matching UDI for variant {variant['sku']}", f"mismatch_{upc}"))
            else:
                results.append(('passed', f"UPC {upc} has a matching UDI for variant {variant['sku']}", f"match_{upc}"))
            if not any(upc in qr for qr in qr_data):
                results.append(('failed', f"UPC {upc} not found in packaging QR for variant {variant['sku']}", f"no_qr_{upc}"))
            else:
                results.append(('passed', f"UPC {upc} found in packaging QR for variant {variant['sku']}", f"qr_ok_{upc}"))
        return results

    def validate_all(self):
        all_results = []
        for sku, variant in self.product_data['variants'].items():
            variant_results = []
            if 'packaging_artwork' not in variant['files']:
                 variant_results.append(('failed', f"Missing packaging artwork for {sku}", f"missing_pkg_{sku}"))
            for doc_type, file_data in variant['files'].items():
                if doc_type == 'packaging_artwork':
                    variant_results.extend(self._validate_origin(file_data['text'], file_data['filename'], 'packaging'))
            variant_results.extend(self._validate_serials_for_variant(variant))
            for res in variant_results:
                all_results.append((res[0], res[1], res[2], sku))
        if 'washtag' in self.product_data['shared_files']:
            wt_file = self.product_data['shared_files']['washtag']
            for res in self._validate_origin(wt_file['text'], wt_file['filename'], 'washtag'):
                 all_results.append((res[0], res[1], res[2], 'Shared'))
        return all_results

# --- UI Components ---
def display_header():
    st.markdown("""
    <style>
        .main-header { padding: 1.5rem; background-color: #0891b2; border-radius: 10px; text-align: center; margin-bottom: 2rem; color: white; }
        .success-box, .error-box, .warning-box { padding: 1rem; margin: 0.5rem 0; border-radius: 5px; }
        .success-box { background-color: #d1fae5; border-left: 5px solid #10b981; }
        .error-box { background-color: #fee2e2; border-left: 5px solid #ef4444; }
        .warning-box { background-color: #fef3c7; border-left: 5px solid #f59e0b; }
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Artwork Verification System</h1>
        <p>Rule-Based Validation with Optional AI-Powered Review</p>
    </div>
    """, unsafe_allow_html=True)

def display_results(results):
    results_by_variant = {}
    for status, msg, res_id, variant_sku in results:
        if variant_sku not in results_by_variant: results_by_variant[variant_sku] = []
        results_by_variant[variant_sku].append((status, msg))
    for variant_sku, variant_results in results_by_variant.items():
        with st.expander(f"### Rule-Based Validation Results for: {variant_sku}", expanded=True):
            st.info("ℹ️ These results are based on precise, rule-based checks and should be considered the primary source of truth.")
            for status, msg in sorted(variant_results, key=lambda x: {'failed': 0, 'warning': 1, 'passed': 2}.get(x[0], 99)):
                icon = {'passed': '✅', 'failed': '❌', 'warning': '⚠️'}.get(status)
                box_class = {'passed': 'success-box', 'failed': 'error-box', 'warning': 'warning-box'}.get(status)
                st.markdown(f'<div class="{box_class}">{icon} {msg}</div>', unsafe_allow_html=True)

def display_ai_review(ai_reviews):
    st.markdown("---")
    st.header("🤖 AI-Powered Review")
    st.warning("""
    ⚠️ **AI Review is Experimental & For Reference Only.**
    This summary is generated by an AI and may contain inaccuracies. It is intended as a supplementary check to help spot potential issues.
    **Always rely on the rule-based validation results and perform a final human review.**
    """)
    for summary in ai_reviews:
        st.markdown(summary, unsafe_allow_html=True)

# --- Main App Logic ---
def main():
    display_header()
    api_keys = check_api_keys()

    if 'validation_complete' not in st.session_state:
        st.session_state.validation_complete = False
    if 'ai_reviews' not in st.session_state:
        st.session_state.ai_reviews = []

    with st.sidebar:
        st.header("🚀 Actions")
        use_demo = st.button("Load Demo Data")
        run_validation = st.button("🔍 Run Validation", type="primary")
        
        st.markdown("---")
        st.header("🤖 AI-Powered Review (Optional)")
        enable_ai_review = st.toggle("Enable AI Review", value=False, help="Use an AI to provide a summary of findings. Requires API keys.")
        
        ai_provider = None
        if enable_ai_review:
            provider_options = {"Select an AI...": None}
            if 'openai' in api_keys: provider_options["OpenAI (GPT-4o Mini)"] = 'openai'
            if 'anthropic' in api_keys: provider_options["Anthropic (Claude Haiku)"] = 'anthropic'
            
            if len(provider_options) > 1:
                selected_provider = st.selectbox("Choose AI Provider", options=list(provider_options.keys()))
                ai_provider = provider_options[selected_provider]
            else:
                st.error("No API keys found in secrets.toml. Please add `openai_api_key` or `anthropic_api_key`.")

        if st.button("Clear & Reset"):
            st.session_state.clear()
            st.rerun()

    files_to_process = []
    # Logic to load demo or uploaded files
    if use_demo:
        demo_files = load_demo_files()
        if demo_files: files_to_process = demo_files
    else:
        uploaded_files = st.file_uploader("Upload all artwork & QC files", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
        if uploaded_files: files_to_process = [{"buffer": file, "name": file.name, "type": file.type} for file in uploaded_files]

    if run_validation and files_to_process:
        with st.spinner("Building product data structure..."):
            builder = ProductDataBuilder(files_to_process)
            product_data = builder.build()
            st.session_state.product_data = product_data
        with st.spinner("Running rule-based validations..."):
            validator = ArtworkValidator(product_data)
            st.session_state.results = validator.validate_all()
            st.session_state.validation_complete = True
            st.session_state.ai_reviews = [] # Clear previous AI reviews

        if enable_ai_review and ai_provider:
            with st.spinner(f"Sending data to {ai_provider} for optional review..."):
                reviewer = AIReviewer(ai_provider, api_keys)
                for sku, variant in product_data.get('variants', {}).items():
                    all_files = list(variant.get('files', {}).values()) + list(product_data.get('shared_files', {}).values())
                    text_bundle = "\n\n---\n\n".join([f"Source File: {d.get('filename', 'N/A')}\n\n{d.get('text', '')}" for d in all_files])
                    summary = reviewer.generate_summary(sku, text_bundle)
                    st.session_state.ai_reviews.append(summary)

    if st.session_state.validation_complete:
        st.header("📊 Validation Report")
        display_results(st.session_state.results)
        
        if st.session_state.ai_reviews:
            display_ai_review(st.session_state.ai_reviews)
            
        st.header("📄 Data Structure Preview")
        with st.expander("Click to see how files were grouped"):
            preview_data = {}
            p_data = st.session_state.product_data
            for sku, variant in p_data.get('variants', {}).items():
                preview_data[sku] = [f.get('filename') for f in variant.get('files', {}).values()]
            preview_data["Shared Files"] = [f.get('filename') for f in p_data.get('shared_files', {}).values()]
            if p_data.get('unassociated'):
                 preview_data["Unassociated Files"] = [f.get('filename') for f in p_data.get('unassociated', [])]
            st.json(preview_data)

def load_demo_files():
    """Loads the provided demo files into memory."""
    # This function should be defined as before, with the correct file paths
    # For brevity, it is omitted here, but should be included in your final script.
    # Make sure the file paths are correct.
    demo_files = {}
    file_paths = [
        "wheelchair_bag_washtag.pdf",
        "wheelchair_bag_tag_purple_250625.pdf",
        "Wheelchair Bag Advanced 020625.xlsx - Purple Floral.csv",
        "Wheelchair Bag Advanced 020625.xlsx - Black.csv",
        "Copy of wheelchair_bag_advanced_purple_floral_240625.pdf",
        "Copy of wheelchair_bag_advanced_quickstart_020625.pdf",
        "Copy of Wheelchair_Bag_Black_Shipping_Mark.pdf",
        "Copy of wheelchair_bag_purple_flower_shipping_mark.pdf",
        "Copy of wheelchair_bag_tag_black_250625.pdf"
    ]
    for filename in file_paths:
        try:
            with open(filename, "rb") as f:
                file_bytes = f.read()
                mime_type = "application/pdf" if filename.endswith((".pdf", ".pdf")) else "text/csv"
                demo_files[filename] = {"buffer": BytesIO(file_bytes), "name": filename, "type": mime_type}
        except FileNotFoundError:
            st.error(f"Demo file not found: {filename}. Please ensure it's in the same directory.")
            return None
    return list(demo_files.values())


if __name__ == "__main__":
    main()
