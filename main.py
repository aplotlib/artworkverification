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
DOC_TYPE_MAP = {
    'packaging_artwork': ['packaging', 'box'],
    'manual': ['manual', 'instructions', 'guide', 'qsg', 'quickstart'],
    'washtag': ['washtag', 'wash tag', 'care'],
    'shipping_mark': ['shipping', 'mark', 'carton'],
    'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
    'logo_tag': ['tag']
}

# --- AI Integration ---
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

    def generate_summary(self, variant_sku, text_bundle, custom_instructions):
        instruction_prompt = ""
        if custom_instructions:
            instruction_prompt = f"""
            In addition to your standard checks, the user has provided the following special instructions: '{custom_instructions}'. Please ensure your review addresses these points specifically.
            """

        prompt = f"""
        You are a meticulous quality assurance specialist for Vive Health. Review the extracted text from artwork files for product variant {variant_sku} and provide a concise summary.

        Here is the combined text from all relevant documents:
        ---
        {text_bundle}
        ---

        Your task:
        1.  **Identify Key Information**: Find the Product Name, SKU, UPC, and UDI.
        2.  **Check for Consistency**: State if this information is consistent across documents.
        3.  **Flag Potential Issues**: Mention any other issues like missing "Made in China" text or conflicting details.
        4.  **Follow Custom Instructions**: {instruction_prompt}

        Present your findings as a brief, bulleted list. Start with "### AI Review Summary for {variant_sku}".
        """
        try:
            if self.provider == 'openai' and 'openai' in self.api_keys:
                client = openai.OpenAI(api_key=self.api_keys['openai'])
                response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1)
                return response.choices[0].message.content
            elif self.provider == 'anthropic' and 'anthropic' in self.api_keys:
                client = anthropic.Anthropic(api_key=self.api_keys['anthropic'])
                response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                return response.content[0].text
            else:
                return f"Error: API key for {self.provider} not found."
        except Exception as e:
            return f"An error occurred during AI review: {str(e)}"

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
        try: return [obj.data.decode('utf-8') for obj in qr_decode(image)]
        except Exception: return []

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
        except Exception as e: return {'success': False, 'error': str(e)}

    @staticmethod
    def _extract_from_qc_sheet(file_buffer, filename):
        try:
            df = pd.read_csv(file_buffer, dtype=str).fillna('')
            specs = {'text': df.to_string()}
            main_product_row = df[df.iloc[:, 2] == 'Wheelchair Bag Advanced']
            if not main_product_row.empty:
                specs['sku'] = str(main_product_row.iloc[0, 5])
                specs['color'] = str(main_product_row.iloc[0, 8])
            return {'success': True, **specs}
        except Exception as e: return {'success': False, 'error': f"Could not parse QC Sheet: {e}"}

    @staticmethod
    def extract(file_buffer, file_type, filename):
        if 'qc' in filename.lower() or '.csv' in filename.lower():
            return DocumentExtractor._extract_from_qc_sheet(file_buffer, filename)
        elif file_type == "application/pdf":
            return DocumentExtractor._extract_from_pdf(file_buffer)
        return {'success': False, 'error': 'Unsupported file type'}

# --- Product Data Structure Builder ---
class ProductDataBuilder:
    def __init__(self, files):
        self.files = files
        self.product_data = {"variants": {}, "shared_files": {}, "unassociated": []}

    def _get_doc_type(self, filename):
        clean_name = filename.lower().replace('copy of ', '')
        for doc_type, keywords in DOC_TYPE_MAP.items():
            if any(kw in clean_name for kw in keywords): return doc_type
        return 'unknown'

    def build(self):
        qc_sheets = [f for f in self.files if self._get_doc_type(f['name']) == 'qc_sheet']
        for file in qc_sheets:
            extraction = DocumentExtractor.extract(file['buffer'], file['type'], file['name'])
            if extraction.get('success') and extraction.get('sku'):
                sku = extraction['sku']
                self.product_data['variants'][sku] = {"sku": sku, "color": extraction.get('color'), "files": {'qc_sheet': {'filename': file['name'], **extraction}}}

        other_files = [f for f in self.files if self._get_doc_type(f['name']) != 'qc_sheet']
        for file in other_files:
            doc_type = self._get_doc_type(file['name'])
            extraction = DocumentExtractor.extract(file['buffer'], file['type'], file['name'])
            if not extraction.get('success'): continue
            file_data = {'filename': file['name'], 'doc_type': doc_type, **extraction}
            if any(kw in file['name'].lower() for kw in SHARED_FILE_KEYWORDS):
                self.product_data['shared_files'][doc_type] = file_data
                continue
            associated = False
            for sku, variant_data in self.product_data['variants'].items():
                if str(sku).lower() in file['name'].lower() or (variant_data.get('color') and str(variant_data['color']).lower() in file['name'].lower()):
                    variant_data['files'][doc_type] = file_data
                    associated = True
                    break
            if not associated:
                self.product_data["unassociated"].append(file_data)
        return self.product_data

# --- Artwork Validator ---
class ArtworkValidator:
    def __init__(self, product_data, reference_text=None):
        self.product_data = product_data
        self.reference_text = reference_text

    def _validate_origin(self, text, filename, doc_type_name):
        if 'made in china' not in text.lower():
            return [('failed', f'Missing "Made in China" on {doc_type_name}: {filename}', f'{doc_type_name}_origin_missing')]
        return [('passed', f'"Made in China" present on {doc_type_name}: {filename}', f'{doc_type_name}_origin_ok')]

    def _validate_reference_text(self, text_bundle, context):
        if self.reference_text and self.reference_text not in text_bundle:
            return [('failed', f"Mandatory text from reference file was NOT found for {context}.", f"ref_text_missing_{context}")]
        elif self.reference_text:
            return [('passed', f"Mandatory text from reference file was found for {context}.", f"ref_text_ok_{context}")]
        return []

    def _validate_serials_for_variant(self, variant):
        # ... (serial validation remains the same)
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
        # Validate categorized variants
        for sku, variant in self.product_data['variants'].items():
            variant_results = []
            if 'packaging_artwork' not in variant['files']:
                 variant_results.append(('failed', f"Missing packaging artwork for {sku}", f"missing_pkg_{sku}"))
            
            variant_files = list(variant.get('files', {}).values())
            shared_files = list(self.product_data.get('shared_files', {}).values())
            all_text_bundle = " ".join(d.get('text', '') for d in variant_files + shared_files)
            
            variant_results.extend(self._validate_serials_for_variant(variant))
            variant_results.extend(self._validate_reference_text(all_text_bundle, sku))
            
            for res in variant_results:
                all_results.append((res[0], res[1], res[2], sku))

        # Flag unassociated files and run checks on them
        for file_data in self.product_data['unassociated']:
            filename = file_data['filename']
            unassociated_results = [('failed', f"File '{filename}' could not be associated with any product variant and requires manual review.", f"unassociated_{filename}")]
            text = file_data.get('text', '')
            unassociated_results.extend(self._validate_origin(text, filename, 'unassociated file'))
            unassociated_results.extend(self._validate_reference_text(text, filename))
            for res in unassociated_results:
                all_results.append((res[0], res[1], res[2], 'Unassociated Files'))

        return all_results

# --- UI Components ---
def display_header():
    st.markdown("""
    <style> .main-header { ... } .success-box, .error-box, .warning-box { ... } </style>
    <div class="main-header">
        <h1>✅ Vive Health Artwork Verification System</h1>
        <p>Rule-Based Validation with Optional AI-Powered Review</p>
    </div>
    """, unsafe_allow_html=True)

def display_results(results):
    results_by_context = {}
    for status, msg, res_id, context in results:
        if context not in results_by_context: results_by_context[context] = []
        results_by_context[context].append((status, msg))
    
    # Display unassociated files first for high visibility
    if 'Unassociated Files' in results_by_context:
        with st.expander("### ⚠️ Unassociated Files Requiring Manual Review", expanded=True):
            st.error("The following files could not be automatically matched to a product variant defined in the QC sheets. They must be manually reviewed.")
            for status, msg in results_by_context['Unassociated Files']:
                icon = '❌'
                st.markdown(f'<div class="error-box">{icon} {msg}</div>', unsafe_allow_html=True)

    for context, context_results in results_by_context.items():
        if context == 'Unassociated Files': continue
        with st.expander(f"### Rule-Based Validation Results for: {context}", expanded=True):
            st.info("ℹ️ These results are based on precise, rule-based checks and are the primary source of truth.")
            for status, msg in sorted(context_results, key=lambda x: {'failed': 0, 'warning': 1, 'passed': 2}.get(x[0], 99)):
                icon = {'passed': '✅', 'failed': '❌', 'warning': '⚠️'}.get(status)
                box_class = {'passed': 'success-box', 'failed': 'error-box', 'warning': 'warning-box'}.get(status)
                st.markdown(f'<div class="{box_class}">{icon} {msg}</div>', unsafe_allow_html=True)

def display_ai_review(ai_reviews):
    # ... (remains the same)
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

    if 'validation_complete' not in st.session_state: st.session_state.validation_complete = False

    with st.sidebar:
        st.header("🚀 Actions")
        run_validation = st.button("🔍 Run Validation", type="primary")
        
        st.markdown("---")
        st.header("📝 Custom Validation Rules (Optional)")
        ref_file = st.file_uploader("Reference Text File (.txt)", type=['txt'], help="Upload a text file with phrases that MUST appear in the artwork.")
        custom_instructions = st.text_area("Custom Instructions for AI", help="e.g., 'Check for a 1-year warranty statement.'")

        st.markdown("---")
        st.header("🤖 AI-Powered Review (Optional)")
        enable_ai_review = st.toggle("Enable AI Review", value=False)
        
        ai_provider = None
        if enable_ai_review:
            provider_options = {"Select AI...": None}
            if 'openai' in api_keys: provider_options["OpenAI (GPT-4o Mini)"] = 'openai'
            if 'anthropic' in api_keys: provider_options["Anthropic (Claude Haiku)"] = 'anthropic'
            if len(provider_options) > 1:
                selected_provider = st.selectbox("Choose AI Provider", options=list(provider_options.keys()))
                ai_provider = provider_options[selected_provider]
            else:
                st.error("No API keys found in secrets.")
        
        if st.button("Clear & Reset"):
            st.session_state.clear()
            st.rerun()

    uploaded_files = st.file_uploader("Upload all artwork & QC files", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if run_validation and uploaded_files:
        files_to_process = [{"buffer": file, "name": file.name, "type": file.type} for file in uploaded_files]
        
        reference_text = None
        if ref_file:
            reference_text = ref_file.read().decode("utf-8").strip()

        with st.spinner("Processing files and building product structure..."):
            builder = ProductDataBuilder(files_to_process)
            product_data = builder.build()
            st.session_state.product_data = product_data
        with st.spinner("Running rule-based validations..."):
            validator = ArtworkValidator(product_data, reference_text)
            st.session_state.results = validator.validate_all()
            st.session_state.validation_complete = True
            st.session_state.ai_reviews = []

        if enable_ai_review and ai_provider:
            with st.spinner(f"Sending data to {ai_provider} for optional review..."):
                reviewer = AIReviewer(ai_provider, api_keys)
                for sku, variant in product_data.get('variants', {}).items():
                    all_files = list(variant.get('files', {}).values()) + list(product_data.get('shared_files', {}).values())
                    text_bundle = "\n\n---\n\n".join([f"Source: {d.get('filename', 'N/A')}\n{d.get('text', '')}" for d in all_files])
                    summary = reviewer.generate_summary(sku, text_bundle, custom_instructions)
                    st.session_state.ai_reviews.append(summary)

    if st.session_state.validation_complete:
        st.header("📊 Automated Validation Report")
        display_results(st.session_state.results)
        
        if st.session_state.get('ai_reviews'):
            display_ai_review(st.session_state.ai_reviews)
            
        st.header("📄 Data Structure Preview")
        with st.expander("Click to see how files were grouped"):
            st.json(st.session_state.product_data, expanded=False)

if __name__ == "__main__":
    main()
