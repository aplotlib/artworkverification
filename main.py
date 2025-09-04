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
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.openai_client = openai.OpenAI(api_key=self.api_keys.get('openai')) if 'openai' in self.api_keys else None
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_keys.get('anthropic')) if 'anthropic' in self.api_keys else None

    def _get_openai_summary(self, prompt):
        if not self.openai_client: return "OpenAI API key not found."
        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1)
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"

    def _get_anthropic_summary(self, prompt):
        if not self.anthropic_client: return "Anthropic API key not found."
        try:
            response = self.anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API Error: {e}"

    def generate_summary(self, provider, variant_sku, text_bundle, custom_instructions):
        instruction_prompt = f"In addition, the user has provided the following special instructions: '{custom_instructions}'. Please incorporate these into your review." if custom_instructions else ""
        prompt = f"""
        You are a meticulous quality assurance specialist for Vive Health. Review the extracted text from artwork files for product variant {variant_sku} and provide a concise summary.
        Key Information to check: Product Name, SKU, UPC, and UDI. Check for consistency and flag any issues like missing "Made in China" text.
        {instruction_prompt}
        Present findings as a brief, bulleted list. Start with "### AI Review Summary for {variant_sku}".
        Here is the combined text:
        ---
        {text_bundle}
        ---
        """
        if provider == 'openai': return self._get_openai_summary(prompt)
        if provider == 'anthropic': return self._get_anthropic_summary(prompt)
        if provider == 'both':
            openai_summary = self._get_openai_summary(prompt)
            anthropic_summary = self._get_anthropic_summary(prompt)
            return f"""<div style="display: flex; gap: 20px;"><div style="flex: 1; border: 1px solid #ddd; border-radius: 5px; padding: 10px;"><h4>OpenAI (GPT-4o Mini) Review</h4>{openai_summary}</div><div style="flex: 1; border: 1px solid #ddd; border-radius: 5px; padding: 10px;"><h4>Anthropic (Claude Haiku) Review</h4>{anthropic_summary}</div></div>"""
        return "Error: Invalid AI provider selected."

# --- File Processing and Extraction (with Enhanced QC Parsing) ---
@contextmanager
def managed_pdf_document(file_buffer):
    pdf_doc = None
    try:
        file_buffer.seek(0); yield fitz.open(stream=file_buffer.read(), filetype="pdf")
    finally:
        if pdf_doc: pdf_doc.close()

class DocumentExtractor:
    @staticmethod
    def _decode_qr_codes(image: Image.Image):
        try: return [obj.data.decode('utf-8') for obj in qr_decode(image)]
        except: return []

    @staticmethod
    def _extract_from_pdf(file_buffer):
        text, qr_data, page_dims = "", [], "N/A"
        try:
            with managed_pdf_document(file_buffer) as pdf_doc:
                if len(pdf_doc) > 0:
                    page_dims = f"{pdf_doc[0].rect.width/72:.2f} x {pdf_doc[0].rect.height/72:.2f} inches"
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
            file_buffer.seek(0)
            content = file_buffer.read().decode('utf-8', errors='ignore')
            specs = {'text': content}
            
            # Use regex to find the first value after the label, ignoring commas
            sku_match = re.search(r'PRODUCT SKU CODE.*?,*([A-Z0-9]+)', content, re.IGNORECASE)
            if sku_match:
                specs['sku'] = sku_match.group(1).strip()

            color_match = re.search(r'PRODUCT COLOR.*?,*([A-Za-z\s]+)', content, re.IGNORECASE)
            if color_match:
                specs['color'] = color_match.group(1).strip()

            if 'sku' not in specs:
                return {'success': False, 'error': "Could not find 'PRODUCT SKU CODE' in QC sheet."}
                
            return {'success': True, **specs}
        except Exception as e:
            return {'success': False, 'error': f"Could not parse QC Sheet: {e}"}

    @staticmethod
    def extract(file_buffer, file_type, filename):
        clean_name = filename.lower()
        if 'qc' in clean_name or '.csv' in clean_name or '.xlsx' in clean_name:
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
                sku_str = str(sku).lower()
                color_str = str(variant_data.get('color')).lower() if variant_data.get('color') else ''
                filename_lower = file['name'].lower()
                if sku_str in filename_lower or (color_str and color_str in filename_lower):
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
            return [('failed', f'Missing "Made in China" on {doc_type_name}: {filename}', f'origin_missing_{filename}')]
        return []

    def _validate_reference_text(self, text_bundle, context):
        if self.reference_text and self.reference_text not in text_bundle:
            return [('failed', f"Mandatory text from reference file was NOT found for {context}.", f"ref_text_missing_{context}")]
        return []

    def _validate_serials(self, all_text, context):
        results, cleaned_text = [], all_text.replace(" ", "").replace("\n", "")
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))
        if not upcs:
            return [('failed', f"No 12-digit UPCs found for {context}", f"no_upc_{context}")]
        for upc in upcs:
            if not any(upc in udi for udi in udis):
                results.append(('failed', f"UPC {upc} has no matching UDI for {context}", f"mismatch_{upc}"))
            else:
                results.append(('passed', f"UPC {upc} has a matching UDI for {context}", f"match_{upc}"))
        return results

    def validate_all(self):
        all_results = []
        for sku, variant in self.product_data['variants'].items():
            variant_files = list(variant.get('files', {}).values())
            shared_files = list(self.product_data.get('shared_files', {}).values())
            all_text_bundle = " ".join(d.get('text', '') for d in variant_files + shared_files)
            
            variant_results = self._validate_serials(all_text_bundle, sku)
            variant_results.extend(self._validate_reference_text(all_text_bundle, sku))
            
            for res in variant_results:
                all_results.append((res[0], res[1], res[2], sku))

        for file_data in self.product_data['unassociated']:
            filename, text = file_data['filename'], file_data.get('text', '')
            unassociated_results = [('failed', f"File '{filename}' could not be associated with a variant and requires manual review.", f"unassociated_{filename}")]
            unassociated_results.extend(self._validate_origin(text, filename, 'unassociated file'))
            unassociated_results.extend(self._validate_reference_text(text, filename))
            for res in unassociated_results:
                all_results.append((res[0], res[1], res[2], 'Unassociated Files'))
        return all_results

# --- UI Components ---
def display_header():
    st.markdown("""<style>.main-header { ... } .success-box, .error-box, .warning-box { ... }</style><div class="main-header">...</div>""", unsafe_allow_html=True)

def display_results(results):
    results_by_context = {}
    for status, msg, _, context in results:
        if context not in results_by_context: results_by_context[context] = []
        results_by_context[context].append((status, msg))
    
    if 'Unassociated Files' in results_by_context:
        with st.expander("### ⚠️ Unassociated Files Requiring Manual Review", expanded=True):
            st.error("The following files could not be matched. They must be manually reviewed.")
            for status, msg in results_by_context['Unassociated Files']:
                st.markdown(f'<div class="error-box">❌ {msg}</div>', unsafe_allow_html=True)

    for context, context_results in results_by_context.items():
        if context == 'Unassociated Files': continue
        with st.expander(f"### Rule-Based Validation Results for: {context}", expanded=True):
            st.info("ℹ️ These are the primary, rule-based checks.")
            for status, msg in sorted(context_results, key=lambda x: {'failed': 0, 'warning': 1, 'passed': 2}.get(x[0], 99)):
                icon = {'passed': '✅', 'failed': '❌', 'warning': '⚠️'}.get(status)
                box_class = {'passed': 'success-box', 'failed': 'error-box', 'warning': 'warning-box'}.get(status)
                st.markdown(f'<div class="{box_class}">{icon} {msg}</div>', unsafe_allow_html=True)

def display_ai_review(ai_reviews):
    st.markdown("---"); st.header("🤖 AI-Powered Review")
    st.warning("⚠️ **AI Review is Experimental.** Always rely on the rule-based validation and perform a final human review.")
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
        ref_file = st.file_uploader("Reference Text File (.txt)", type=['txt'])
        custom_instructions = st.text_area("Custom Instructions for AI")
        st.markdown("---")
        st.header("🤖 AI-Powered Review (Optional)")
        enable_ai_review = st.toggle("Enable AI Review", value=False)
        ai_provider = None
        if enable_ai_review:
            provider_options = {"Select AI...": None}
            if 'openai' in api_keys: provider_options["OpenAI (GPT-4o Mini)"] = 'openai'
            if 'anthropic' in api_keys: provider_options["Anthropic (Claude Haiku)"] = 'anthropic'
            if 'openai' in api_keys and 'anthropic' in api_keys:
                provider_options["Both (Compare Results)"] = 'both'
            if len(provider_options) > 1:
                selected_provider = st.selectbox("Choose AI Provider", options=list(provider_options.keys()))
                ai_provider = provider_options[selected_provider]
            else: st.error("No API keys found in secrets.")
        if st.button("Clear & Reset"):
            st.session_state.clear(); st.rerun()

    st.markdown("### Manual Checkpoints")
    st.info("Please manually double-check these key areas in your artwork. This tool uses OCR and AI APIs to review the files, no tool is 100%, no human is 100%, please use as an additional layer of review, but not the only layer. Thank you!")
    cols = st.columns(2)
    cols[0].checkbox("Is the **Country of Origin** correct?", key="check1")
    cols[0].checkbox("Does the **Logo Color** match the QC sheet?", key="check2")
    cols[1].checkbox("Are all **QR codes** present and correct?", key="check3")
    cols[1].checkbox("Do **dimensions** on tags match specs?", key="check4")
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload all artwork & QC files", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if run_validation and uploaded_files:
        files_to_process = [{"buffer": file, "name": file.name, "type": file.type} for file in uploaded_files]
        reference_text = ref_file.read().decode("utf-8").strip() if ref_file else None
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
            with st.spinner(f"Sending data to AI for optional review..."):
                reviewer = AIReviewer(api_keys)
                for sku, variant in product_data.get('variants', {}).items():
                    all_files = list(variant.get('files', {}).values()) + list(product_data.get('shared_files', {}).values())
                    text_bundle = "\n\n---\n\n".join([f"Source: {d.get('filename', 'N/A')}\n{d.get('text', '')}" for d in all_files])
                    summary = reviewer.generate_summary(ai_provider, sku, text_bundle, custom_instructions)
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
