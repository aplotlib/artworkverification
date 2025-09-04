import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image, ImageFile
import fitz  # PyMuPDF
from datetime import datetime
from typing import Dict, List, Any
from contextlib import contextmanager
from pyzbar.pyzbar import decode as qr_decode
import openai
import anthropic

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Artwork Verification Tool", page_icon="✅", layout="wide")

# --- Constants ---
SHARED_FILE_KEYWORDS = ['manual', 'qsg', 'quickstart', 'washtag', 'logo']
DOC_TYPE_MAP = {
    'packaging_artwork': ['packaging', 'box'],
    'manual': ['manual', 'qsg', 'quickstart'],
    'washtag': ['washtag', 'wash tag', 'care'],
    'shipping_mark': ['shipping', 'mark', 'carton'],
    'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
    'logo_tag': ['tag']
}

# --- AI Integration ---
def check_api_keys():
    keys = {}
    if hasattr(st, 'secrets'):
        for key in ['openai_api_key', 'OPENAI_API_KEY']:
            if key in st.secrets and st.secrets[key]:
                keys['openai'] = st.secrets[key]
        for key in ['anthropic_api_key', 'ANTHROPIC_API_KEY', 'claude_api_key']:
            if key in st.secrets and st.secrets[key]:
                keys['anthropic'] = st.secrets[key]
    return keys

class AIReviewer:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.openai_client = openai.OpenAI(api_key=self.api_keys.get('openai')) if 'openai' in api_keys else None
        self.anthropic_client = anthropic.Anthropic(api_key=self.api_keys.get('anthropic')) if 'anthropic' in api_keys else None

    def _get_summary(self, client_type, prompt):
        try:
            if client_type == 'openai' and self.openai_client:
                response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1)
                return response.choices[0].message.content
            elif client_type == 'anthropic' and self.anthropic_client:
                response = self.anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}], temperature=0.1)
                return response.content[0].text
            return f"{client_type.capitalize()} API key not found."
        except Exception as e:
            return f"{client_type.capitalize()} API Error: {e}"

    def generate_summary(self, provider, variant_sku, text_bundle, custom_instructions):
        instruction_prompt = f"Additionally, follow these instructions: '{custom_instructions}'" if custom_instructions else ""
        prompt = f"You are a QA specialist. Review the text from artwork files for product SKU {variant_sku}. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {instruction_prompt}. Present findings as a bulleted list. Start with '### AI Review for {variant_sku}'.\n\n---BEGIN DATA---\n{text_bundle}\n---END DATA---"
        
        if provider == 'both':
            openai_summary = self._get_summary('openai', prompt)
            anthropic_summary = self._get_summary('anthropic', prompt)
            return f"""<div style="display: flex; gap: 1rem;"><div style="flex: 1;"><h4>OpenAI Review</h4>{openai_summary}</div><div style="flex: 1;"><h4>Anthropic Review</h4>{anthropic_summary}</div></div>"""
        return self._get_summary(provider, prompt)

# --- File Processing ---
class DocumentExtractor:
    @staticmethod
    def extract_from_pdf(file_buffer):
        text, qr_data = "", []
        try:
            with fitz.open(stream=file_buffer.read(), filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()
                    pix = page.get_pixmap(dpi=200)
                    with Image.open(BytesIO(pix.tobytes("png"))) as img:
                        qr_data.extend(obj.data.decode('utf-8') for obj in qr_decode(img))
            return {'success': True, 'text': text, 'qr_data': qr_data}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def extract_from_qc_sheet(file_buffer):
        try:
            df = pd.read_csv(file_buffer, dtype=str, header=None).fillna('')
            specs = {}
            np_array = df.to_numpy()
            for row_idx, row in enumerate(np_array):
                for col_idx, cell in enumerate(row):
                    cell_str = str(cell).upper().strip()
                    if "PRODUCT SKU CODE" in cell_str:
                        specs['sku'] = np_array[row_idx, col_idx + 2]
                    if "PRODUCT COLOR" in cell_str:
                        specs['color'] = np_array[row_idx, col_idx + 2]
            if 'sku' not in specs:
                return {'success': False, 'error': "Could not find 'PRODUCT SKU CODE' in QC sheet."}
            return {'success': True, **specs, 'text': df.to_string()}
        except Exception as e:
            return {'success': False, 'error': f"Could not parse QC Sheet: {e}"}

# --- Core Logic ---
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
        for file in self.files:
            doc_type = self._get_doc_type(file['name'])
            if doc_type == 'qc_sheet':
                extraction = DocumentExtractor.extract_from_qc_sheet(file['buffer'])
                if extraction.get('success') and extraction.get('sku'):
                    sku = extraction['sku']
                    self.product_data['variants'][sku] = {"sku": sku, "color": extraction.get('color'), "files": {'qc_sheet': {'filename': file['name'], **extraction}}}
        
        for file in self.files:
            doc_type = self._get_doc_type(file['name'])
            if doc_type == 'qc_sheet': continue
            extraction = DocumentExtractor.extract_from_pdf(file['buffer'])
            if not extraction.get('success'): continue
            file_data = {'filename': file['name'], 'doc_type': doc_type, **extraction}
            
            if any(kw in file['name'].lower() for kw in SHARED_FILE_KEYWORDS):
                self.product_data['shared_files'][doc_type] = file_data
                continue

            associated = False
            for sku, variant_data in self.product_data['variants'].items():
                if str(sku).lower() in file['name'].lower() or (variant_data.get('color') and str(variant_data.get('color')).lower() in file['name'].lower()):
                    variant_data['files'][doc_type] = file_data
                    associated = True
                    break
            if not associated:
                self.product_data["unassociated"].append(file_data)
        return self.product_data

class ArtworkValidator:
    def __init__(self, product_data, reference_text=None):
        self.product_data = product_data
        self.reference_text = reference_text

    def _run_checks(self, text_bundle, context):
        results = []
        cleaned_text = text_bundle.replace(" ", "").replace("\n", "")
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))
        
        if not upcs:
            results.append(('failed', f"No 12-digit UPCs found for {context}", f"no_upc_{context}"))
        else:
            for upc in upcs:
                if not any(upc in udi for udi in udis):
                    results.append(('failed', f"UPC {upc} has no matching UDI for {context}", f"mismatch_{upc}"))
                else:
                    results.append(('passed', f"UPC {upc} has a matching UDI for {context}", f"match_{upc}"))
        
        if self.reference_text and self.reference_text not in text_bundle:
            results.append(('failed', f"Mandatory text from reference file was NOT found for {context}.", f"ref_text_missing_{context}"))
        
        return results

    def validate_all(self):
        all_results = []
        for sku, variant in self.product_data['variants'].items():
            all_files = list(variant.get('files', {}).values()) + list(self.product_data.get('shared_files', {}).values())
            text_bundle = " ".join(d.get('text', '') for d in all_files)
            for res in self._run_checks(text_bundle, sku):
                all_results.append((*res, sku))

        for file_data in self.product_data['unassociated']:
            filename = file_data['filename']
            all_results.append(('failed', f"File '{filename}' could not be associated with a variant.", f"unassociated_{filename}", "Unassociated Files"))
        
        return all_results

# --- UI Components ---
def display_results(results):
    results_by_context = {}
    for status, msg, _, context in results:
        if context not in results_by_context: results_by_context[context] = []
        results_by_context[context].append((status, msg))
    
    st.header("📊 Automated Validation Report")
    if 'Unassociated Files' in results_by_context:
        with st.expander("⚠️ Unassociated Files", expanded=True):
            st.error("These files could not be matched to a product variant and require manual review.")
            for _, msg in results_by_context['Unassociated Files']:
                st.markdown(f"❌ {msg}")

    for context, context_results in results_by_context.items():
        if context == 'Unassociated Files': continue
        with st.expander(f"Results for: {context}", expanded=True):
            for status, msg in sorted(context_results, key=lambda x: {'failed': 0, 'passed': 1}.get(x[0], 99)):
                icon = '✅' if status == 'passed' else '❌'
                st.markdown(f"{icon} {msg}")

def main():
    st.markdown("<h1>Artwork Verification Tool</h1>", unsafe_allow_html=True)
    api_keys = check_api_keys()
    if 'validation_complete' not in st.session_state: st.session_state.validation_complete = False

    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Validation", type="primary")
        
        st.header("📝 Custom Rules (Optional)")
        ref_file = st.file_uploader("Reference Text File (.txt)", type=['txt'])
        custom_instructions = st.text_area("Custom Instructions for AI")
        
        st.header("🤖 AI Review (Optional)")
        enable_ai_review = st.toggle("Enable AI Review")
        ai_provider = None
        if enable_ai_review:
            options = {"Select AI...": None}
            if 'openai' in api_keys: options["OpenAI"] = 'openai'
            if 'anthropic' in api_keys: options["Anthropic"] = 'anthropic'
            if 'openai' in api_keys and 'anthropic' in api_keys: options["Both"] = 'both'
            
            if len(options) > 1:
                selected = st.selectbox("Choose AI Provider", options=options.keys())
                ai_provider = options[selected]
            else:
                st.error("No API keys found in secrets.")
        
        if st.button("Clear & Reset"):
            st.session_state.clear(); st.rerun()

    uploaded_files = st.file_uploader("Upload all artwork & QC files", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if run_validation and uploaded_files:
        files = [{"buffer": BytesIO(file.getvalue()), "name": file.name} for file in uploaded_files]
        reference_text = ref_file.read().decode("utf-8").strip() if ref_file else None
        
        with st.spinner("Processing files and running validations..."):
            builder = ProductDataBuilder(files)
            product_data = builder.build()
            st.session_state.product_data = product_data
            
            validator = ArtworkValidator(product_data, reference_text)
            st.session_state.results = validator.validate_all()
            st.session_state.validation_complete = True
            st.session_state.ai_reviews = []

        if enable_ai_review and ai_provider and product_data.get('variants'):
            with st.spinner("Sending data to AI for optional review..."):
                reviewer = AIReviewer(api_keys)
                for sku, variant in product_data['variants'].items():
                    all_files = list(variant.get('files', {}).values()) + list(product_data.get('shared_files', {}).values())
                    text_bundle = "\n".join(d.get('text', '') for d in all_files)
                    summary = reviewer.generate_summary(ai_provider, sku, text_bundle, custom_instructions)
                    st.session_state.ai_reviews.append(summary)
    
    if st.session_state.validation_complete:
        display_results(st.session_state.results)
        if st.session_state.ai_reviews:
            st.markdown("---"); st.header("🤖 AI-Powered Review")
            st.warning("⚠️ **AI Review is Experimental.** Always rely on the rule-based validation and perform a final human review.")
            for summary in st.session_state.ai_reviews:
                st.markdown(summary, unsafe_allow_html=True)
        with st.expander("📄 Data Structure Preview"):
            st.json(st.session_state.product_data)

if __name__ == "__main__":
    main()
