import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image, ImageFile
import fitz  # PyMuPDF
from typing import Dict, List, Any
from pyzbar.pyzbar import decode as qr_decode
import openai
import anthropic

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="Artwork Verification Tool", page_icon="✅", layout="wide")

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

    def generate_summary(self, provider, text_bundle, custom_instructions):
        instruction_prompt = f"Additionally, follow these instructions: '{custom_instructions}'" if custom_instructions else ""
        prompt = f"You are a QA specialist reviewing artwork files. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {instruction_prompt}. Present findings as a bulleted list. Start with '### AI Review'.\n\n---BEGIN DATA---\n{text_bundle}\n---END DATA---"
        
        if provider == 'both':
            openai_summary = self._get_summary('openai', prompt)
            anthropic_summary = self._get_summary('anthropic', prompt)
            return f"""<div style="display: flex; gap: 1rem;"><div style="flex: 1;"><h4>OpenAI Review</h4>{openai_summary}</div><div style="flex: 1;"><h4>Anthropic Review</h4>{anthropic_summary}</div></div>"""
        return self._get_summary(provider, prompt)

# --- File Processing ---
class DocumentProcessor:
    def __init__(self, files):
        self.files = files
        self.extracted_data = []
        self.skus = set()

    def _extract_from_pdf(self, file_buffer):
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

    def process_files(self):
        for file in self.files:
            if file['name'].lower().endswith('.pdf'):
                result = self._extract_from_pdf(file['buffer'])
                if result['success']:
                    self.extracted_data.append({'filename': file['name'], **result})
                    # Simple SKU extraction from filename for variant detection
                    sku_match = re.search(r'([A-Z0-9]{5,})', file['name'])
                    if sku_match:
                        self.skus.add(sku_match.group(1))
        return self.extracted_data, list(self.skus)

# --- Core Logic ---
class ArtworkValidator:
    def __init__(self, all_text, reference_text=None):
        self.all_text = all_text
        self.reference_text = reference_text

    def validate(self):
        results = []
        cleaned_text = self.all_text.replace(" ", "").replace("\n", "")
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))

        # UDI & UPC Analysis
        if not upcs and not udis:
            results.append(('failed', "No UPCs or UDIs found in any of the documents.", "no_serials"))
        else:
            for upc in upcs:
                if not any(upc in udi for udi in udis):
                    results.append(('failed', f"UPC {upc} does not have a matching UDI.", f"mismatch_{upc}"))
                else:
                    results.append(('passed', f"UPC {upc} has a matching UDI.", f"match_{upc}"))
        
        # Reference Text Check
        if self.reference_text and self.reference_text not in self.all_text:
            results.append(('failed', "Mandatory text from the reference file was NOT found.", "ref_text_missing"))
        elif self.reference_text:
            results.append(('passed', "Mandatory text from the reference file was found.", "ref_text_ok"))
            
        return results

# --- UI Components ---
def display_report(results, skus):
    st.header("📊 Automated Validation Report")

    # Variant Disclaimer
    if len(skus) > 1:
        st.warning(f"**Multiple Variants Detected**: The files appear to contain artwork for multiple SKUs ({', '.join(skus)}). Please manually ensure that the differences between these variants are correctly reflected on their respective labels.")

    # UDI & Serial Analysis
    with st.expander("UDI & Serial Number Analysis", expanded=True):
        st.info("This section checks for the presence of UPCs (12-digit serials) and UDIs, and verifies that they match.")
        serial_results = [res for res in results if 'no_serials' in res[2] or 'mismatch' in res[2] or 'match' in res[2]]
        if not serial_results:
            st.markdown("- No UPCs or UDIs found to analyze.")
        else:
            for status, msg, _ in serial_results:
                icon = '✅' if status == 'passed' else '❌'
                st.markdown(f"{icon} {msg}")

    # Other Checks
    other_results = [res for res in results if not ('no_serials' in res[2] or 'mismatch' in res[2] or 'match' in res[2])]
    if other_results:
        with st.expander("Additional Checks", expanded=True):
            for status, msg, _ in other_results:
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

    st.markdown("### Manual Checkpoints")
    st.info("Based on past issues, please manually double-check these key areas.")
    cols = st.columns(2)
    cols[0].checkbox("Is the **Country of Origin** correct on all labels?", key="check1")
    cols[0].checkbox("Does the **Logo Color** match specifications?", key="check2")
    cols[1].checkbox("Are all **QR codes** present and correct?", key="check3")
    cols[1].checkbox("Do **UDIs** appear correctly formatted on all labels?", key="check4")
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload all artwork files for one product", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if run_validation and uploaded_files:
        files = [{"buffer": BytesIO(file.getvalue()), "name": file.name} for file in uploaded_files]
        reference_text = ref_file.read().decode("utf-8").strip() if ref_file else None
        
        with st.spinner("Analyzing all documents..."):
            processor = DocumentProcessor(files)
            extracted_data, skus = processor.process_files()
            st.session_state.extracted_data = extracted_data
            st.session_state.skus = skus
            
            all_text_bundle = "\n\n---***---\n\n".join([f"File: {d['filename']}\n\n{d['text']}" for d in extracted_data])
            st.session_state.all_text_bundle = all_text_bundle
            
            validator = ArtworkValidator(all_text_bundle, reference_text)
            st.session_state.results = validator.validate()
            st.session_state.validation_complete = True
            st.session_state.ai_review_summary = ""

        if enable_ai_review and ai_provider and all_text_bundle:
            with st.spinner("Sending data to AI for optional review..."):
                reviewer = AIReviewer(api_keys)
                summary = reviewer.generate_summary(ai_provider, all_text_bundle, custom_instructions)
                st.session_state.ai_review_summary = summary
    
    if st.session_state.validation_complete:
        display_report(st.session_state.results, st.session_state.skus)
        if st.session_state.ai_review_summary:
            st.markdown("---"); st.header("🤖 AI-Powered Review")
            st.warning("⚠️ **AI Review is Experimental.** Always rely on the rule-based validation and perform a final human review.")
            st.markdown(st.session_state.ai_review_summary, unsafe_allow_html=True)
        with st.expander("📄 View Extracted Text"):
            st.text(st.session_state.all_text_bundle)

if __name__ == "__main__":
    main()
