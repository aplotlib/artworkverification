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

# --- AI Integration (with Chained Review) ---
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

    def _get_anthropic_review(self, text_bundle, custom_instructions):
        if not self.anthropic_client: return "Anthropic API key not found."
        prompt = f"You are a QA specialist. Review the following artwork text. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {custom_instructions}. Present findings as a bulleted list.\n\n---DATA---\n{text_bundle}\n---END DATA---"
        try:
            response = self.anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=1024, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        except Exception as e: return f"Anthropic API Error: {e}"

    def _get_openai_synthesis(self, text_bundle, anthropic_review, custom_instructions):
        if not self.openai_client: return "OpenAI API key not found."
        prompt = f"""
        You are a senior QA manager. Your task is to provide a final, consolidated summary based on an initial AI review and the original source text.
        A junior AI (Anthropic Claude) has provided an initial analysis. Your job is to review its findings, cross-reference them with the source text, and produce a single, definitive summary. Correct any mistakes or omissions from the first review.

        {custom_instructions}

        ---ORIGINAL ARTWORK TEXT---
        {text_bundle}
        ---END ORIGINAL ARTWORK TEXT---

        ---CLAUDE'S INITIAL REVIEW---
        {anthropic_review}
        ---END CLAUDE'S INITIAL REVIEW---

        Provide your final, synthesized review below as a bulleted list. Start with '### Final AI Analysis'.
        """
        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI API Error: {e}"

    def generate_summary(self, provider, text_bundle, custom_instructions):
        cust_instr = f"Pay special attention to the user's instructions: '{custom_instructions}'" if custom_instructions else ""
        if provider == 'both':
            anthropic_review = self._get_anthropic_review(text_bundle, cust_instr)
            final_summary = self._get_openai_synthesis(text_bundle, anthropic_review, cust_instr)
            return final_summary
        elif provider == 'anthropic':
            return self._get_anthropic_review(text_bundle, cust_instr)
        # Default to OpenAI if it's the only one or selected
        else:
            prompt = f"You are a QA specialist. Review the artwork text. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues. {cust_instr}. Present findings as a bulleted list. Start with '### AI Review'.\n\n---DATA---\n{text_bundle}\n---END DATA---"
            return self._get_summary('openai', prompt)

# --- File Processing ---
class DocumentProcessor:
    def __init__(self, files):
        self.files = files

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
        all_text = []
        all_skus = set()
        for file in self.files:
            file_content = ""
            if file['name'].lower().endswith('.pdf'):
                result = self._extract_from_pdf(file['buffer'])
                if result['success']:
                    file_content = result['text']
            elif file['name'].lower().endswith(('.csv', '.xlsx')):
                file['buffer'].seek(0)
                file_content = file['buffer'].read().decode('utf-8', errors='ignore')
            
            all_text.append(f"--- File: {file['name']} ---\n{file_content}")
            skus_found = re.findall(r'([A-Z]{3,}\d{3,}[A-Z]*)', file_content, re.IGNORECASE)
            for sku in skus_found:
                all_skus.add(sku.upper())
        
        return "\n\n".join(all_text), list(all_skus)

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
        
        if not upcs and not udis:
            results.append(('failed', "No UPCs or UDIs found across all documents.", "no_serials"))
        else:
            for upc in upcs:
                if not any(upc in udi for udi in udis):
                    results.append(('failed', f"UPC `{upc}` does not have a matching UDI.", f"mismatch_{upc}"))
                else:
                    results.append(('passed', f"UPC `{upc}` has a matching UDI.", f"match_{upc}"))
        
        if self.reference_text and self.reference_text not in self.all_text:
            results.append(('failed', "Mandatory text from the reference file was NOT found.", "ref_text_missing"))
        elif self.reference_text:
            results.append(('passed', "Mandatory text from the reference file was found.", "ref_text_ok"))
            
        return results

# --- UI Components ---
def display_report(results, skus):
    st.header("📊 Automated Validation Report")
    if len(skus) > 1:
        st.warning(f"**Multiple Variants Detected**: SKUs `{', '.join(skus)}` were found. Please manually ensure differences are correctly reflected on labels.")

    with st.expander("UDI & Serial Number Analysis", expanded=True):
        st.info("Checks for matching UPCs (12-digit) and UDIs across all files.")
        for status, msg, _ in results:
            icon = '✅' if status == 'passed' else '❌'
            st.markdown(f"{icon} {msg}")

def main():
    st.markdown("<h1>Artwork Verification Tool</h1>", unsafe_allow_html=True)
    api_keys = check_api_keys()
    if 'validation_complete' not in st.session_state: st.session_state.validation_complete = False

    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Validation", type="primary")
        
        st.header("🤖 AI Review")
        custom_instructions = st.text_area("Custom Instructions for AI (Optional)", help="Guide the AI's focus, e.g., 'Check for a 1-year warranty statement.'")
        
        options = {"Select AI Provider...": None}
        if 'openai' in api_keys: options["OpenAI"] = 'openai'
        if 'anthropic' in api_keys: options["Anthropic"] = 'anthropic'
        if 'openai' in api_keys and 'anthropic' in api_keys: options["Both (Anthropic -> OpenAI)"] = 'both'
        
        if len(options) > 1:
            selected = st.selectbox("Choose AI Provider", options=options.keys())
            ai_provider = options[selected]
        else:
            st.warning("No AI API keys found in secrets. AI review is disabled.")
            ai_provider = None
        
        if st.button("Clear & Reset"):
            st.session_state.clear(); st.rerun()

    st.markdown("### Manually Review High-Risk Areas")
    st.info("""
    Based on common errors, please manually check these critical points:
    - **Country of Origin**: Ensure "Made in China" (or correct country) is present and accurate.
    - **Color Matching**: Confirm that colors on packaging match labels and specifications.
    - **UDI Formatting**: Verify that all UDIs are present, correct, and scannable.
    """)
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload all artwork files for one product", type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)
    
    if run_validation and uploaded_files:
        files = [{"buffer": BytesIO(file.getvalue()), "name": file.name} for file in uploaded_files]
        
        with st.spinner("Analyzing all documents..."):
            processor = DocumentProcessor(files)
            all_text_bundle, skus = processor.process_files()
            st.session_state.all_text_bundle = all_text_bundle
            st.session_state.skus = skus
            
            validator = ArtworkValidator(all_text_bundle)
            st.session_state.results = validator.validate()
            st.session_state.validation_complete = True
            st.session_state.ai_review_summary = ""

        if ai_provider and all_text_bundle:
            with st.spinner("Sending data to AI for review..."):
                reviewer = AIReviewer(api_keys)
                summary = reviewer.generate_summary(ai_provider, all_text_bundle, custom_instructions)
                st.session_state.ai_review_summary = summary
    
    if st.session_state.validation_complete:
        display_report(st.session_state.results, st.session_state.skus)
        if st.session_state.ai_review_summary:
            st.markdown("---"); st.header("🤖 AI-Powered Review")
            st.markdown(st.session_state.ai_review_summary, unsafe_allow_html=True)
        with st.expander("📄 View Combined Extracted Text"):
            st.text_area("", st.session_state.all_text_bundle, height=300)

if __name__ == "__main__":
    main()
