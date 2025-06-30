"""
PRODUCTION-READY Packaging Validator for Vive Health
v4.0 FINAL - The Definitive Edition. Implements a nuanced AI with an interactive
sign-off checklist, a fully working and auditable CSV export, and clearer
workflow guidance.
"""

import streamlit as st
import re
import pandas as pd
import logging
import shutil
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
import docx

# --- Basic Configuration ---
# Configure logging for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Vive Health Package Reviewer", page_icon="✅", layout="wide")

# --- Dependency Availability Checks ---
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Core Logic Classes ---

class DocumentProcessor:
    """Handles robust text and image extraction from multiple file types."""

    @staticmethod
    def extract_media(file, filename):
        """Routes the file to the correct extraction method based on its type."""
        file_type = file.type
        try:
            if file_type == "application/pdf":
                return DocumentProcessor._extract_from_pdf(file, filename)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = DocumentProcessor._extract_text_from_word(file, filename)
                return {'success': True, 'text': text['text'], 'images': [], 'method': 'DOCX-Parser'}
            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
                text = DocumentProcessor._extract_text_from_excel(file, filename)
                return {'success': True, 'text': text['text'], 'images': [], 'method': 'Spreadsheet-Parser'}
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.warning(error_msg)
                return {'success': False, 'text': '', 'images': [], 'method': 'Unsupported', 'errors': [error_msg]}
        except Exception as e:
            logger.error(f"Fatal error processing file {filename} with type {file_type}: {e}")
            return {'success': False, 'text': '', 'images': [], 'method': 'Error', 'errors': [f"A critical error occurred: {e}"]}

    @staticmethod
    def _extract_from_pdf(file_buffer, filename):
        """Extracts both text (via OCR) and key images from a PDF."""
        text = ""
        images = []
        file_buffer.seek(0)
        pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            image = Image.open(BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(image, lang='eng')
            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            images.append(image)
        pdf_document.close()
        logger.info(f"Successfully extracted text and images from PDF {filename}.")
        return {'success': True, 'text': text, 'images': images, 'method': 'PDF-OCR+Image', 'errors': []}

    @staticmethod
    def _extract_text_from_word(file_buffer, filename):
        text = ""
        doc = docx.Document(file_buffer)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return {'text': text}

    @staticmethod
    def _extract_text_from_excel(file_buffer, filename):
        text = ""
        xls = pd.ExcelFile(file_buffer)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            text += f"\n\n--- Sheet: {sheet_name} ---\n{df.to_string()}"
        return {'text': text}


class DocumentIdentifier:
    """Identifies document type based on keywords and content patterns."""
    TYPE_RULES = {
        'Requirements Checklist': {'keywords': ['checklist', 'proofreading', 'requirements'], 'content': ['proofreading checklist', 'hcpcs code']},
        'Packaging Artwork': {'keywords': ['tag', 'lva3100', 'artwork'], 'content': ['california proposition', 'distributed by', r'\(\d{2}\)\d{14}\(\d{3}\)']},
        'Quick Start Guide': {'keywords': ['quickstart', 'qsg'], 'content': ['quick start guide', 'application instructions', 'warranty information']},
        'Shipping Mark': {'keywords': ['shipping', 'mark'], 'content': ['item name:', 'po #:', r'[A-Z]{3,4}\d{4,}[A-Z]{0,3}\s*-\s*\d+']},
        'Washtag': {'keywords': ['washtag'], 'content': ['polyester', 'machine wash', 'do not iron']},
        'Logo Tag': {'keywords': ['logo'], 'content': []},
    }

    @staticmethod
    def identify(filename, text):
        info = {'type': 'Unknown', 'sku': 'N/A', 'product_name': 'N/A'}
        text_lower = text.lower()
        filename_lower = filename.lower()
        for doc_type, rules in DocumentIdentifier.TYPE_RULES.items():
            if any(k in filename_lower for k in rules['keywords']):
                info['type'] = doc_type
                break
            if rules.get('content') and any(re.search(c, text_lower, re.IGNORECASE) for c in rules['content']):
                info['type'] = doc_type
                break
        sku_match = re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})', text.upper())
        if not sku_match:
            sku_match = re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})', filename.upper())
        if sku_match:
            info['sku'] = sku_match.group(1)
        if info['type'] == 'Shipping Mark':
            name_match = re.search(r'Item Name:\s*(.*)', text, re.IGNORECASE)
            if name_match:
                info['product_name'] = name_match.group(1).strip()
        elif 'wheelchair bag' in text_lower:
             info['product_name'] = 'Wheelchair Bag Advanced'
        return info

class DynamicValidator:
    """Validates documents against Vive Health's dynamic proofreading checklist."""
    SKU_SUFFIX_MAP = {'BLK': 'black', 'PUR': 'purple floral', 'BGES': 'beige'}

    @staticmethod
    def validate(text, doc_info):
        doc_type = doc_info.get('type', 'Unknown')
        validator_func = getattr(DynamicValidator, f"_validate_{doc_type.lower().replace(' ', '_')}", DynamicValidator._validate_default)
        return validator_func(text, doc_info)

    @staticmethod
    def _determine_status(results):
        if not results['issues']:
            results['status'] = 'NEEDS REVIEW' if results['warnings'] else 'PASS'
        else:
            results['status'] = 'FAIL'
        return results

    @staticmethod
    def _validate_default(text, doc_info):
        return {'status': 'NEEDS REVIEW', 'issues': [], 'warnings': ['No specific validation rules for this document type.']}

    @staticmethod
    def _validate_requirements_checklist(text, doc_info):
        return {'status': 'PASS', 'issues': [], 'warnings': ['This is a reference document for AI review.']}

    @staticmethod
    def _validate_packaging_artwork(text, doc_info):
        results = {'issues': [], 'warnings': []}
        text_lower = text.lower()
        if 'made in china' not in text_lower and 'made in taiwan' not in text_lower:
            results['issues'].append('Missing Country of Origin (e.g., "Made in China").')
        sku = doc_info.get('sku', 'N/A')
        for suffix, color in DynamicValidator.SKU_SUFFIX_MAP.items():
            if sku.upper().endswith(suffix) and color not in text_lower:
                results['issues'].append(f"SKU/Color Mismatch: SKU is '{sku}' but the color '{color}' was not found.")
                break
        if not re.search(r'\(\s*01\s*\)\s*\d{14}', text):
            results['warnings'].append("UDI format may be missing or incorrect. Expected format: (01)xxxxxxxxxxxxxx.")
        return DynamicValidator._determine_status(results)

    @staticmethod
    def _validate_shipping_mark(text, doc_info):
        results = {'issues': [], 'warnings': []}
        if 'made in china' not in text.lower() and 'made in taiwan' not in text.lower():
            results['issues'].append('Missing Country of Origin (e.g., "Made in China").')
        if not re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})\s*-\s*(\d+)', text.upper()):
            results['issues'].append("Shipping Mark format error: Expected 'SKU - QTY'.")
        return DynamicValidator._determine_status(results)

class AIReviewer:
    """Handles interaction with AI models for advanced analysis."""

    @staticmethod
    def get_available_models():
        models = []
        if hasattr(st, 'secrets'):
            if st.secrets.get("ANTHROPIC_API_KEY"):
                models.append("Anthropic Claude 3.5 Sonnet")
            if st.secrets.get("OPENAI_API_KEY"):
                models.append("OpenAI GPT-4o")
            if st.secrets.get("GOOGLE_API_KEY"):
                models.append("Google Gemini 1.5 Pro (Visual Analysis)")
        return models

    @staticmethod
    def get_batch_review(all_products_data, custom_instructions, model_choice, api_keys):
        if not model_choice: return {"error": "No AI model selected or configured."}
        
        # --- MODIFICATION START ---
        # The AI prompt is upgraded to be more nuanced and act as a Design Manager.
        prompt = f"""
        You are a senior Graphic Design Manager at Vive Health. Your task is to provide an expert-level review of product packaging documents. Your tone should be collaborative and helpful.

        **CRITICAL CUSTOM INSTRUCTIONS FOR THIS SESSION:**
        ---
        {custom_instructions if custom_instructions else "No custom instructions provided. Standard review procedures apply."}
        ---
        You MUST treat these custom instructions as a direct order from the project lead and apply them globally to all product groups in this batch.

        Begin your entire response with a single "Session Configuration" block that restates these custom instructions to confirm you have understood them.

        Then, for each product group below, provide a structured review:
        1.  **Product Title:** `### Product: [Product Name]`
        2.  **Manager's Overview:** A high-level paragraph on your overall impression.
        3.  **Actionable Findings:**
            - For each issue, you MUST classify it as either a "Critical Error" or a "Potential Inconsistency".
            - A **Critical Error** is a definite mistake (e.g., typo, wrong SKU, direct contradiction).
            - A **Potential Inconsistency** is a subtle issue that requires human review (e.g., slight color name variation, ambiguous wording).
            - Present each finding using the "Claim, Evidence, Reasoning" format.
            - **Claim:** Start with the classification (e.g., `Critical Error: SKU Mismatch`).
            - **Evidence:** Quote the exact text or describe the visual element, and name the source file.
            - **Reasoning:** Explain why it's a problem.
        4.  **Final Recommendation:** Conclude with "Approved for Production" or "Needs Correction" and a clear justification.
        5.  **Repeat** this structured process for every product group.
        """
        # --- MODIFICATION END ---
        
        try:
            model_input = [prompt]
            if "Gemini" in model_choice:
                for product_name, data in all_products_data.items():
                    model_input.append(f"\n\n--- Start of Product Group: {product_name} ---\n")
                    for filename, file_data in data['files'].items():
                        model_input.append(f"File: {filename}\nText Content:\n{file_data['extraction']['text'][:1500]}")
                        for img in file_data['extraction']['images']: model_input.append(img)
            else:
                text_content = ""
                for product_name, data in all_products_data.items():
                    text_content += f"\n\n--- Start of Product Group: {product_name} ---\n"
                    for filename, file_data in data['files'].items():
                         text_content += f"File: {filename}\nText Content:\n{file_data['extraction']['text'][:2500]}"
                model_input.append(text_content)

            response_text = ""
            if "Claude" in model_choice and CLAUDE_AVAILABLE:
                client = anthropic.Anthropic(api_key=api_keys['anthropic'], max_retries=3)
                messages = [{"role": "user", "content": " ".join(str(item) for item in model_input)}]
                response = client.messages.create(model="claude-3-5-sonnet-20240620", max_tokens=4096, temperature=0.1, messages=messages)
                response_text = response.content[0].text
            elif "GPT" in model_choice and OPENAI_AVAILABLE:
                client = openai.OpenAI(api_key=api_keys['openai'])
                messages = [{"role": "user", "content": " ".join(str(item) for item in model_input)}]
                response = client.chat.completions.create(model="gpt-4o", max_tokens=4096, temperature=0.1, messages=messages)
                response_text = response.choices[0].message.content
            elif "Gemini" in model_choice and GEMINI_AVAILABLE:
                genai.configure(api_key=api_keys['google'])
                model = genai.GenerativeModel('gemini-1.5-pro-latest')
                response = model.generate_content(model_input)
                response_text = response.text
            else:
                return {"error": "Selected AI model is not available or configured correctly."}

            ai_reviews = {}
            product_sections = re.split(r'###\s*Product:\s*(.*)', response_text)
            if len(product_sections) > 1:
                session_config = product_sections[0]
                ai_reviews['session_config'] = session_config
                for i in range(1, len(product_sections), 2):
                    product_name_from_ai = product_sections[i].strip()
                    product_review = product_sections[i+1].strip()
                    for original_product_name in all_products_data.keys():
                        if product_name_from_ai.lower().replace(" ", "") in original_product_name.lower().replace(" ", ""):
                            ai_reviews[original_product_name] = f"### {product_name_from_ai}\n\n{product_review}"
                            break
            else:
                ai_reviews['batch_summary'] = response_text
            return ai_reviews

        except Exception as e:
            logger.error(f"AI batch review failed with model {model_choice}: {e}")
            return {"error": f"An error occurred during AI batch review: {e}"}

# --- Helper Functions & UI ---

def group_files_by_product(uploaded_files):
    """Groups files based on a common prefix in their names."""
    groups = {}
    for file in uploaded_files:
        match = re.match(r'([a-zA-Z_]+(?:bag|guide|mark|tag|checklist))', file.name.lower())
        if match:
             base_name = re.sub(r'_(?:bag|guide|mark|tag|checklist)$', '', match.group(1)).replace('_', ' ').title().strip()
        else:
             base_name = "Uncategorized"
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(file)
    return groups

def prepare_report_data_for_export(results):
    """Converts the nested results dictionary to a flat, auditable list for DataFrame creation."""
    report_rows = []
    custom_instructions = st.session_state.get('run_custom_instructions', 'N/A')
    for product_name, data in results.items():
        ai_review = data.get('ai_review', 'N/A')
        product_findings = st.session_state.get(f"findings_{product_name}", {})
        claims = re.findall(r"\*\*Claim:\*\*(.+?)(?=\*\*Claim:\*\*|\*\*Recommendation:\*\*|\Z)", ai_review, re.DOTALL)
        if claims:
            for i, claim_block in enumerate(claims):
                claim_text = claim_block.split('**Evidence:**')[0].strip()
                evidence_text = claim_block.split('**Evidence:**')[1].split('**Reasoning:**')[0].strip()
                finding_state = product_findings.get('states', {}).get(i, {})
                row = {
                    'Product': product_name, 'Finding Type': 'AI Discrepancy', 'Details': claim_text,
                    'Evidence': evidence_text, 'User Decision': finding_state.get('decision', 'Pending Review'),
                    'User Notes': finding_state.get('notes', ''), 'Custom Instructions for Run': custom_instructions
                }
                report_rows.append(row)
        else:
            row = {
                'Product': product_name, 'Finding Type': 'AI Review', 'Details': 'No discrepancies found by AI.',
                'Evidence': 'N/A', 'User Decision': 'N/A', 'User Notes': 'N/A',
                'Custom Instructions for Run': custom_instructions
            }
            report_rows.append(row)
        for filename, result in data['files'].items():
            if result.get('validation', {}).get('issues'):
                for issue in result['validation']['issues']:
                    report_rows.append({
                        'Product': product_name, 'Finding Type': f'Automated Check ({filename})', 'Details': issue,
                        'Evidence': 'N/A', 'User Decision': 'N/A', 'User Notes': 'N/A',
                        'Custom Instructions for Run': custom_instructions
                    })
    return report_rows

def render_interactive_ai_review(review_text, product_name):
    """Parses the AI review and renders it with interactive sign-off widgets."""
    if not review_text: return
    claims = re.findall(r"(\*\*Claim:\*\*.+?)(?=\*\*Claim:\*\*|\*\*Recommendation:\*\*|\Z)", review_text, re.DOTALL)
    if f"findings_{product_name}" not in st.session_state:
        st.session_state[f"findings_{product_name}"] = {
            "total": len(claims), "states": {i: {"decision": "Pending Review", "notes": ""} for i in range(len(claims))}
        }
    product_state = st.session_state[f"findings_{product_name}"]
    st.info(review_text.split("**Claim:**")[0])
    if not claims:
        recommendation = review_text.split("**Recommendation:**")[-1]
        st.markdown(f"**Recommendation:** {recommendation}")
        return
    for i, claim_block in enumerate(claims):
        with st.container(border=True):
            claim_text_full = claim_block.split('**Evidence:**')[0].replace('**Claim:**','').strip()
            claim_icon = "❌" if "Critical Error" in claim_text_full else "⚠️"
            st.markdown(f"**Claim:** {claim_icon} {claim_text_full}")
            st.markdown(f"**Evidence:**{claim_block.split('**Evidence:**')[1].split('**Reasoning:**')[0].strip()}")
            st.markdown(f"**Reasoning:**{claim_block.split('**Reasoning:**')[1].strip()}")
            st.markdown("---")
            cols = st.columns([2, 3])
            decision_key = f"decision_{product_name}_{i}"
            notes_key = f"notes_{product_name}_{i}"
            with cols[0]:
                product_state["states"][i]["decision"] = st.radio(
                    "Your Action:", options=["Pending Review", "Tool is Correct - Correction Made", "Tool is Incorrect - Ignore Finding"],
                    key=decision_key, index=["Pending Review", "Tool is Correct - Correction Made", "Tool is Incorrect - Ignore Finding"].index(product_state["states"][i]["decision"]),
                    label_visibility="collapsed"
                )
            with cols[1]:
                product_state["states"][i]["notes"] = st.text_input(
                    "Notes (optional):", key=notes_key, value=product_state["states"][i]["notes"],
                    placeholder="e.g., 'Updated artwork and sent to printer.'"
                )
    recommendation = review_text.split("**Recommendation:**")[-1]
    st.markdown(f"**Recommendation:** {recommendation}")
    reviewed_count = sum(1 for state in product_state["states"].values() if state["decision"] != "Pending Review")
    if reviewed_count == product_state["total"] and product_state["total"] > 0:
        st.success("✅ Well done! All findings for this product have been reviewed.")
        return True
    return False

def main():
    """Main function to run the Streamlit application."""
    st.markdown("""
    <style>
        .main-header { padding: 1.5rem 1rem; margin-bottom: 2rem; background: #f0f2f6; border-radius: 10px; text-align: center; }
        .st-emotion-cache-1y4p8pa { padding-top: 1rem; }
        .stButton>button { width: 100%; }
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Package Reviewer</h1>
        <p>Your expert AI design manager for packaging and artwork proofreading.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'run_custom_instructions' not in st.session_state:
        st.session_state.run_custom_instructions = ""

    with st.sidebar:
        st.markdown("### ⚙️ AI Configuration")
        available_models = AIReviewer.get_available_models()
        if not available_models:
            st.warning("No AI models configured. Please add an API key to your Streamlit secrets to enable AI review.")
            model_choice = None
        else:
            model_choice = st.selectbox("Choose AI Model:", available_models, help="Gemini is recommended for visual analysis.")
        
        st.markdown("### 📝 Custom Instructions for this Session")
        custom_instructions = st.text_area(
            "Add special rules for this review. For example: 'For this batch, all products are made in Taiwan.'",
            height=150, key="custom_instructions"
        )
        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload all files for one or more products:",
            type=['pdf', 'xlsx', 'docx', 'csv'], accept_multiple_files=True,
            help="The app automatically groups files by product name."
        )

    if uploaded_files:
        if st.button("🚀 Review All Packages", type="primary", disabled=(not available_models)):
            st.session_state.clear()
            st.session_state.results = {}
            st.session_state.run_custom_instructions = custom_instructions
            product_groups = group_files_by_product(uploaded_files)
            total_files = len(uploaded_files)
            
            with st.spinner(f"Analyzing {total_files} files... This may take a moment."):
                progress_bar = st.progress(0, "Initializing...")
                files_processed = 0
                for product_name, files in product_groups.items():
                    st.session_state.results[product_name] = {'files': {}, 'ai_review': 'Pending...'}
                    for file in files:
                        files_processed += 1
                        progress_bar.progress(files_processed / total_files, f"Analyzing: {file.name}")
                        extraction = DocumentProcessor.extract_media(file, file.name)
                        if not extraction['success']:
                            st.session_state.results[product_name]['files'][file.name] = {'error': extraction['errors'][0]}
                            continue
                        doc_info = DocumentIdentifier.identify(file.name, extraction['text'])
                        validation = DynamicValidator.validate(extraction['text'], doc_info)
                        st.session_state.results[product_name]['files'][file.name] = {
                            'extraction': extraction, 'doc_info': doc_info, 'validation': validation
                        }
                progress_bar.progress(1.0, "Submitting to AI for final, evidence-based review...")
                api_keys = {
                    'anthropic': st.secrets.get("ANTHROPIC_API_KEY"),
                    'openai': st.secrets.get("OPENAI_API_KEY"),
                    'google': st.secrets.get("GOOGLE_API_KEY")
                }
                batched_ai_reviews = AIReviewer.get_batch_review(st.session_state.results, st.session_state.run_custom_instructions, model_choice, api_keys)
                if "error" in batched_ai_reviews:
                    st.error(f"AI review failed: {batched_ai_reviews['error']}")
                else:
                    st.session_state.global_ai_config = batched_ai_reviews.pop('session_config', None)
                    for product_name, review in batched_ai_reviews.items():
                        if product_name in st.session_state.results:
                            st.session_state.results[product_name]['ai_review'] = review
                progress_bar.empty()

    if st.session_state.results:
        st.markdown("--- \n ## 📊 Validation Report")
        
        if st.session_state.get('global_ai_config'):
            with st.container(border=True):
                st.markdown("#### 📝 AI Session Configuration")
                st.markdown(st.session_state.global_ai_config)

        all_findings_reviewed = True
        for product_name, data in st.session_state.results.items():
            with st.expander(f"Product: {product_name}", expanded=True):
                if data.get('ai_review') and data['ai_review'] != 'Pending...':
                    st.markdown("#### 🤖 AI Design Manager Review")
                    is_complete = render_interactive_ai_review(data['ai_review'], product_name)
                    if not is_complete:
                        all_findings_reviewed = False
                st.markdown("---")
                st.markdown("#### 📄 Automated Checks")
                for filename, result in data['files'].items():
                    if 'error' in result:
                        st.error(f"**{filename}:** Could not process file. Reason: {result['error']}")
                        continue
                    status = result['validation']['status']
                    icon = "✅" if status == 'PASS' else "⚠️" if status == 'NEEDS REVIEW' else "❌"
                    st.markdown(f"**{icon} {filename}** (Type: *{result['doc_info']['type']}* | SKU: *{result['doc_info']['sku']}*) - **Status: {status}**")
                    if result['validation']['issues']:
                        for issue in result['validation']['issues']: st.error(f"- {issue}")
                    if result['validation']['warnings']:
                        for warning in result['validation']['warnings']: st.warning(f"- {warning}")
                    if status == 'PASS' and not result['validation']['warnings']:
                        st.success("- All automated checks passed.")
        
        st.markdown("---")
        st.markdown("### 📥 Finalize & Export")
        if all_findings_reviewed:
            st.success("All AI findings have been reviewed. You can now generate the final audit log.")
            report_df = pd.DataFrame(prepare_report_data_for_export(st.session_state.results))
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Final Audit Log (CSV)", data=csv,
                file_name="vive_health_audit_log.csv", mime="text/csv"
            )
        else:
            st.warning("Please review and sign off on all AI findings above before exporting the final report.")


if __name__ == "__main__":
    main()
