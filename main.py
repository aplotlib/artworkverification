"""
PRODUCTION-READY Packaging Validator for Vive Health
v3.0 FINAL - Batch AI processing and production stability fixes.

This application analyzes a complete set of product packaging documents,
groups them by product, extracts text from PDF, DOCX, and XLSX formats,
runs automated checks, and uses a single, batched AI call for a final,
comprehensive review.
"""

import streamlit as st
import re
import pandas as pd
import logging
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

# --- Core Logic Classes ---

class DocumentProcessor:
    """Handles robust text extraction from multiple file types."""

    @staticmethod
    def extract_text(file, filename):
        """Routes the file to the correct text extraction method based on its type."""
        file_type = file.type
        try:
            if file_type == "application/pdf":
                return DocumentProcessor._extract_text_from_pdf(file, filename)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return DocumentProcessor._extract_text_from_word(file, filename)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return DocumentProcessor._extract_text_from_excel(file, filename)
            else:
                error_msg = f"Unsupported file type: {file_type}"
                logger.warning(error_msg)
                return {'success': False, 'text': '', 'method': 'Unsupported', 'errors': [error_msg]}
        except Exception as e:
            logger.error(f"Fatal error processing file {filename} with type {file_type}: {e}")
            return {'success': False, 'text': '', 'method': 'Error', 'errors': [f"A critical error occurred: {e}"]}


    @staticmethod
    def _extract_text_from_pdf(file_buffer, filename):
        """Extracts text from a PDF using OCR for robustness."""
        text = ""
        file_buffer.seek(0)
        pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            image = Image.open(BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(image, lang='eng')
            text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
        pdf_document.close()
        logger.info(f"Successfully extracted text from PDF {filename} using OCR.")
        return {'success': True, 'text': text, 'method': 'PDF-OCR', 'errors': []}

    @staticmethod
    def _extract_text_from_word(file_buffer, filename):
        """Extracts text from a .docx file."""
        text = ""
        doc = docx.Document(file_buffer)
        for para in doc.paragraphs:
            text += para.text + "\n"
        logger.info(f"Successfully extracted text from Word file {filename}.")
        return {'success': True, 'text': text, 'method': 'DOCX-Parser', 'errors': []}

    @staticmethod
    def _extract_text_from_excel(file_buffer, filename):
        """Extracts text from all sheets of an .xlsx file."""
        text = ""
        xls = pd.ExcelFile(file_buffer)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            text += f"\n\n--- Sheet: {sheet_name} ---\n"
            text += df.to_string()
        logger.info(f"Successfully extracted text from Excel file {filename}.")
        return {'success': True, 'text': text, 'method': 'XLSX-Parser', 'errors': []}


class DocumentIdentifier:
    """Identifies document type based on keywords and content patterns."""
    TYPE_RULES = {
        'Requirements Checklist': {'keywords': ['checklist', 'proofreading'], 'content': ['proofreading checklist', 'hcpcs code']},
        'Packaging Artwork': {'keywords': ['tag', 'lva3100'], 'content': ['california proposition', 'distributed by', r'\(\d{2}\)\d{14}\(\d{3}\)']},
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
        doc_type = doc_info['type']
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
        if 'made in china' not in text_lower:
            results['issues'].append('Missing "Made in China" text.')
        if 'vive health' not in text_lower and 'vive®' not in text_lower:
            results['issues'].append('Missing Vive branding.')
        sku = doc_info.get('sku', 'N/A')
        for suffix, color in DynamicValidator.SKU_SUFFIX_MAP.items():
            if sku.endswith(suffix) and color not in text_lower:
                results['issues'].append(f"SKU/Color Mismatch: SKU is '{sku}' but the color '{color}' was not found.")
                break
        return DynamicValidator._determine_status(results)

    @staticmethod
    def _validate_shipping_mark(text, doc_info):
        results = {'issues': [], 'warnings': []}
        if 'made in china' not in text.lower():
            results['issues'].append('Missing "Made in China" text.')
        if not re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})\s*-\s*(\d+)', text.upper()):
            results['issues'].append("Shipping Mark format error: Expected 'SKU - QTY'.")
        return DynamicValidator._determine_status(results)


class AIReviewer:
    """Handles interaction with AI models for advanced analysis."""

    @staticmethod
    def get_api_config():
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE:
                return 'openai', st.secrets['OPENAI_API_KEY']
            if 'ANTHROPIC_API_KEY' in st.secrets and CLAUDE_AVAILABLE:
                return 'claude', st.secrets['ANTHROPIC_API_KEY']
        return None, None

    @staticmethod
    def get_batch_review(all_products_data, api_type, api_key):
        """Performs a single, batched AI review for all product groups."""
        if not api_key: return {"error": "AI review disabled (No API Key)."}
        if not all_products_data: return {"error": "No data to review."}

        full_text_context = ""
        for product_name, data in all_products_data.items():
            full_text_context += f"\n\n================ PRODUCT GROUP: {product_name} ================\n"
            for filename, file_data in data['files'].items():
                if file_data['extraction']['success']:
                    full_text_context += f"\n--- FILE: {filename} (Type: {file_data['doc_info']['type']}) ---\n"
                    full_text_context += file_data['extraction']['text'][:2000]
                    full_text_context += f"\n--- END OF FILE ---\n"

        prompt = f"""
        You are a meticulous Quality Control specialist for Vive Health. Review a batch of documents for multiple products.
        For each product group, perform the following steps and format your response in structured Markdown:

        1.  **Start with the Product Title:** Use a level-3 Markdown header for each product (e.g., `### Product: Wheelchair Bag Advanced`).
        2.  **Cross-File Consistency Check:** Compare all files within that product group. Is the SKU, product name, and color/variant consistent? If a 'Requirements Checklist' file is present, use it as the source of truth.
        3.  **List Discrepancies:** Provide a clear, bulleted list of all inconsistencies or errors found for that product.
        4.  **Recommendation:** State if the package is "Approved for Production" or "Needs Correction" with a brief reason.
        5.  **Repeat** this process for every product group in the batch.
        """
        try:
            if api_type == 'claude':
                client = anthropic.Anthropic(api_key=api_key, max_retries=3)
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.1
                )
                response_text = response.content[0].text
                ai_reviews = {}
                product_sections = re.split(r'###\s*Product:\s*(.*)', response_text)
                if len(product_sections) > 1:
                    for i in range(1, len(product_sections), 2):
                        product_name_from_ai = product_sections[i].strip()
                        product_review = product_sections[i+1].strip()
                        for original_product_name in all_products_data.keys():
                            if product_name_from_ai.lower() in original_product_name.lower():
                                ai_reviews[original_product_name] = f"### {product_name_from_ai}\n\n{product_review}"
                                break
                else:
                    ai_reviews['batch_summary'] = response_text
                return ai_reviews
            # Add OpenAI logic here if needed in the future
            return {"error": "Selected AI provider logic not fully implemented."}
        except Exception as e:
            logger.error(f"AI batch review failed: {e}")
            return {"error": f"An error occurred during AI batch review: {e}"}

# --- Helper Functions & UI ---

def group_files_by_product(uploaded_files):
    """Groups files based on a common prefix in their names."""
    groups = {}
    for file in uploaded_files:
        match = re.match(r'([a-z_]+)', file.name.lower())
        base_name = match.group(1).replace('_', ' ').title() if match else 'Uncategorized'
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(file)
    return groups

def main():
    """Main function to run the Streamlit application."""
    st.markdown("""
    <style>
        .main-header { padding: 2rem 1rem; margin-bottom: 2rem; background: #f0f2f6; border-radius: 10px; text-align: center; }
        .st-emotion-cache-1y4p8pa { padding-top: 1rem; } /* Reduce top padding */
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Package Reviewer</h1>
        <p>Upload artwork (PDF) and requirements (XLSX, DOCX) to perform an automated review.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_type, api_key = AIReviewer.get_api_config()
        if api_key:
            st.success(f"AI Review Enabled ({api_type.capitalize()})")
        else:
            st.warning("AI Review Disabled (No API Key)")
        st.info("This app runs OCR on PDFs. Ensure your `packages.txt` file is configured correctly for Streamlit Cloud.")
        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload all files for one or more products:",
            type=['pdf', 'xlsx', 'docx'],
            accept_multiple_files=True,
            help="The app automatically groups files by product name."
        )

    if uploaded_files:
        if st.button("🚀 Review All Packages", type="primary", use_container_width=True):
            st.session_state.results = {}
            product_groups = group_files_by_product(uploaded_files)
            total_files = len(uploaded_files)
            progress_bar = st.progress(0, "Initializing...")

            with st.spinner("Analyzing files... This may take a moment."):
                # Step 1: Process all files locally
                for i, (product_name, files) in enumerate(product_groups.items()):
                    st.session_state.results[product_name] = {'files': {}, 'ai_review': 'Pending...'}
                    for file in files:
                        progress_text = f"Analyzing file {i*len(files)+1}/{total_files}: {file.name}"
                        progress_bar.progress((i*len(files)+1)/total_files, text=progress_text)
                        extraction = DocumentProcessor.extract_text(file, file.name)
                        if not extraction['success']:
                            st.session_state.results[product_name]['files'][file.name] = {'error': extraction['errors'][0]}
                            continue
                        doc_info = DocumentIdentifier.identify(file.name, extraction['text'])
                        validation = DynamicValidator.validate(extraction['text'], doc_info)
                        st.session_state.results[product_name]['files'][file.name] = {
                            'extraction': extraction, 'doc_info': doc_info, 'validation': validation
                        }

                # Step 2: Perform one batched AI call
                progress_bar.progress(1.0, "Submitting to AI for final review...")
                if api_key:
                    batched_ai_reviews = AIReviewer.get_batch_review(st.session_state.results, api_type, api_key)
                    if "error" in batched_ai_reviews:
                        st.error(f"AI review failed: {batched_ai_reviews['error']}")
                    else:
                        for product_name, review in batched_ai_reviews.items():
                            if product_name in st.session_state.results:
                                st.session_state.results[product_name]['ai_review'] = review
                progress_bar.empty()

    if st.session_state.results:
        st.markdown("--- \n ## 📊 Validation Report")
        for product_name, data in st.session_state.results.items():
            with st.expander(f"### Product: {product_name}", expanded=True):
                if data['ai_review'] and data['ai_review'] != 'Pending...':
                    st.markdown("#### 🤖 AI-Powered Summary")
                    st.info(data['ai_review'])
                st.markdown("#### 📄 File-by-File Analysis")
                for filename, result in data['files'].items():
                    if 'error' in result:
                        st.error(f"**{filename}:** Could not process file. Reason: {result['error']}")
                        continue
                    status = result['validation']['status']
                    icon = "✅" if status == 'PASS' else "⚠️" if status == 'NEEDS REVIEW' else "❌"
                    st.markdown(f"**{icon} {filename}** (Type: *{result['doc_info']['type']}* | Method: *{result['extraction']['method']}*) - **Status: {status}**")
                    if result['validation']['issues']:
                        for issue in result['validation']['issues']: st.error(f"- {issue}")
                    if result['validation']['warnings']:
                        for warning in result['validation']['warnings']: st.warning(f"- {warning}")
                    if status == 'PASS' and not result['validation']['warnings']:
                        st.success("- All automated checks passed.")

if __name__ == "__main__":
    main()
