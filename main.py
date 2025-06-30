"""
PRODUCTION-READY Packaging Validator for Vive Health
v3.4 FINAL - Adds a comprehensive CSV export feature that includes the
custom instructions used for the review session.

This application analyzes a complete set of product packaging documents,
groups them by product, extracts text from PDF, DOCX, and XLSX formats,
runs automated checks, and uses a single, batched AI call for a final,
comprehensive review.
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
            elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
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
            pix = page.get_pixmap(dpi=300) # Higher DPI for better OCR quality
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
        """Extracts text from all sheets of an .xlsx or .csv file."""
        text = ""
        # Use pandas to read both Excel and CSV
        xls = pd.ExcelFile(file_buffer)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            text += f"\n\n--- Sheet: {sheet_name} ---\n"
            text += df.to_string()
        logger.info(f"Successfully extracted text from Excel/CSV file {filename}.")
        return {'success': True, 'text': text, 'method': 'Spreadsheet-Parser', 'errors': []}


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
        # Extract SKU (handles variants like BLK, PUR, BGES)
        sku_match = re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})', text.upper())
        if not sku_match:
            sku_match = re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})', filename.upper())
        if sku_match:
            info['sku'] = sku_match.group(1)

        # Extract Product Name
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
        """Routes to the correct validation function based on document type."""
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
    def get_api_config():
        """Securely get API keys from Streamlit secrets."""
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE and st.secrets['OPENAI_API_KEY']:
                return 'openai', st.secrets['OPENAI_API_KEY']
            if 'ANTHROPIC_API_KEY' in st.secrets and CLAUDE_AVAILABLE and st.secrets['ANTHROPIC_API_KEY']:
                return 'claude', st.secrets['ANTHROPIC_API_KEY']
        return None, None

    @staticmethod
    def get_batch_review(all_products_data, custom_instructions, api_type, api_key):
        """Performs a single, batched AI review for all product groups."""
        if not api_key: return {"error": "AI review disabled (No API Key)."}
        if not all_products_data: return {"error": "No data to review."}

        full_text_context = ""
        for product_name, data in all_products_data.items():
            full_text_context += f"\n\n================ PRODUCT GROUP: {product_name} ================\n"
            for filename, file_data in data['files'].items():
                if file_data.get('extraction', {}).get('success', False):
                    full_text_context += f"\n--- FILE: {filename} (Type: {file_data['doc_info']['type']}) ---\n"
                    full_text_context += file_data['extraction']['text'][:2500] # Limit text per file for context window
                    full_text_context += f"\n--- END OF FILE ---\n"

        prompt = f"""
        You are a meticulous Quality Control specialist for Vive Health. Your primary goal is to find and provide evidence for critical inconsistencies, typos, or grammatical errors.
        
        **CRITICAL CUSTOM INSTRUCTIONS FOR THIS SESSION:**
        ---
        {custom_instructions if custom_instructions else "No custom instructions provided."}
        ---
        You MUST follow these custom instructions above all other rules.

        For each product group below, perform the following steps and format your response in structured Markdown:
        1.  **Start with the Product Title:** Use a level-3 Markdown header (e.g., `### Product: Wheelchair Bag`).
        2.  **Brand Verification:** Confirm the "Vive" brand name or logo is present. If missing, report this as a critical error.
        3.  **Analyze and Report Other Errors:**
            - If you find an error, you MUST follow the "Claim, Evidence, Reasoning" format.
            - **Claim:** A 1-line summary of the error (e.g., "Product Name Mismatch").
            - **Evidence:** Quote the exact text from each conflicting file, and name the source file in parentheses.
            - **Reasoning:** A 1-line explanation of why it's a problem.
        4.  **Recommendation:** State if the package is "Approved" or "Needs Correction" and why.
        5.  **No Errors:** If a product group is perfect, simply write: "**Recommendation:** Approved for Production. All documents are consistent and no errors were found."
        6.  **Repeat** for every product group in the batch.
        """
        try:
            if api_type == 'claude':
                client = anthropic.Anthropic(api_key=api_key, max_retries=3) # Use the client's built-in retry
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.05
                )
                response_text = response.content[0].text
                ai_reviews = {}
                # Split the response by product headers to parse the output
                product_sections = re.split(r'###\s*Product:\s*(.*)', response_text)
                if len(product_sections) > 1:
                    for i in range(1, len(product_sections), 2):
                        product_name_from_ai = product_sections[i].strip()
                        product_review = product_sections[i+1].strip()
                        # Match the AI's response back to the original product group name
                        for original_product_name in all_products_data.keys():
                            if product_name_from_ai.lower().replace(" ", "") in original_product_name.lower().replace(" ", ""):
                                ai_reviews[original_product_name] = f"### {product_name_from_ai}\n\n{product_review}"
                                break
                else: # Fallback if splitting fails
                    ai_reviews['batch_summary'] = response_text
                return ai_reviews
            return {"error": "Selected AI provider logic not fully implemented."}
        except Exception as e:
            logger.error(f"AI batch review failed: {e}")
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

# --- MODIFICATION START ---
# New function to prepare a detailed DataFrame for export.
def prepare_report_data_for_export(results, custom_instructions):
    """Converts the nested results dictionary to a flat list for DataFrame creation, including custom instructions."""
    report_rows = []
    for product_name, data in results.items():
        ai_review = data.get('ai_review', 'N/A')
        for filename, result in data['files'].items():
            if 'error' in result:
                row = {
                    'Product': product_name,
                    'File Name': filename,
                    'Status': 'PROCESSING ERROR',
                    'Details': result['error'],
                    'Document Type': 'N/A',
                    'SKU': 'N/A',
                    'AI Review Summary': ai_review,
                    'Custom Instructions for Run': custom_instructions
                }
            else:
                issues = "; ".join(result['validation']['issues'])
                warnings = "; ".join(result['validation']['warnings'])
                details = f"Issues: {issues}. Warnings: {warnings}." if issues or warnings else "All automated checks passed."
                row = {
                    'Product': product_name,
                    'File Name': filename,
                    'Status': result['validation']['status'],
                    'Details': details,
                    'Document Type': result['doc_info']['type'],
                    'SKU': result['doc_info']['sku'],
                    'AI Review Summary': ai_review,
                    'Custom Instructions for Run': custom_instructions
                }
            report_rows.append(row)
    return report_rows
# --- MODIFICATION END ---


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
        <p>Upload artwork (PDF), requirements (XLSX, DOCX), and other files to perform an automated review.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'run_custom_instructions' not in st.session_state:
        st.session_state.run_custom_instructions = ""


    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_type, api_key = AIReviewer.get_api_config()
        if api_key:
            st.success(f"AI Review Enabled ({api_type.capitalize()})")
        else:
            st.warning("AI Review Disabled")
            st.markdown("To enable AI analysis, please add your API Key to your Streamlit secrets.")

        if not shutil.which("tesseract"):
            st.error("Tesseract OCR is not installed or not in the system's PATH. PDF processing will fail.")
            st.info("For deployment on Streamlit Cloud, ensure `packages.txt` contains `tesseract-ocr`.")
        
        st.markdown("---")
        
        st.markdown("### 📝 Custom Instructions for this Session")
        custom_instructions = st.text_area(
            "Add any special rules or context for this review. For example: 'For this batch, all products are made in Taiwan.'",
            height=150,
            key="custom_instructions"
        )
        
        st.markdown("---")
        uploaded_files = st.file_uploader(
            "Upload all files for one or more products:",
            type=['pdf', 'xlsx', 'docx', 'csv'],
            accept_multiple_files=True,
            help="The app automatically groups files by product name."
        )

    if uploaded_files:
        if st.button("🚀 Review All Packages", type="primary"):
            st.session_state.results = {}
            st.session_state.run_custom_instructions = st.session_state.get("custom_instructions", "")
            product_groups = group_files_by_product(uploaded_files)
            total_files = len(uploaded_files)
            
            with st.spinner(f"Analyzing {total_files} files... This may take a moment."):
                progress_bar = st.progress(0, "Initializing...")
                
                # Step 1: Process all files locally
                files_processed = 0
                for product_name, files in product_groups.items():
                    st.session_state.results[product_name] = {'files': {}, 'ai_review': 'Pending...'}
                    for file in files:
                        files_processed += 1
                        progress_bar.progress(files_processed / total_files, f"Analyzing: {file.name}")
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
                progress_bar.progress(1.0, "Submitting to AI for final, evidence-based review...")
                if api_key:
                    batched_ai_reviews = AIReviewer.get_batch_review(st.session_state.results, st.session_state.run_custom_instructions, api_type, api_key)
                    if "error" in batched_ai_reviews:
                        st.error(f"AI review failed: {batched_ai_reviews['error']}")
                    else:
                        for product_name, review in batched_ai_reviews.items():
                            if product_name in st.session_state.results:
                                st.session_state.results[product_name]['ai_review'] = review
                progress_bar.empty()

    if st.session_state.results:
        st.markdown("--- \n ## 📊 Validation Report")
        
        # --- MODIFICATION START ---
        # Add the download button here, using the new report generation function
        report_df = pd.DataFrame(prepare_report_data_for_export(st.session_state.results, st.session_state.run_custom_instructions))
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Report to CSV",
            data=csv,
            file_name="vive_health_validation_report.csv",
            mime="text/csv",
            key='download_csv'
        )
        # --- MODIFICATION END ---

        for product_name, data in st.session_state.results.items():
            overall_status = "PASS"
            for result in data['files'].values():
                if 'error' in result or result.get('validation', {}).get('status') == 'FAIL':
                    overall_status = "FAIL"
                    break
                if result.get('validation', {}).get('status') == 'NEEDS REVIEW':
                    overall_status = "NEEDS REVIEW"
            
            status_icon = "✅" if overall_status == "PASS" else "⚠️" if overall_status == "NEEDS REVIEW" else "❌"

            with st.expander(f"{status_icon} Product: {product_name}", expanded=True):
                if data['ai_review'] and data['ai_review'] != 'Pending...':
                    st.markdown("#### 🤖 AI-Powered Evidentiary Review")
                    st.info(data['ai_review'])
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

if __name__ == "__main__":
    main()
