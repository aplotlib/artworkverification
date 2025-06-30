"""
PRODUCTION-READY Packaging Validator for Vive Health
v2.1 - Now supports PDF (with OCR), DOCX, and XLSX files.

This application analyzes a complete set of product packaging documents,
groups them by product, extracts text from various file formats, and validates them
against Vive Health's specific proofreading checklists.
"""

import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
import docx # For reading .docx files

# --- Configuration ---
# Configure logging
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
        """
        Routes the file to the correct text extraction method based on its type.
        """
        file_type = file.type
        file_buffer = file
        
        if file_type == "application/pdf":
            return DocumentProcessor._extract_text_from_pdf(file_buffer, filename)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocumentProcessor._extract_text_from_word(file_buffer, filename)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return DocumentProcessor._extract_text_from_excel(file_buffer, filename)
        else:
            error_msg = f"Unsupported file type: {file_type}"
            logger.warning(error_msg)
            return {'success': False, 'text': '', 'method': 'Unsupported', 'errors': [error_msg]}

    @staticmethod
    def _extract_text_from_pdf(file_buffer, filename):
        """Extracts text from a PDF using OCR for robustness."""
        text = ""
        try:
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
        except Exception as e:
            logger.error(f"PDF-OCR extraction failed for {filename}: {e}")
            return {'success': False, 'text': '', 'method': 'PDF-OCR', 'errors': [f"Error during PDF processing: {e}"]}

    @staticmethod
    def _extract_text_from_word(file_buffer, filename):
        """Extracts text from a .docx file."""
        text = ""
        try:
            doc = docx.Document(file_buffer)
            for para in doc.paragraphs:
                text += para.text + "\n"
            logger.info(f"Successfully extracted text from Word file {filename}.")
            return {'success': True, 'text': text, 'method': 'DOCX-Parser', 'errors': []}
        except Exception as e:
            logger.error(f"DOCX extraction failed for {filename}: {e}")
            return {'success': False, 'text': '', 'method': 'DOCX-Parser', 'errors': [f"Error reading .docx file: {e}"]}

    @staticmethod
    def _extract_text_from_excel(file_buffer, filename):
        """Extracts text from all sheets of an .xlsx file."""
        text = ""
        try:
            xls = pd.ExcelFile(file_buffer)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                # Convert the entire sheet to a string format
                text += f"\n\n--- Sheet: {sheet_name} ---\n"
                text += df.to_string()
            logger.info(f"Successfully extracted text from Excel file {filename}.")
            return {'success': True, 'text': text, 'method': 'XLSX-Parser', 'errors': []}
        except Exception as e:
            logger.error(f"XLSX extraction failed for {filename}: {e}")
            return {'success': False, 'text': '', 'method': 'XLSX-Parser', 'errors': [f"Error reading .xlsx file: {e}"]}


class DocumentIdentifier:
    """Identifies document type based on keywords and content patterns."""
    
    # Enhanced rules based on the provided checklist and examples
    TYPE_RULES = {
        'Packaging Artwork': {'keywords': ['tag', 'lva3100'], 'content': ['california proposition', 'distributed by', r'\(\d{2}\)\d{14}\(\d{3}\)']},
        'Quick Start Guide': {'keywords': ['quickstart', 'qsg'], 'content': ['quick start guide', 'application instructions', 'warranty information']},
        'Shipping Mark': {'keywords': ['shipping', 'mark'], 'content': ['item name:', 'po #:', r'[A-Z]{3,4}\d{4,}[A-Z]{0,3}\s*-\s*\d+']},
        'Washtag': {'keywords': ['washtag'], 'content': ['polyester', 'machine wash', 'do not iron']},
        'Logo Tag': {'keywords': ['logo'], 'content': []},
        'Requirements Checklist': {'keywords': ['checklist', 'proofreading'], 'content': ['proofreading checklist', 'hcpcs code']}
    }

    @staticmethod
    def identify(filename, text):
        info = {'type': 'Unknown', 'sku': 'N/A', 'product_name': 'N/A'}
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Identify by content and filename
        for doc_type, rules in DocumentIdentifier.TYPE_RULES.items():
            if any(k in filename_lower for k in rules['keywords']):
                info['type'] = doc_type
                break
            if rules.get('content') and any(re.search(c, text_lower) for c in rules['content']):
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

    SKU_SUFFIX_MAP = {
        'BLK': 'black',
        'PUR': 'purple floral',
        'BGES': 'beige - small' 
        # Add more mappings as needed
    }
    
    @staticmethod
    def validate(text, doc_info):
        doc_type = doc_info['type']
        validator_func = getattr(DynamicValidator, f"_validate_{doc_type.lower().replace(' ', '_')}", DynamicValidator._validate_default)
        return validator_func(text, doc_info)

    @staticmethod
    def _validate_default(text, doc_info):
        return {'status': 'NEEDS REVIEW', 'issues': [], 'warnings': ['No specific validation rules for this document type.']}

    @staticmethod
    def _validate_requirements_checklist(text, doc_info):
        # Checklists are for reference and don't need validation themselves
        return {'status': 'PASS', 'issues': [], 'warnings': ['This is a reference document.']}

    @staticmethod
    def _validate_packaging_artwork(text, doc_info):
        results = {'issues': [], 'warnings': []}
        text_lower = text.lower()
        
        if 'made in china' not in text_lower:
            results['issues'].append('Missing "Made in China" text.')
        if 'vive health' not in text_lower and 'vive®' not in text_lower:
            results['issues'].append('Missing Vive branding.')
        if 'california proposition' not in text_lower and 'p65warnings' not in text_lower:
            results['warnings'].append('Missing Prop 65 Warning (if applicable).')
        if not re.search(r'(\(01\)\d{14})', text):
            results['warnings'].append('UDI Giftbox format may be missing or incorrect.')

        # ** CRITICAL VARIANT CHECK **
        sku = doc_info.get('sku', 'N/A')
        for suffix, color in DynamicValidator.SKU_SUFFIX_MAP.items():
            if sku.endswith(suffix):
                if color not in text_lower:
                    results['issues'].append(f"SKU/Color Mismatch: SKU is '{sku}' but the color '{color}' was not found on the artwork.")
                break
        
        return DynamicValidator._determine_status(results)

    @staticmethod
    def _validate_shipping_mark(text, doc_info):
        results = {'issues': [], 'warnings': []}
        text_lower = text.lower()
        
        if 'made in china' not in text_lower:
            results['issues'].append('Missing "Made in China" text.')
        
        # Check format SKU - QTY
        sku_qty_match = re.search(r'([A-Z]{3}\d{4,}[A-Z]{0,4})\s*-\s*(\d+)', text.upper())
        if not sku_qty_match:
            results['issues'].append("Shipping Mark format error: Expected 'SKU - QTY'.")
        else:
            # Check if the SKU on the mark matches the detected SKU
            if sku_qty_match.group(1) != doc_info.get('sku'):
                 results['issues'].append(f"SKU Mismatch: Shipping Mark shows '{sku_qty_match.group(1)}' but document is for '{doc_info.get('sku')}'.")
        
        return DynamicValidator._determine_status(results)
        
    @staticmethod
    def _validate_quick_start_guide(text, doc_info):
        results = {'issues': [], 'warnings': []}
        text_lower = text.lower()
        if 'vivehealth.com' not in text_lower:
            results['issues'].append('Missing vivehealth.com URL.')
        if 'warranty information' not in text_lower:
            results['warnings'].append('Missing warranty information.')
        return DynamicValidator._determine_status(results)

    @staticmethod
    def _validate_washtag(text, doc_info):
        results = {'issues': [], 'warnings': []}
        text_lower = text.lower()
        if 'made in china' not in text_lower:
            results['issues'].append('Missing "Made in China" text.')
        if not any(icon in text_lower for icon in ['machine wash', 'do not iron', 'polyester']):
             results['warnings'].append('Washtag seems to be missing care instructions/icons.')
        return DynamicValidator._determine_status(results)

    @staticmethod
    def _determine_status(results):
        if not results['issues']:
            results['status'] = 'NEEDS REVIEW' if results['warnings'] else 'PASS'
        else:
            results['status'] = 'FAIL'
        return results

class AIReviewer:
    """Handles interaction with AI models for advanced analysis."""
    @staticmethod
    def get_api_config():
        # Securely get API keys from Streamlit secrets
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE:
                return 'openai', st.secrets['OPENAI_API_KEY']
            if 'ANTHROPIC_API_KEY' in st.secrets and CLAUDE_AVAILABLE:
                return 'claude', st.secrets['ANTHROPIC_API_KEY']
        return None, None

    @staticmethod
def get_review(product_files_data, api_type, api_key):
        if not api_key: return "AI review disabled (no API key)."
        if not product_files_data: return "No data to review."

        # Consolidate text from all files for cross-comparison
        full_text_context = ""
        for filename, data in product_files_data.items():
            if data['extraction']['success']:
                full_text_context += f"\n\n--- START OF FILE: {filename} (Type: {data['doc_info']['type']}) ---\n"
                full_text_context += data['extraction']['text']
                full_text_context += f"\n--- END OF FILE: {filename} ---\n"

        prompt = f"""
        You are a meticulous Quality Control specialist for Vive Health, reviewing a package of documents for a single product before production.
        One of these documents is likely a 'Requirements Checklist' from an Excel or Word file. The other files are artwork (PDFs).

        Your task is to find inconsistencies and errors.

        PRODUCT DOCUMENT SET:
        {full_text_context[:8000]} 

        INSTRUCTIONS:
        1.  **Cross-File Consistency Check (CRITICAL):** Carefully compare all provided artwork files against the rules defined in the 'Requirements Checklist' document. Then, compare all the artwork files to EACH OTHER. Is the product name, SKU, color/variant, and other key information perfectly consistent across all documents?
        2.  **List Discrepancies:** Provide a clear, bulleted list of any and all inconsistencies you find. For each point, state the filename and the specific error.
        3.  **Spelling & Grammar:** Briefly note any spelling or grammatical errors found in the text on the artwork files.
        4.  **Final Recommendation:** Based on your review, state if the package is "Approved for Production" or "Needs Correction" and provide a brief summary of why. If there are no issues, state that "All documents are consistent and approved."
        """
        
        try:
            if api_type == 'openai':
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=800, temperature=0.1)
                return response.choices[0].message.content
            elif api_type == 'claude':
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}], max_tokens=800, temperature=0.1)
                return response.content[0].text
        except Exception as e:
            logger.error(f"AI review failed: {e}")
            return f"An error occurred during AI review: {e}"
        return "AI model not available."

# --- Helper Functions ---
def group_files_by_product(uploaded_files):
    """Groups files based on a common prefix in their names."""
    groups = {}
    for file in uploaded_files:
        # Extract a base name, ignoring variants like _black, _purple, dates, etc.
        match = re.match(r'([a-z_]+)', file.name.lower())
        if match:
            base_name = match.group(1).replace('_', ' ').title()
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file)
        else:
            if 'Uncategorized' not in groups:
                groups['Uncategorized'] = []
            groups['Uncategorized'].append(file)
    return groups

def prepare_report_data(results):
    """Converts the nested results dictionary to a flat list for DataFrame creation."""
    report_rows = []
    for product_name, data in results.items():
        for filename, result in data['files'].items():
            if 'error' in result:
                row = {'Product': product_name, 'File Name': filename, 'Status': 'PROCESSING ERROR', 'Details': result['error']}
            else:
                issues = "; ".join(result['validation']['issues'])
                warnings = "; ".join(result['validation']['warnings'])
                details = f"Issues: {issues}. Warnings: {warnings}." if issues or warnings else "All checks passed."
                row = {
                    'Product': product_name,
                    'File Name': filename,
                    'Status': result['validation']['status'],
                    'Details': details,
                    'Document Type': result['doc_info']['type'],
                    'SKU': result['doc_info']['sku']
                }
            report_rows.append(row)
    return report_rows

# --- UI and Main Application ---
def render_ui():
    """Renders the Streamlit user interface."""
    st.markdown("""
    <style>
        .main-header { padding: 2rem 1rem; margin-bottom: 2rem; background: #f0f2f6; border-radius: 10px; text-align: center; }
        .st-emotion-cache-1y4p8pa { padding-top: 2rem; }
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Package Reviewer</h1>
        <p>Upload a complete set of product documents (artwork, manual, shipping mark, etc.) to perform an automated review.</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    render_ui()
    
    if 'results' not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_type, api_key = AIReviewer.get_api_config()
        if api_key:
            st.success(f"AI Review Enabled ({api_type.capitalize()})")
        else:
            st.warning("AI Review Disabled (No API Key)")
        st.info("The Tesseract OCR engine must be installed on the system for this app to function correctly.")
        st.markdown("---")
        st.markdown("### 📤 Upload Documents")
        # --- MODIFIED FILE UPLOADER ---
        uploaded_files = st.file_uploader(
            "Upload all artwork (PDF) and requirements (XLSX, DOCX) files:",
            type=['pdf', 'xlsx', 'docx'],
            accept_multiple_files=True,
            help="The app will automatically group files by product name."
        )

    if uploaded_files:
        if st.button("🚀 Review All Packages", type="primary", use_container_width=True):
            st.session_state.results = {}
            product_groups = group_files_by_product(uploaded_files)
            
            progress_bar = st.progress(0, "Initializing review...")
            total_files = len(uploaded_files)
            files_processed = 0

            for product_name, files in product_groups.items():
                st.session_state.results[product_name] = {'files': {}, 'ai_review': ''}
                product_files_data = {}

                for file in files:
                    files_processed += 1
                    progress_bar.progress(files_processed / total_files, f"Analyzing: {file.name}")
                    
                    # Use the new universal text extractor
                    extraction = DocumentProcessor.extract_text(file, file.name)
                    if not extraction['success']:
                        st.session_state.results[product_name]['files'][file.name] = {'error': extraction['errors'][0]}
                        continue

                    doc_info = DocumentIdentifier.identify(file.name, extraction['text'])
                    validation = DynamicValidator.validate(extraction['text'], doc_info)
                    
                    result_data = {
                        'extraction': extraction, 'doc_info': doc_info, 'validation': validation
                    }
                    st.session_state.results[product_name]['files'][file.name] = result_data
                    product_files_data[file.name] = result_data

                # Perform AI review on the entire product package
                if api_key:
                    st.session_state.results[product_name]['ai_review'] = AIReviewer.get_review(product_files_data, api_type, api_key)

            progress_bar.empty()

    if st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Validation Report")

        report_df = pd.DataFrame(prepare_report_data(st.session_state.results))
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Report to CSV",
            data=csv,
            file_name="vive_health_validation_report.csv",
            mime="text/csv",
        )

        for product_name, data in st.session_state.results.items():
            with st.expander(f"### Product: {product_name}", expanded=True):
                # Overall AI Review at the top
                if data['ai_review']:
                    st.markdown("#### 🤖 AI-Powered Summary")
                    st.info(data['ai_review'])

                # File-by-file analysis
                st.markdown("#### 📄 File-by-File Analysis")
                for filename, result in data['files'].items():
                    if 'error' in result:
                        st.error(f"**{filename}:** Could not process file. Reason: {result['error']}")
                        continue
                    
                    status = result['validation']['status']
                    doc_type = result['doc_info']['type']
                    sku = result['doc_info']['sku']

                    if status == 'PASS':
                        icon = "✅"
                    elif status == 'NEEDS REVIEW':
                        icon = "⚠️"
                    else:
                        icon = "❌"
                    
                    st.markdown(f"**{icon} {filename}** (Type: *{doc_type}* | Method: *{result['extraction']['method']}*) - **Status: {status}**")
                    
                    if result['validation']['issues']:
                        for issue in result['validation']['issues']:
                            st.error(f"- {issue}")
                    if result['validation']['warnings']:
                        for warning in result['validation']['warnings']:
                            st.warning(f"- {warning}")
                    if status == 'PASS' and not result['validation']['warnings']:
                        st.success("- All specific checks passed.")

if __name__ == "__main__":
    main()
