"""
PRODUCTION-READY Packaging Validator for Vive Health
Bulletproof PDF reading and validation with AI-powered review and a clean, modern interface.
"""

import streamlit as st
import PyPDF2
import re
from datetime import datetime
import json
import logging
from io import BytesIO

# --- Configuration ---
# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vive Health Production Validator",
    page_icon="🏥",
    layout="wide"
)

# --- Library Availability Checks ---
# Try to import optional, high-performance libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("pdfplumber is available for PDF processing.")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Falling back to other methods.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF (fitz) is available for robust PDF processing.")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Falling back to other methods.")

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
class PDFExtractor:
    """Production-grade PDF text extraction with multiple fallbacks."""
    @staticmethod
    def extract_text(file_buffer, filename):
        """
        Extracts text from a PDF using the best available method, from PyMuPDF
        to a raw byte search.
        """
        methods_to_try = []
        if PYMUPDF_AVAILABLE: methods_to_try.append(PDFExtractor._try_pymupdf)
        if PDFPLUMBER_AVAILABLE: methods_to_try.append(PDFExtractor._try_pdfplumber)
        methods_to_try.append(PDFExtractor._try_pypdf2) # Always available fallback
        methods_to_try.append(PDFExtractor._try_raw_extraction) # Last resort

        for method in methods_to_try:
            try:
                file_buffer.seek(0)
                result = method(file_buffer, filename)
                if result['success']:
                    logger.info(f"Successfully extracted text from {filename} using {result['method']}.")
                    return result
            except Exception as e:
                logger.error(f"Method failed for {filename}: {e}")
        
        return {'success': False, 'text': '', 'errors': ['All extraction methods failed.']}

    @staticmethod
    def _try_pymupdf(file_buffer, filename):
        pdf_bytes = file_buffer.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(f"\n=== Page {i+1} ===\n{page.get_text()}" for i, page in enumerate(doc))
        doc.close()
        return {'success': bool(text.strip()), 'text': text, 'method': 'PyMuPDF', 'errors': []}

    @staticmethod
    def _try_pdfplumber(file_buffer, filename):
        with pdfplumber.open(file_buffer) as pdf:
            text = "".join(f"\n=== Page {i+1} ===\n{page.extract_text()}" for i, page in enumerate(pdf.pages) if page.extract_text())
        return {'success': bool(text.strip()), 'text': text, 'method': 'pdfplumber', 'errors': []}

    @staticmethod
    def _try_pypdf2(file_buffer, filename):
        reader = PyPDF2.PdfReader(file_buffer)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text: text += f"\n=== Page {i+1} ===\n{page_text}"
        return {'success': bool(text.strip()), 'text': text, 'method': 'PyPDF2', 'errors': []}
    
    @staticmethod
    def _try_raw_extraction(file_buffer, filename):
        pdf_content = file_buffer.read()
        text_pattern = rb'stream\s*\n(.*?)\nendstream'
        matches = re.findall(text_pattern, pdf_content, re.DOTALL)
        decoded_text = []
        for match in matches:
            try:
                decoded = match.decode('utf-8', errors='ignore')
                clean = ''.join(c for c in decoded if c.isprintable() or c.isspace())
                if len(clean.strip()) > 20: decoded_text.append(clean)
            except: pass
        text = "\n".join(decoded_text)
        return {'success': bool(text.strip()), 'text': text, 'method': 'Raw Bytes', 'errors': []}


class DocumentIdentifier:
    """Identifies document type and extracts key information using rules."""
    @staticmethod
    def identify(filename, text):
        info = {'type': 'Unknown', 'detected_elements': [], 'product_hints': []}
        text_lower = text.lower()
        
        type_rules = {
            'Packaging Artwork': {'keywords': ['packaging', 'package', 'artwork', 'box'], 'content': ['distributed by']},
            'Wash Tag / Care Label': {'keywords': ['wash', 'tag', 'care', 'label'], 'content': ['polyester', 'machine wash']},
            'Quick Start Guide': {'keywords': ['quick', 'start', 'guide', 'qsg'], 'content': ['instructions', 'how to']},
            'Shipping Mark': {'keywords': ['shipping', 'ship', 'mark'], 'content': ['carton', 'qty']},
            'User Manual': {'keywords': ['manual', 'instruction'], 'content': ['warranty', 'table of contents']}
        }
        
        # Simple keyword-based type detection
        for doc_type, rules in type_rules.items():
            if any(k in filename.lower() for k in rules['keywords']) or any(c in text_lower for c in rules['content']):
                info['type'] = doc_type
                break

        # Extract specific elements
        if 'made in china' in text_lower: info['detected_elements'].append('Made in China')
        if 'vive' in text_lower: info['detected_elements'].append('Vive Branding')
        
        websites = re.findall(r'vivehealth\.com|vhealth\.link/\w+', text_lower)
        if websites: info['detected_elements'].append(f"Website: {websites[0]}")

        skus = re.findall(r'[A-Z]{3}\d{4}[A-Z]{0,3}', text.upper())
        if skus: info['detected_elements'].append(f"SKU: {skus[0]}")

        if 'wheelchair bag' in text_lower: info['product_hints'].append('Wheelchair Bag')
        
        return info


class Validator:
    """Validates documents against Vive Health requirements."""
    @staticmethod
    def validate(text, doc_info):
        results = {'status': 'FAIL', 'issues': [], 'warnings': []}
        if not text:
            results['issues'].append('No text could be extracted for validation.')
            return results

        text_lower = text.lower()
        # Define checks: (condition, issue_message, is_critical)
        checks = [
            ('made in china' in text_lower, 'Missing "Made in China" text', True),
            ('vive' in text_lower, 'Missing Vive branding', True),
            ('vivehealth.com' in text_lower or 'vhealth.link' in text_lower, 'Missing website URL', False),
            (bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}|service@vivehealth\.com', text_lower)), 'Missing contact info', False),
        ]

        # Run checks
        for condition, issue, is_critical in checks:
            if not condition:
                if is_critical: results['issues'].append(issue)
                else: results['warnings'].append(issue)

        # Determine final status
        if not results['issues']:
            results['status'] = 'NEEDS REVIEW'
            if not results['warnings']:
                results['status'] = 'PASS'
        
        return results

class AIReviewer:
    """Handles interaction with AI models for analysis."""
    @staticmethod
    def get_api_config():
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE:
                return 'openai', st.secrets['OPENAI_API_KEY']
            if 'ANTHROPIC_API_KEY' in st.secrets and CLAUDE_AVAILABLE:
                return 'claude', st.secrets['ANTHROPIC_API_KEY']
        return None, None

    @staticmethod
    def get_review(text, doc_info, validation_result, api_type, api_key):
        if not api_key or not text: return "AI review not available."

        prompt = f"""As a Vive Health QC expert, briefly review this document.
- Document Type: {doc_info.get('type')}
- Key Findings: {', '.join(doc_info.get('detected_elements', ['None']))}
- Initial Status: {validation_result.get('status')}
- Issues: {', '.join(validation_result.get('issues', ['None']))}

Document Excerpt:
---
{text[:1500]}
---
Based on the text, provide a 1-2 sentence summary confirming the product, a bulleted list of critical issues you see (if any), and one key suggestion for improvement."""
        
        try:
            if api_type == 'openai':
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, temperature=0.2
                )
                return response.choices[0].message.content
            elif api_type == 'claude':
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3.5-sonnet-20240620", # Using the latest model
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, temperature=0.2
                )
                return response.content[0].text
        except Exception as e:
            logger.error(f"AI review failed: {e}")
            return f"An error occurred during AI review: {e}"
        return "AI model not available."


# --- UI and Main Application ---
def render_ui():
    """Renders the Streamlit user interface."""
    st.markdown("""
    <style>
        .main-header { background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem; text-align: center; }
        .status-pass { border-left: 5px solid #28a745; padding: 1rem; background-color: #f0fff4; border-radius: 5px; margin-bottom: 1rem; }
        .status-fail { border-left: 5px solid #dc3545; padding: 1rem; background-color: #fff5f5; border-radius: 5px; margin-bottom: 1rem; }
        .status-review { border-left: 5px solid #ffc107; padding: 1rem; background-color: #fffbeb; border-radius: 5px; margin-bottom: 1rem; }
    </style>
    <div class="main-header">
        <h1>🏥 Vive Health Production Validator</h1>
        <p>Upload documents for automated compliance and AI-powered review.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🔧 System Status")
        readers = ["PyPDF2 (Fallback)"]
        if PDFPLUMBER_AVAILABLE: readers.insert(0, "pdfplumber")
        if PYMUPDF_AVAILABLE: readers.insert(0, "PyMuPDF (Preferred)")
        st.success("PDF Readers Available:\n" + "\n".join(f"- {r}" for r in readers))
        
        api_type, api_key = AIReviewer.get_api_config()
        if api_key: st.success(f"AI Review Enabled ({api_type.capitalize()})")
        else: st.warning("AI Review Disabled (No API Key)")

        st.markdown("---")
        st.info("Upload one or more PDFs, then click 'Validate All' to begin the analysis.")

def main():
    render_ui()
    
    # Initialize session state for storing results between runs
    if 'results' not in st.session_state:
        st.session_state.results = {}

    st.markdown("### 📤 Upload Packaging Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files for validation", type=['pdf'], accept_multiple_files=True,
        help="Upload any Vive Health packaging, labels, or documentation"
    )

    if uploaded_files:
        if st.button("🚀 Validate All Documents", type="primary", use_container_width=True):
            st.session_state.results = {} # Clear previous results
            api_type, api_key = AIReviewer.get_api_config()
            progress_bar = st.progress(0, "Starting validation...")

            for i, file in enumerate(uploaded_files):
                filename = file.name
                progress_bar.progress((i + 1) / len(uploaded_files), f"Processing: {filename}")
                
                # Process each file and store results
                extraction = PDFExtractor.extract_text(file, filename)
                if not extraction['success']:
                    st.session_state.results[filename] = {'error': extraction['errors'][0]}
                    continue

                doc_info = DocumentIdentifier.identify(filename, extraction['text'])
                validation = Validator.validate(extraction['text'], doc_info)
                ai_review = AIReviewer.get_review(extraction['text'], doc_info, validation, api_type, api_key)
                
                st.session_state.results[filename] = {
                    'extraction': extraction, 'doc_info': doc_info,
                    'validation': validation, 'ai_review': ai_review
                }
            progress_bar.empty()

    if st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Validation Results")

        for filename, result in st.session_state.results.items():
            st.markdown(f"### {filename}")

            if 'error' in result:
                st.error(f"**Could not process file:** {result['error']}")
                continue

            status = result['validation']['status']
            status_class = f"status-{status.lower().replace(' ', '_')}"
            
            with st.container():
                st.markdown(f'<div class="{status_class}">', unsafe_allow_html=True)
                st.subheader(f"Status: {status}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Document Type:** {result['doc_info']['type']}")
                    st.markdown(f"**Extraction Method:** {result['extraction']['method']}")
                    if result['doc_info']['detected_elements']:
                        st.markdown("**Key Elements Found:**")
                        for elem in result['doc_info']['detected_elements']:
                            st.markdown(f"- `{elem}`")
                
                with col2:
                    if result['validation']['issues']:
                        st.error("**Critical Issues:**")
                        for issue in result['validation']['issues']: st.markdown(f"- {issue}")
                    if result['validation']['warnings']:
                        st.warning("**Warnings:**")
                        for warning in result['validation']['warnings']: st.markdown(f"- {warning}")

                if result['ai_review']:
                    with st.expander("🤖 View AI-Powered Review", expanded=(status != 'PASS')):
                        st.info(result['ai_review'])

                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
