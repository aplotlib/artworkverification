"""
Advanced Packaging Validation System for Vive Health
Dynamic, database-free validation with AI chat interface
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import PyPDF2
from collections import defaultdict
import time
import base64
import io
from difflib import SequenceMatcher
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Vive Health Advanced Packaging Validator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import AI libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    logger.warning("Anthropic not available")

# Try to import PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def inject_advanced_css():
    """Enhanced CSS for professional UI"""
    st.markdown("""
    <style>
        /* Main theme */
        .stApp {
            background-color: #f5f7fa;
        }
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .product-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .cross-check-alert {
            background: #fff5f5;
            border: 2px solid #feb2b2;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .user-message {
            background: #e6f2ff;
            margin-left: 2rem;
        }
        .ai-message {
            background: #f0f4f8;
            margin-right: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }
        .status-pass {
            background: #c6f6d5;
            color: #276749;
        }
        .status-fail {
            background: #fed7d7;
            color: #9b2c2c;
        }
        .status-warning {
            background: #fefcbf;
            color: #975a16;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from various sources"""
    keys = {}
    if hasattr(st, 'secrets'):
        for key_name in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
            if key_name in st.secrets:
                keys['openai'] = st.secrets[key_name]
                break
        for key_name in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key']:
            if key_name in st.secrets:
                keys['claude'] = st.secrets[key_name]
                break
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    return {k: v for k, v in keys.items() if v}

class GenericProductDetector:
    """
    Advanced, database-free product detection from filenames and content
    using heuristics.
    """
    @staticmethod
    def detect_product(filename: str, text_content: str = "") -> Dict[str, Any]:
        result = {
            'guessed_product_name': 'Unknown',
            'guessed_sku': 'Not found',
            'guessed_color': 'Not found',
            'file_type': 'Unknown',
            'detection_confidence': 0
        }
        filename_lower = filename.lower()
        text_lower = text_content.lower() if text_content else ""

        # Heuristic 1: Find potential SKUs (e.g., LVAXXXX, strings with numbers and letters)
        sku_patterns = [
            r'\b[A-Z]{2,3}\d{3,4}[A-Z\-]*\b',  # E.g., LVA1004, LVA3100BLK
            r'\b\w{2,}-\d{4,}\b' # E.g., RHB-1001
        ]
        for pattern in sku_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                result['guessed_sku'] = match.group(0).upper()
                result['detection_confidence'] += 40
                break

        # Heuristic 2: Guess product name from filename (remove extensions, separators)
        name_from_file = os.path.splitext(filename)[0]
        name_from_file = re.sub(r'[\-_]', ' ', name_from_file).title()
        # Clean up common terms
        for term in ['Packaging', 'Artwork', 'Manual', 'Qsg', 'Wash Tag', 'Vive Health']:
             name_from_file = name_from_file.replace(term, '')
        result['guessed_product_name'] = name_from_file.strip()
        result['detection_confidence'] += 20


        # Heuristic 3: Detect file type from filename
        file_types = {
            'packaging': ['packaging', 'package', 'box', 'artwork'],
            'washtag': ['wash', 'tag', 'care', 'label'],
            'quickstart': ['quick', 'start', 'guide', 'qsg'],
            'manual': ['manual', 'instruction', 'user'],
            'shipping': ['shipping', 'ship', 'mark'],
            'qc': ['qc', 'quality', 'check']
        }
        for file_type, keywords in file_types.items():
            if any(keyword in filename_lower for keyword in keywords):
                result['file_type'] = file_type
                result['detection_confidence'] += 15
                break

        return result


class CrossFileValidator:
    """Validates consistency across multiple files"""
    def __init__(self):
        self.files_data = []

    def add_file(self, filename: str, product_info: Dict, extracted_text: str):
        """Add a file to the validation set"""
        self.files_data.append({
            'filename': filename,
            'info': product_info,
            'text': extracted_text
        })

    def validate_consistency(self) -> List[Dict]:
        """Check for inconsistencies across files based on detected SKUs"""
        inconsistencies = []
        sku_groups = defaultdict(list)

        # Group files by the first part of their guessed SKU (e.g., LVA1004)
        for file_data in self.files_data:
            sku = file_data['info'].get('guessed_sku', 'N/A')
            sku_base = sku.split('-')[0] if '-' in sku else re.match(r'[A-Z]+\d+', sku, re.IGNORECASE)
            if sku_base:
                sku_groups[sku_base.group(0) if hasattr(sku_base, 'group') else sku_base].append(file_data)

        for sku_base, files in sku_groups.items():
            if len(files) > 1:
                # Check for multiple full SKUs within the same base group
                full_skus = {f['info'].get('guessed_sku') for f in files}
                if len(full_skus) > 1:
                    inconsistencies.append({
                        'type': 'variant_difference',
                        'severity': 'info',
                        'files': [f['filename'] for f in files],
                        'message': f"Multiple product variants detected for base '{sku_base}': {', '.join(full_skus)}",
                        'expected': True
                    })
        return inconsistencies

class AIValidator:
    """Advanced AI validation with context awareness"""
    def __init__(self, api_keys: Dict):
        self.api_keys = api_keys
        self.context = []

    def add_context(self, filename: str, product_info: Dict, text: str):
        """Add file to validation context"""
        self.context.append({
            'filename': filename,
            'product_info': product_info,
            'text_preview': text[:1000] if text else ""
        })

    def create_comprehensive_prompt(self, filename: str, text: str, product_info: Dict) -> str:
        prompt = f"""You are a meticulous quality control expert for Vive Health. You are reviewing a technical document.

CURRENT FILE:
- Filename: {filename}
- Guessed Product Name: {product_info.get('guessed_product_name', 'Unknown')}
- Guessed File Type: {product_info.get('file_type', 'Unknown')}
- Detected SKU: {product_info.get('guessed_sku', 'Not detected')}

EXTRACTED TEXT (first 3000 chars):
---
{text[:3000]}
---

VALIDATION REQUIREMENTS (Be strict):
1.  **Universal Branding:**
    - Must say "Made in China". Check for exact phrasing.
    - Must have Vive Health branding (e.g., "vive®", "Vive Health" logo).
    - Must have a website, preferably "vivehealth.com".
    - Must have contact information (phone number or support email).
2.  **Content & Accuracy:**
    - Check for any obvious spelling or grammatical errors.
    - Ensure information seems logical and complete for the detected file type (e.g., a 'wash tag' should have material composition).
    - If a SKU is present, verify it is formatted correctly (e.g., not garbled).
3.  **Regulatory & Safety:**
    - Flag the presence or absence of any safety warnings (like California Proposition 65).

Provide your response in valid JSON format only, following this structure:
{{
    "overall_assessment": "APPROVED" or "NEEDS_REVISION" or "REVIEW_REQUIRED",
    "sku_validation": {{
        "found": "{product_info.get('guessed_sku', 'Not detected')}",
        "comment": "Provide a brief comment on the SKU's validity or format."
    }},
    "requirements_check": {{
        "made_in_china": {{ "found": true/false, "details": "Quote the exact text if found" }},
        "vive_branding": {{ "found": true/false, "details": "Describe branding found" }},
        "website_correct": {{ "found": true/false, "details": "List URLs found" }},
        "contact_info": {{ "found": true/false, "details": "List contact info found" }}
    }},
    "critical_issues": ["List any issues that absolutely require a revision."],
    "warnings": ["List any potential issues or items that need a second look."],
    "improvement_suggestions": ["Suggest improvements for clarity, branding, or compliance."]
}}
"""
        return prompt

    def validate_with_ai(self, filename: str, text: str, product_info: Dict) -> Dict:
        prompt = self.create_comprehensive_prompt(filename, text, product_info)
        results = {}

        # Use Claude if available
        if 'claude' in self.api_keys and CLAUDE_AVAILABLE:
            try:
                client = anthropic.Anthropic(api_key=self.api_keys['claude'])
                response = client.messages.create(
                    model="claude-3.5-sonnet-20240620",  # FIXED MODEL
                    max_tokens=2000,
                    temperature=0.1,
                    system="You are a quality control expert. Always respond with valid JSON.",
                    messages=[{"role": "user", "content": prompt}]
                )
                results['claude'] = self._parse_response(response.content[0].text)
            except Exception as e:
                logger.error(f"Claude error: {e}")
                results['claude'] = {"error": str(e)}

        # Use OpenAI if available
        if 'openai' in self.api_keys and OPENAI_AVAILABLE:
            try:
                client = openai.OpenAI(api_key=self.api_keys['openai'])
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a quality control expert. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                results['openai'] = self._parse_response(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
                results['openai'] = {"error": str(e)}

        return results

    def _parse_response(self, response_text: str) -> Dict:
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.error(f"Failed to parse AI JSON response: {e}")
        return {"error": "Failed to parse response", "raw": response_text}

class ChatInterface:
    """Interactive chat interface for Q&A about validation results"""
    def __init__(self, api_keys: Dict):
        self.api_keys = api_keys
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []

    def add_message(self, role: str, content: str):
        st.session_state.chat_messages.append({'role': role, 'content': content})

    def get_context_prompt(self, user_question: str, validation_results: Dict) -> str:
        summary = {
            'total_files': len(validation_results),
            'critical_issue_count': 0,
            'files_with_issues': []
        }
        for filename, results in validation_results.items():
            for provider, ai_result in results.get('ai_results', {}).items():
                if ai_result.get('critical_issues'):
                    summary['critical_issue_count'] += len(ai_result['critical_issues'])
                    summary['files_with_issues'].append(filename)
        summary['files_with_issues'] = list(set(summary['files_with_issues']))

        prompt = f"""You are a helpful QC expert assistant for Vive Health.
VALIDATION SUMMARY:
{json.dumps(summary, indent=2, default=str)}

USER QUESTION: {user_question}

Answer the user's question based on the provided summary and the detailed results (which you should assume you have access to). Be concise, helpful, and suggest actionable next steps. Reference specific filenames if relevant.
"""
        return prompt

    def process_message(self, user_message: str, validation_results: Dict) -> str:
        self.add_message('user', user_message)
        prompt = self.get_context_prompt(user_message, validation_results)
        response = "AI service is unavailable. Please check API keys and configurations."

        if 'claude' in self.api_keys and CLAUDE_AVAILABLE:
            try:
                client = anthropic.Anthropic(api_key=self.api_keys['claude'])
                messages_for_api = [{"role": m['role'], "content": m['content']} for m in st.session_state.chat_messages[-5:]]
                
                ai_response = client.messages.create(
                    model="claude-3.5-sonnet-20240620", # FIXED MODEL
                    max_tokens=1500,
                    temperature=0.7,
                    system="You are a helpful QC expert assistant for Vive Health.",
                    messages=messages_for_api
                )
                response = ai_response.content[0].text
            except Exception as e:
                logger.error(f"Chat error: {e}")
                response = f"🤖 Error: {str(e)}"
        
        self.add_message('assistant', response)
        return response

def extract_text_from_pdf(file_bytes, filename=""):
    """Extract text from PDF with multiple methods"""
    extracted_text = ""
    extraction_method = "none"
    page_count = 0
    try:
        # Try pdfplumber first
        if PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                file_bytes.seek(0)
                with pdfplumber.open(file_bytes) as pdf:
                    page_count = len(pdf.pages)
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                            extraction_method = "pdfplumber"
            except: pass
        # Try PyMuPDF
        if not extracted_text and PYMUPDF_AVAILABLE:
            try:
                import fitz
                file_bytes.seek(0)
                doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
                page_count = len(doc)
                for page in doc:
                    text = page.get_text()
                    if text:
                        extracted_text += text + "\n"
                        extraction_method = "pymupdf"
                doc.close()
            except: pass
        # Fallback to PyPDF2
        if not extracted_text:
            try:
                file_bytes.seek(0)
                reader = PyPDF2.PdfReader(file_bytes)
                page_count = len(reader.pages)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
                        extraction_method = "pypdf2"
            except: pass
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
    return extracted_text.strip(), extraction_method, page_count

def display_validation_dashboard(results: Dict, cross_validator: CrossFileValidator):
    st.markdown("### 📊 Validation Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    total_files = len(results)
    approved = sum(1 for r in results.values() if any(ai.get('overall_assessment') == 'APPROVED' for ai in r.get('ai_results', {}).values()))
    needs_revision = sum(1 for r in results.values() if any(ai.get('overall_assessment') == 'NEEDS_REVISION' for ai in r.get('ai_results', {}).values()))
    inconsistencies = cross_validator.validate_consistency()

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total_files}</div><div>Total Files</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #48bb78;">{approved}</div><div>Approved</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #f56565;">{needs_revision}</div><div>Needs Revision</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: #ed8936;">{len(inconsistencies)}</div><div>Product Groups</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔄 Cross-File Validation")
    if inconsistencies:
        st.info("The following product groups were detected based on SKU prefixes, indicating multiple variants may be present.")
        for issue in inconsistencies:
            st.markdown(f"**{issue['type'].replace('_', ' ').title()}:** {issue['message']}")
    else:
        st.success("✅ All files appear to be for the same product or no overlapping SKU prefixes were found.")

def display_file_results(filename: str, results: Dict):
    product_info = results.get('product_info', {})
    st.markdown(f"""
    <div class="product-card">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div><strong>Guessed Product:</strong><br>{product_info.get('guessed_product_name', 'Unknown')}</div>
            <div><strong>Detected SKU:</strong><br>{product_info.get('guessed_sku', 'Not specified')}</div>
            <div><strong>File Type:</strong><br>{product_info.get('file_type', 'Unknown')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'ai_results' in results:
        st.markdown("#### 🤖 AI Validation Results")
        provider_tabs = st.tabs([provider.title() for provider in results['ai_results'].keys()])
        for idx, (provider, ai_result) in enumerate(results['ai_results'].items()):
            with provider_tabs[idx]:
                if 'error' in ai_result:
                    st.error(f"Error from {provider}: {ai_result['error']}")
                else:
                    assessment = ai_result.get('overall_assessment', 'UNKNOWN')
                    assessment_class = {'APPROVED': 'status-pass', 'NEEDS_REVISION': 'status-fail'}.get(assessment, 'status-warning')
                    st.markdown(f'**Overall Assessment:** <span class="status-badge {assessment_class}">{assessment}</span>', unsafe_allow_html=True)

                    critical = ai_result.get('critical_issues', [])
                    if critical:
                        st.error("**Critical Issues:**")
                        for issue in critical: st.markdown(f"- {issue}")
                    
                    warnings = ai_result.get('warnings', [])
                    if warnings:
                        st.warning("**Warnings:**")
                        for warning in warnings: st.markdown(f"- {warning}")

                    suggestions = ai_result.get('improvement_suggestions', [])
                    if suggestions:
                        st.info("**Improvement Suggestions:**")
                        for suggestion in suggestions: st.markdown(f"- {suggestion}")

                    with st.expander("Show detailed requirements check"):
                        req_check = ai_result.get('requirements_check', {})
                        for req, details in req_check.items():
                            status = "✅ Found" if details.get('found') else "❌ Not Found"
                            st.markdown(f"**{req.replace('_', ' ').title()}:** {status} - *{details.get('details')}*")

    with st.expander("📄 Extraction Details"):
        st.markdown(f"""
        - **Extraction Method:** {results.get('extraction_method', 'Unknown')}
        - **Pages:** {results.get('page_count', 0)}
        - **Text Length:** {len(results.get('text', ''))} characters
        """)
        if results.get('text'):
            st.text_area("Text Preview", results.get('text', '')[:500] + "...", height=150, key=f"preview_{filename}")

def main():
    inject_advanced_css()

    if 'validation_results' not in st.session_state: st.session_state.validation_results = {}
    if 'cross_validator' not in st.session_state: st.session_state.cross_validator = CrossFileValidator()
    if 'ai_validator' not in st.session_state: st.session_state.ai_validator = None
    if 'chat_interface' not in st.session_state: st.session_state.chat_interface = None

    st.markdown("""
    <div class="main-header">
        <h1>🏥 Vive Health Advanced Packaging Validator</h1>
        <p>Dynamic, database-free validation with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)

    api_keys = get_api_keys()

    with st.sidebar:
        st.markdown("### 🔧 System Status")
        if api_keys:
            if 'claude' in api_keys and CLAUDE_AVAILABLE: st.success("✅ Claude AI Ready")
            if 'openai' in api_keys and OPENAI_AVAILABLE: st.success("✅ OpenAI Ready")
            if not st.session_state.ai_validator: st.session_state.ai_validator = AIValidator(api_keys)
            if not st.session_state.chat_interface: st.session_state.chat_interface = ChatInterface(api_keys)
        else:
            st.error("❌ No AI providers configured")
            st.info("Please add API keys to use AI features.")
        st.markdown("---")
        st.markdown("### ❓ Help")
        st.info("""
        1.  **Upload:** Go to the "Upload & Validate" tab and select all related PDFs for a product.
        2.  **Validate:** Click "Validate All" to start the analysis.
        3.  **Review:** Check the "Results Dashboard" for a summary and detailed file-by-file feedback.
        4.  **Inquire:** Use the "AI Assistant" to ask questions about the results.
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Validate", "📊 Results Dashboard", "💬 AI Assistant", "📥 Export"])

    with tab1:
        st.markdown("### Upload Packaging Files")
        st.info("Upload all packaging files for a product (e.g., box art, manual, labels) for comprehensive cross-validation.")
        uploaded_files = st.file_uploader("Select PDF files", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("🚀 Validate All", type="primary", use_container_width=True):
                with st.spinner("Processing files... This may take a moment."):
                    st.session_state.cross_validator = CrossFileValidator()
                    st.session_state.validation_results = {}
                    st.session_state.ai_validator = AIValidator(api_keys) # Re-init AI Validator context

                    progress_bar = st.progress(0, "Initializing...")
                    for idx, file in enumerate(uploaded_files):
                        progress_bar.progress((idx + 1) / len(uploaded_files), f"Analyzing: {file.name}")
                        file.seek(0)
                        text, method, pages = extract_text_from_pdf(file, file.name)
                        product_info = GenericProductDetector.detect_product(file.name, text)
                        
                        results = {'product_info': product_info, 'text': text, 'extraction_method': method, 'page_count': pages}
                        
                        st.session_state.cross_validator.add_file(file.name, product_info, text)
                        st.session_state.ai_validator.add_context(file.name, product_info, text)

                        if text and st.session_state.ai_validator:
                            ai_results = st.session_state.ai_validator.validate_with_ai(file.name, text, product_info)
                            results['ai_results'] = ai_results
                        
                        st.session_state.validation_results[file.name] = results
                st.success("✅ Validation complete!")
                st.balloons()

    with tab2:
        if st.session_state.validation_results:
            display_validation_dashboard(st.session_state.validation_results, st.session_state.cross_validator)
            st.markdown("---")
            st.markdown("### 📁 Individual File Results")
            for filename, results in st.session_state.validation_results.items():
                with st.expander(f"▶️ View results for: **{filename}**"):
                    display_file_results(filename, results)
        else:
            st.info("Upload and validate files in the first tab to see results here.")

    with tab3:
        st.markdown("### 💬 AI Quality Assistant")
        if st.session_state.validation_results and st.session_state.chat_interface:
            st.info("Ask questions about your validation results.")
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if user_input := st.chat_input("e.g., What are the most critical issues found?"):
                st.session_state.chat_interface.process_message(user_input, st.session_state.validation_results)
                st.rerun()
        else:
            st.warning("Please validate files first to enable the AI assistant.")

    with tab4:
        st.markdown("### 📥 Export Options")
        if st.session_state.validation_results:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'company': 'Vive Health',
                'total_files': len(st.session_state.validation_results),
                'cross_validation': st.session_state.cross_validator.validate_consistency(),
                'results': st.session_state.validation_results
            }
            st.download_button(
                "📄 Download Full JSON Report",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"vive_validation_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("Upload and validate files to enable export options.")

if __name__ == "__main__":
    main()
