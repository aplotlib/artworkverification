"""
PRODUCTION-READY Packaging Validator for Vive Health
Bulletproof PDF reading and validation
"""

import streamlit as st
import PyPDF2
import re
from datetime import datetime
import json
import logging
import traceback
from io import BytesIO

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Vive Health Packaging Validator",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}

# Try to import optional libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("pdfplumber available")
except:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("PyMuPDF available")
except:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except:
    CLAUDE_AVAILABLE = False

class PDFExtractor:
    """Production-grade PDF text extraction with multiple fallbacks"""
    
    @staticmethod
    def extract_text(file_buffer, filename):
        """Extract text using all available methods"""
        results = {
            'filename': filename,
            'success': False,
            'text': '',
            'method': 'none',
            'page_count': 0,
            'errors': []
        }
        
        # Ensure we're at the start of the file
        file_buffer.seek(0)
        
        # Method 1: PyMuPDF (best for complex PDFs)
        if PYMUPDF_AVAILABLE:
            try:
                logger.info(f"Trying PyMuPDF for {filename}")
                import fitz
                
                file_buffer.seek(0)
                pdf_bytes = file_buffer.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                
                text = ""
                results['page_count'] = len(doc)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text += f"\n=== Page {page_num + 1} ===\n{page_text}"
                
                doc.close()
                
                if text.strip():
                    results['text'] = text.strip()
                    results['method'] = 'PyMuPDF'
                    results['success'] = True
                    logger.info(f"PyMuPDF extracted {len(text)} chars from {filename}")
                    return results
                    
            except Exception as e:
                results['errors'].append(f"PyMuPDF: {str(e)}")
                logger.warning(f"PyMuPDF failed for {filename}: {e}")
        
        # Method 2: pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                logger.info(f"Trying pdfplumber for {filename}")
                file_buffer.seek(0)
                
                with pdfplumber.open(file_buffer) as pdf:
                    text = ""
                    results['page_count'] = len(pdf.pages)
                    
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n=== Page {i + 1} ===\n{page_text}"
                    
                    if text.strip():
                        results['text'] = text.strip()
                        results['method'] = 'pdfplumber'
                        results['success'] = True
                        logger.info(f"pdfplumber extracted {len(text)} chars from {filename}")
                        return results
                        
            except Exception as e:
                results['errors'].append(f"pdfplumber: {str(e)}")
                logger.warning(f"pdfplumber failed for {filename}: {e}")
        
        # Method 3: PyPDF2 (always available)
        try:
            logger.info(f"Trying PyPDF2 for {filename}")
            file_buffer.seek(0)
            
            reader = PyPDF2.PdfReader(file_buffer)
            text = ""
            results['page_count'] = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n=== Page {i + 1} ===\n{page_text}"
                except Exception as e:
                    results['errors'].append(f"PyPDF2 page {i+1}: {str(e)}")
            
            if text.strip():
                results['text'] = text.strip()
                results['method'] = 'PyPDF2'
                results['success'] = True
                logger.info(f"PyPDF2 extracted {len(text)} chars from {filename}")
                return results
                
        except Exception as e:
            results['errors'].append(f"PyPDF2: {str(e)}")
            logger.error(f"PyPDF2 failed for {filename}: {e}")
        
        # Method 4: Raw text search in PDF bytes
        try:
            logger.info(f"Trying raw text extraction for {filename}")
            file_buffer.seek(0)
            pdf_content = file_buffer.read()
            
            # Look for text between stream markers
            text_pattern = rb'stream\s*\n(.*?)\nendstream'
            matches = re.findall(text_pattern, pdf_content, re.DOTALL)
            
            decoded_text = []
            for match in matches:
                try:
                    # Try to decode as UTF-8
                    decoded = match.decode('utf-8', errors='ignore')
                    # Keep only printable characters
                    clean = ''.join(c for c in decoded if c.isprintable() or c.isspace())
                    if len(clean) > 10:  # Minimum meaningful text
                        decoded_text.append(clean)
                except:
                    pass
            
            if decoded_text:
                results['text'] = '\n'.join(decoded_text)
                results['method'] = 'raw_extraction'
                results['success'] = True
                logger.info(f"Raw extraction got {len(results['text'])} chars from {filename}")
                
        except Exception as e:
            results['errors'].append(f"Raw extraction: {str(e)}")
            logger.error(f"Raw extraction failed for {filename}: {e}")
        
        return results

class DocumentIdentifier:
    """Identify document type and extract key information"""
    
    @staticmethod
    def identify(filename, text):
        """Identify document type and extract metadata"""
        info = {
            'type': 'Unknown',
            'subtype': '',
            'confidence': 0,
            'detected_elements': [],
            'product_hints': []
        }
        
        filename_lower = filename.lower()
        text_lower = text.lower() if text else ""
        
        # Document type detection rules
        type_rules = [
            # Wash tags / Care labels
            {
                'keywords': ['wash', 'tag', 'care', 'label', 'washtag'],
                'content': ['polyester', 'machine wash', 'wash cold', 'tumble dry', 'material'],
                'type': 'Wash Tag / Care Label'
            },
            # Packaging artwork
            {
                'keywords': ['packaging', 'package', 'artwork', 'box'],
                'content': ['wheelchair bag', 'vive health', 'distributed by'],
                'type': 'Packaging Artwork'
            },
            # Quick start guides
            {
                'keywords': ['quick', 'start', 'guide', 'qsg'],
                'content': ['instructions', 'step 1', 'application', 'how to'],
                'type': 'Quick Start Guide'
            },
            # Shipping marks
            {
                'keywords': ['shipping', 'ship', 'mark'],
                'content': ['shipping mark', 'carton', 'qty'],
                'type': 'Shipping Mark'
            },
            # Manuals
            {
                'keywords': ['manual', 'instruction', 'user guide'],
                'content': ['table of contents', 'chapter', 'warranty'],
                'type': 'User Manual'
            }
        ]
        
        # Check each rule
        best_score = 0
        for rule in type_rules:
            score = 0
            
            # Check filename
            for keyword in rule['keywords']:
                if keyword in filename_lower:
                    score += 50
            
            # Check content
            for keyword in rule['content']:
                if keyword in text_lower:
                    score += 30
            
            if score > best_score:
                best_score = score
                info['type'] = rule['type']
                info['confidence'] = score
        
        # Extract key elements
        if text:
            # Check for Made in China
            if 'made in china' in text_lower:
                info['detected_elements'].append('Made in China')
            
            # Check for Vive branding
            if 'vive' in text_lower:
                info['detected_elements'].append('Vive Branding')
                if 'vive®' in text or 'vive®' in text_lower:
                    info['detected_elements'].append('Vive® Trademark')
            
            # Check for website
            websites = re.findall(r'(?:www\.)?vivehealth\.com|vhealth\.link/\w+', text_lower)
            if websites:
                info['detected_elements'].append(f'Website: {websites[0]}')
            
            # Extract SKUs
            sku_patterns = [
                r'LVA\d{4}[A-Z]{0,3}',
                r'SUP\d{4}[A-Z]{0,3}',
                r'MOB\d{4}[A-Z]{0,3}',
                r'[A-Z]{3}\d{4}[A-Z]{0,3}'
            ]
            
            for pattern in sku_patterns:
                skus = re.findall(pattern, text.upper())
                if skus:
                    info['detected_elements'].append(f'SKU: {skus[0]}')
                    break
            
            # Detect product hints
            if 'wheelchair bag' in text_lower:
                info['product_hints'].append('Wheelchair Bag')
                if 'advanced' in text_lower:
                    info['product_hints'].append('Wheelchair Bag Advanced')
            
            # Detect color variants
            colors = ['black', 'purple', 'blue', 'red', 'floral']
            for color in colors:
                if color in text_lower:
                    info['product_hints'].append(f'Color: {color.title()}')
        
        return info

class Validator:
    """Validate documents against Vive Health requirements"""
    
    @staticmethod
    def validate(filename, text, doc_info):
        """Comprehensive validation"""
        validation = {
            'filename': filename,
            'status': 'FAIL',
            'score': 0,
            'checks': {},
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        if not text:
            validation['issues'].append('No text could be extracted from PDF')
            validation['suggestions'].append('Ensure PDF contains searchable text, not just images')
            return validation
        
        text_lower = text.lower()
        
        # Universal requirements (all documents)
        universal_checks = {
            'made_in_china': {
                'check': 'made in china' in text_lower,
                'issue': 'Missing "Made in China" text',
                'critical': True
            },
            'vive_branding': {
                'check': 'vive' in text_lower,
                'issue': 'Missing Vive branding',
                'critical': True
            },
            'website': {
                'check': 'vivehealth.com' in text_lower or 'vhealth.link' in text_lower,
                'issue': 'Missing website URL',
                'critical': False
            },
            'contact_info': {
                'check': bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}|service@vivehealth\.com', text_lower)),
                'issue': 'Missing contact information',
                'critical': False
            }
        }
        
        # Check universal requirements
        for check_name, check_data in universal_checks.items():
            validation['checks'][check_name] = check_data['check']
            if not check_data['check']:
                if check_data['critical']:
                    validation['issues'].append(check_data['issue'])
                else:
                    validation['warnings'].append(check_data['issue'])
            else:
                validation['score'] += 20
        
        # Document-specific validation
        doc_type = doc_info.get('type', 'Unknown')
        
        if doc_type == 'Wash Tag / Care Label':
            # Check materials
            if any(x in text_lower for x in ['polyester', 'cotton', 'nylon', 'pvc', 'ldpe']):
                validation['checks']['materials'] = True
                validation['score'] += 10
            else:
                validation['warnings'].append('No material composition found')
            
            # Check care instructions
            if any(x in text_lower for x in ['machine wash', 'hand wash', 'dry clean']):
                validation['checks']['care_instructions'] = True
                validation['score'] += 10
            else:
                validation['issues'].append('Missing care instructions')
        
        elif doc_type == 'Packaging Artwork':
            # Check for SKU
            if any('SKU:' in elem for elem in doc_info.get('detected_elements', [])):
                validation['checks']['sku_visible'] = True
                validation['score'] += 10
            else:
                validation['warnings'].append('SKU not clearly visible')
            
            # Check for product name
            if doc_info.get('product_hints'):
                validation['checks']['product_identified'] = True
                validation['score'] += 10
        
        elif doc_type == 'Quick Start Guide':
            # Check for instructions
            if any(x in text_lower for x in ['step', 'instruction', 'attach', 'install']):
                validation['checks']['has_instructions'] = True
                validation['score'] += 10
            else:
                validation['issues'].append('No clear instructions found')
        
        # Calculate final status
        critical_issues = [iss for iss in validation['issues'] if any(
            crit in iss for crit in ['Made in China', 'Vive branding']
        )]
        
        if validation['score'] >= 80 and not critical_issues:
            validation['status'] = 'PASS'
        elif validation['score'] >= 50 or (validation['score'] >= 30 and not critical_issues):
            validation['status'] = 'NEEDS REVIEW'
        else:
            validation['status'] = 'FAIL'
        
        # Add suggestions based on issues
        if 'Made in China' in ' '.join(validation['issues']):
            validation['suggestions'].append('Add "Made in China" text to the document')
        
        if 'Vive branding' in ' '.join(validation['issues']):
            validation['suggestions'].append('Include Vive or vive® branding prominently')
        
        return validation

def get_api_config():
    """Get API configuration"""
    api_key = None
    api_type = None
    
    if hasattr(st, 'secrets'):
        if 'OPENAI_API_KEY' in st.secrets and OPENAI_AVAILABLE:
            api_key = st.secrets['OPENAI_API_KEY']
            api_type = 'openai'
        elif 'ANTHROPIC_API_KEY' in st.secrets and CLAUDE_AVAILABLE:
            api_key = st.secrets['ANTHROPIC_API_KEY']
            api_type = 'claude'
    
    return api_type, api_key

def get_ai_review(text, doc_info, validation_result, api_type, api_key):
    """Get AI review of the document"""
    if not api_key or not text:
        return None
    
    prompt = f"""Review this Vive Health packaging document:

Document Type: {doc_info.get('type')}
Detected Elements: {', '.join(doc_info.get('detected_elements', []))}
Validation Status: {validation_result.get('status')}
Issues Found: {', '.join(validation_result.get('issues', []))}

Document excerpt:
{text[:1000]}

Provide:
1. Confirm what product this is for
2. List any critical compliance issues
3. Suggest specific improvements
Keep response under 200 words."""

    try:
        if api_type == 'openai' and OPENAI_AVAILABLE:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        
        elif api_type == 'claude' and CLAUDE_AVAILABLE:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            return response.content[0].text
            
    except Exception as e:
        logger.error(f"AI review failed: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .status-pass { background-color: #d4edda; color: #155724; padding: 0.5rem; border-radius: 5px; }
    .status-fail { background-color: #f8d7da; color: #721c24; padding: 0.5rem; border-radius: 5px; }
    .status-review { background-color: #fff3cd; color: #856404; padding: 0.5rem; border-radius: 5px; }
    </style>
    
    <div class="main-header">
        <h1>🏥 Vive Health Packaging Validator</h1>
        <p>Production-Ready Document Validation System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get API config
    api_type, api_key = get_api_config()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # PDF readers
        readers = ["✅ PyPDF2 (primary)"]
        if PDFPLUMBER_AVAILABLE:
            readers.append("✅ pdfplumber")
        if PYMUPDF_AVAILABLE:
            readers.append("✅ PyMuPDF")
        
        st.success("PDF Readers:\n" + "\n".join(readers))
        
        # AI status
        if api_key:
            st.success(f"✅ AI Review ({api_type})")
        else:
            st.info("ℹ️ AI Review not configured")
        
        st.markdown("---")
        st.markdown("### 📋 Validation Checks")
        st.markdown("""
        **Critical Requirements:**
        - Made in China
        - Vive branding
        
        **Standard Requirements:**
        - Website URL
        - Contact information
        - Product identification
        
        **Document-Specific:**
        - Materials (wash tags)
        - Care instructions
        - SKU visibility
        - Setup instructions
        """)
    
    # Main content
    st.markdown("### 📤 Upload Packaging Documents")
    
    uploaded_files = st.file_uploader(
        "Select PDF files for validation",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload any Vive Health packaging, labels, or documentation"
    )
    
    if uploaded_files:
        if st.button("🚀 Validate All Documents", type="primary", use_container_width=True):
            
            # Process each file
            for file in uploaded_files:
                st.markdown("---")
                
                # Create columns for real-time status
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 📄 {file.name}")
                
                with col2:
                    status_placeholder = st.empty()
                    status_placeholder.info("⏳ Processing...")
                
                # Extract text
                with st.spinner(f"Reading {file.name}..."):
                    extraction_result = PDFExtractor.extract_text(file, file.name)
                
                # Show extraction results
                if extraction_result['success']:
                    st.success(f"✅ Text extracted successfully using {extraction_result['method']} ({len(extraction_result['text'])} characters)")
                    
                    # Show text preview
                    with st.expander("📝 View Extracted Text"):
                        st.text_area("Text Preview", extraction_result['text'][:1000] + "...", height=200)
                else:
                    st.error("❌ Could not extract text automatically")
                    
                    # Show errors
                    if extraction_result['errors']:
                        with st.expander("🔍 Technical Details"):
                            for error in extraction_result['errors']:
                                st.text(error)
                    
                    # Manual input option
                    st.warning("📝 You can manually input the text content below:")
                    manual_text = st.text_area(
                        "Paste the document text here:",
                        key=f"manual_{file.name}",
                        height=200,
                        placeholder="Copy and paste the text from your PDF here..."
                    )
                    
                    if manual_text:
                        extraction_result['text'] = manual_text
                        extraction_result['success'] = True
                        extraction_result['method'] = 'manual_input'
                        st.success("✅ Manual text input received")
                
                # Only proceed if we have text
                if extraction_result['success'] and extraction_result['text']:
                    
                    # Identify document
                    with st.spinner("Identifying document type..."):
                        doc_info = DocumentIdentifier.identify(file.name, extraction_result['text'])
                    
                    # Document info
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.info(f"📋 Document Type: **{doc_info['type']}**")
                        if doc_info['product_hints']:
                            st.info(f"🏷️ Product: **{', '.join(doc_info['product_hints'])}**")
                    
                    with info_col2:
                        if doc_info['detected_elements']:
                            st.success(f"✅ Found: {', '.join(doc_info['detected_elements'][:3])}")
                    
                    # Validate
                    with st.spinner("Running validation checks..."):
                        validation_result = Validator.validate(
                            file.name, 
                            extraction_result['text'], 
                            doc_info
                        )
                    
                    # Update status
                    if validation_result['status'] == 'PASS':
                        status_placeholder.markdown('<div class="status-pass">✅ PASSED</div>', unsafe_allow_html=True)
                    elif validation_result['status'] == 'FAIL':
                        status_placeholder.markdown('<div class="status-fail">❌ FAILED</div>', unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown('<div class="status-review">⚠️ NEEDS REVIEW</div>', unsafe_allow_html=True)
                    
                    # Show validation results
                    val_col1, val_col2 = st.columns(2)
                    
                    with val_col1:
                        st.markdown("**✅ Checks Passed:**")
                        for check, passed in validation_result['checks'].items():
                            if passed:
                                st.success(f"✓ {check.replace('_', ' ').title()}")
                    
                    with val_col2:
                        if validation_result['issues']:
                            st.markdown("**❌ Critical Issues:**")
                            for issue in validation_result['issues']:
                                st.error(f"✗ {issue}")
                        
                        if validation_result['warnings']:
                            st.markdown("**⚠️ Warnings:**")
                            for warning in validation_result['warnings']:
                                st.warning(f"! {warning}")
                    
                    # AI Review (if available)
                    if api_key:
                        with st.spinner("Getting AI review..."):
                            ai_review = get_ai_review(
                                extraction_result['text'],
                                doc_info,
                                validation_result,
                                api_type,
                                api_key
                            )
                        
                        if ai_review:
                            st.markdown("**🤖 AI Review:**")
                            st.info(ai_review)
                    
                    # Suggestions
                    if validation_result['suggestions']:
                        st.markdown("**💡 Suggestions for Improvement:**")
                        for suggestion in validation_result['suggestions']:
                            st.markdown(f"• {suggestion}")
                    
                    # Store results
                    st.session_state.validation_results[file.name] = {
                        'extraction': extraction_result,
                        'doc_info': doc_info,
                        'validation': validation_result,
                        'ai_review': ai_review if api_key else None
                    }
                
                else:
                    status_placeholder.error("❌ No text available")
                    st.error("Cannot validate without text content. Please provide manual input above.")
            
            # Summary section
            if st.session_state.validation_results:
                st.markdown("---")
                st.markdown("## 📊 Validation Summary")
                
                # Calculate totals
                total = len(st.session_state.validation_results)
                passed = sum(1 for r in st.session_state.validation_results.values() 
                           if r['validation']['status'] == 'PASS')
                failed = sum(1 for r in st.session_state.validation_results.values() 
                           if r['validation']['status'] == 'FAIL')
                review = total - passed - failed
                
                # Display metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Files", total)
                
                with metric_col2:
                    st.metric("Passed", passed)
                
                with metric_col3:
                    st.metric("Failed", failed)
                
                with metric_col4:
                    st.metric("Needs Review", review)
                
                # Export results
                st.markdown("### 💾 Export Results")
                
                # Prepare export data
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total_files': total,
                        'passed': passed,
                        'failed': failed,
                        'needs_review': review
                    },
                    'details': {}
                }
                
                for filename, results in st.session_state.validation_results.items():
                    export_data['details'][filename] = {
                        'document_type': results['doc_info']['type'],
                        'detected_elements': results['doc_info']['detected_elements'],
                        'validation_status': results['validation']['status'],
                        'issues': results['validation']['issues'],
                        'warnings': results['validation']['warnings'],
                        'ai_review': results.get('ai_review', 'N/A')
                    }
                
                # Download button
                st.download_button(
                    label="📥 Download Validation Report (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"vive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
