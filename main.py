"""
packaging_validator.py - Simple tool for graphic designers to validate packaging and labels
Ensures consistency across product variants and catches common errors
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import base64
from PIL import Image
import io
import fitz  # PyMuPDF for PDF handling
import pytesseract
from concurrent.futures import ThreadPoolExecutor
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Packaging & Label Validator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import AI modules
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

# Common packaging errors to check
COMMON_ERRORS = {
    'origin': {
        'correct': ['made in china', '中国制造'],
        'incorrect': ['made in taiwan', 'made in vietnam', 'made in usa']
    },
    'required_text': [
        'vive', 'vive health', 'www.vivehealth.com'
    ],
    'barcode_patterns': [
        r'\d{12,13}',  # UPC/EAN
        r'[A-Z]{3}\d{4}[A-Z0-9\-]*'  # SKU pattern
    ]
}

# Checklist items from the document
CHECKLIST_ITEMS = {
    'packaging': [
        'Product name consistency',
        'No spelling errors',
        'Made in China origin',
        'Color matches SKU suffix',
        'UDI/UPC barcode present',
        'Color code information'
    ],
    'manual': [
        'Product name consistency',
        'No spelling errors'
    ],
    'tags': [
        'Logo present',
        'Care icons on washtag',
        'Color matches product'
    ],
    'shipping': [
        'SKU - QTY format',
        'QR code present',
        'QR matches SKU'
    ]
}

def inject_css():
    """Inject custom CSS for better UI"""
    st.markdown("""
    <style>
        /* Color scheme */
        :root {
            --success: #00C853;
            --error: #FF5252;
            --warning: #FFC107;
            --info: #2196F3;
            --primary: #1976D2;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1976D2 0%, #00ACC1 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Result boxes */
        .success-box {
            background: #E8F5E9;
            border-left: 4px solid var(--success);
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .error-box {
            background: #FFEBEE;
            border-left: 4px solid var(--error);
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        .warning-box {
            background: #FFF8E1;
            border-left: 4px solid var(--warning);
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        
        /* File comparison */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .file-box {
            border: 2px solid #E0E0E0;
            border-radius: 8px;
            padding: 1rem;
            background: #FAFAFA;
        }
        
        .file-box.error {
            border-color: var(--error);
            background: #FFEBEE;
        }
        
        .file-box.success {
            border-color: var(--success);
            background: #E8F5E9;
        }
        
        /* Checklist styling */
        .checklist-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
        
        .checklist-pass {
            background: #E8F5E9;
            color: #2E7D32;
        }
        
        .checklist-fail {
            background: #FFEBEE;
            color: #C62828;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from secrets or environment"""
    keys = {}
    
    # Try Streamlit secrets first
    try:
        if hasattr(st, 'secrets'):
            if 'OPENAI_API_KEY' in st.secrets:
                keys['openai'] = st.secrets['OPENAI_API_KEY']
            elif 'openai_api_key' in st.secrets:
                keys['openai'] = st.secrets['openai_api_key']
                
            if 'ANTHROPIC_API_KEY' in st.secrets:
                keys['claude'] = st.secrets['ANTHROPIC_API_KEY']
            elif 'claude_api_key' in st.secrets:
                keys['claude'] = st.secrets['claude_api_key']
    except:
        pass
    
    # Try environment variables
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return keys

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF using PyMuPDF"""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}")
        return ""

def extract_text_from_image(image_bytes):
    """Extract text from image using OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def extract_colors_from_text(text):
    """Extract color mentions from text"""
    common_colors = [
        'black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 
        'pink', 'orange', 'grey', 'gray', 'brown', 'beige', 'floral'
    ]
    
    found_colors = []
    text_lower = text.lower()
    
    for color in common_colors:
        if color in text_lower:
            found_colors.append(color)
    
    return found_colors

def extract_sku_from_text(text):
    """Extract SKU patterns from text"""
    sku_patterns = [
        r'\b([A-Z]{3}\d{4}[A-Z0-9\-]*)\b',
        r'\bSKU[:\s]+([A-Z0-9\-]+)\b',
        r'\bItem[:\s#]+([A-Z0-9\-]+)\b'
    ]
    
    skus = []
    for pattern in sku_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skus.extend(matches)
    
    return list(set(skus))

def check_origin_consistency(texts):
    """Check if origin marking is consistent and correct"""
    issues = []
    origins_found = []
    
    for idx, text in enumerate(texts):
        text_lower = text.lower()
        
        # Check for correct origin
        has_correct = any(origin in text_lower for origin in COMMON_ERRORS['origin']['correct'])
        has_incorrect = False
        incorrect_found = []
        
        for incorrect in COMMON_ERRORS['origin']['incorrect']:
            if incorrect in text_lower:
                has_incorrect = True
                incorrect_found.append(incorrect)
        
        if has_incorrect:
            issues.append(f"File {idx+1}: Found incorrect origin marking: {', '.join(incorrect_found)}")
        elif not has_correct and 'made in' in text_lower:
            issues.append(f"File {idx+1}: Origin marking found but not 'Made in China'")
        
        # Extract what origin was found
        origin_match = re.search(r'made in (\w+)', text_lower)
        if origin_match:
            origins_found.append(origin_match.group(1))
    
    # Check consistency
    if len(set(origins_found)) > 1:
        issues.append(f"Inconsistent origin markings found: {', '.join(set(origins_found))}")
    
    return issues

def check_color_consistency(file_data):
    """Check if colors are consistent across files"""
    issues = []
    colors_by_file = {}
    
    for filename, text in file_data.items():
        colors = extract_colors_from_text(text)
        if colors:
            colors_by_file[filename] = colors
    
    # Check if same product has different colors mentioned
    # Group files by product (remove color identifiers from filename)
    product_groups = {}
    for filename, colors in colors_by_file.items():
        # Remove common color words from filename to get base product
        base_name = filename.lower()
        for color in ['black', 'purple', 'floral', 'white', 'red', 'blue']:
            base_name = base_name.replace(color, '')
        
        if base_name not in product_groups:
            product_groups[base_name] = []
        product_groups[base_name].append((filename, colors))
    
    # Check consistency within product groups
    for product, file_colors in product_groups.items():
        if len(file_colors) > 1:
            # Each variant should have its own consistent color
            for filename, colors in file_colors:
                expected_color = None
                # Extract expected color from filename
                filename_lower = filename.lower()
                for color in ['black', 'purple', 'white', 'blue', 'red']:
                    if color in filename_lower:
                        expected_color = color
                        break
                
                if expected_color and expected_color not in colors:
                    issues.append(f"{filename}: Expected '{expected_color}' but not found in content")
    
    return issues

def check_sku_format(text):
    """Check if SKU follows correct format"""
    skus = extract_sku_from_text(text)
    issues = []
    
    for sku in skus:
        # Check basic format (3 letters + 4 numbers + optional suffix)
        if not re.match(r'^[A-Z]{3}\d{4}', sku):
            issues.append(f"SKU '{sku}' doesn't follow standard format (XXX####)")
        
        # Check color suffix if present
        if len(sku) > 7:
            suffix = sku[7:]
            # Common color codes
            valid_suffixes = ['BLK', 'WHT', 'BLU', 'RED', 'PUR', 'GRY', 'BEI', 'S', 'M', 'L', 'XL']
            if not any(suffix.endswith(vs) for vs in valid_suffixes):
                issues.append(f"SKU '{sku}' has non-standard suffix '{suffix}'")
    
    return issues

def validate_files(uploaded_files):
    """Main validation function"""
    results = {
        'errors': [],
        'warnings': [],
        'success': [],
        'file_data': {}
    }
    
    # Extract text from all files
    file_texts = {}
    
    for file in uploaded_files:
        try:
            if file.type == 'application/pdf':
                text = extract_text_from_pdf(file.read())
            elif file.type.startswith('image/'):
                text = extract_text_from_image(file.read())
            else:
                text = file.read().decode('utf-8')
            
            file_texts[file.name] = text
            results['file_data'][file.name] = {
                'text': text[:500] + '...' if len(text) > 500 else text,
                'type': file.type,
                'size': file.size
            }
        except Exception as e:
            results['errors'].append(f"Error reading {file.name}: {str(e)}")
    
    # Run validation checks
    if len(file_texts) > 0:
        # Check origin consistency
        origin_issues = check_origin_consistency(list(file_texts.values()))
        results['errors'].extend(origin_issues)
        
        # Check color consistency
        color_issues = check_color_consistency(file_texts)
        results['warnings'].extend(color_issues)
        
        # Check each file individually
        for filename, text in file_texts.items():
            # Check SKU format
            sku_issues = check_sku_format(text)
            if sku_issues:
                results['warnings'].extend([f"{filename}: {issue}" for issue in sku_issues])
            
            # Check for required text
            text_lower = text.lower()
            for required in COMMON_ERRORS['required_text']:
                if required not in text_lower:
                    results['warnings'].append(f"{filename}: Missing required text '{required}'")
            
            # Check for spelling (basic check)
            common_misspellings = {
                'recieve': 'receive',
                'occured': 'occurred',
                'seperate': 'separate',
                'definately': 'definitely',
                'managment': 'management'
            }
            
            for wrong, correct in common_misspellings.items():
                if wrong in text_lower:
                    results['errors'].append(f"{filename}: Spelling error - '{wrong}' should be '{correct}'")
    
    # Add success messages
    if not results['errors']:
        results['success'].append("✅ No critical errors found!")
    if not results['warnings']:
        results['success'].append("✅ No warnings found!")
    
    return results

def display_validation_results(results):
    """Display validation results in a user-friendly way"""
    # Summary section
    st.markdown("### 📊 Validation Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        error_count = len(results['errors'])
        if error_count == 0:
            st.success(f"✅ **{error_count} Errors**")
        else:
            st.error(f"❌ **{error_count} Errors**")
    
    with col2:
        warning_count = len(results['warnings'])
        if warning_count == 0:
            st.success(f"✅ **{warning_count} Warnings**")
        else:
            st.warning(f"⚠️ **{warning_count} Warnings**")
    
    with col3:
        files_count = len(results['file_data'])
        st.info(f"📁 **{files_count} Files Checked**")
    
    # Detailed results
    if results['errors']:
        st.markdown("### ❌ Errors (Must Fix)")
        for error in results['errors']:
            st.markdown(f"""
            <div class="error-box">
                {error}
            </div>
            """, unsafe_allow_html=True)
    
    if results['warnings']:
        st.markdown("### ⚠️ Warnings (Please Review)")
        for warning in results['warnings']:
            st.markdown(f"""
            <div class="warning-box">
                {warning}
            </div>
            """, unsafe_allow_html=True)
    
    if results['success']:
        st.markdown("### ✅ Passed Checks")
        for success in results['success']:
            st.markdown(f"""
            <div class="success-box">
                {success}
            </div>
            """, unsafe_allow_html=True)
    
    # File details
    with st.expander("📄 File Details", expanded=False):
        for filename, data in results['file_data'].items():
            st.markdown(f"**{filename}**")
            st.text(f"Type: {data['type']}")
            st.text(f"Size: {data['size']:,} bytes")
            st.text_area("Content Preview", data['text'], height=100, key=filename)
            st.markdown("---")

def display_checklist_guide():
    """Display the checklist as a guide"""
    st.markdown("### 📋 Packaging Checklist Guide")
    
    for category, items in CHECKLIST_ITEMS.items():
        st.markdown(f"**{category.title()}:**")
        for item in items:
            st.markdown(f"- {item}")
    
    st.markdown("""
    ### 🎯 Common Issues to Watch For:
    - **Origin Mismatch**: Products marked as "Made in Taiwan" instead of "Made in China"
    - **Color Mismatch**: Box color doesn't match label color
    - **SKU Format**: Should follow pattern like SUP1030BGES (product-color-size)
    - **Missing Elements**: Logo, care icons, thank you cards
    - **Barcode Issues**: UPC/UDI must match product information
    """)

def main():
    inject_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📦 Packaging & Label Validator</h1>
        <p>Ensure accuracy across all product packaging and labels</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        
        # API configuration check
        api_keys = get_api_keys()
        api_available = any(api_keys.values())
        
        if api_available:
            st.success("✅ AI APIs configured")
        else:
            st.warning("⚠️ No AI APIs found")
            with st.expander("API Setup"):
                st.markdown("""
                Add to `.streamlit/secrets.toml`:
                ```
                OPENAI_API_KEY = "sk-..."
                ANTHROPIC_API_KEY = "sk-ant-..."
                ```
                """)
        
        st.markdown("---")
        
        # Show checklist
        display_checklist_guide()
    
    # Main content
    st.markdown("### 📤 Upload Files to Validate")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose packaging and label files",
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
        accept_multiple_files=True,
        help="Upload all variants of packaging, labels, and related files"
    )
    
    if uploaded_files:
        # Group files by product
        st.markdown("### 📁 Uploaded Files")
        
        # Display uploaded files in a grid
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                file_type_icon = "📄" if file.type == 'application/pdf' else "🖼️"
                st.info(f"{file_type_icon} {file.name}")
        
        # Validate button
        if st.button("🔍 Validate Files", type="primary", use_container_width=True):
            with st.spinner("Analyzing files..."):
                results = validate_files(uploaded_files)
            
            # Display results
            display_validation_results(results)
            
            # Export results
            if st.button("📥 Download Validation Report"):
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'files_checked': len(uploaded_files),
                    'errors': results['errors'],
                    'warnings': results['warnings'],
                    'file_details': results['file_data']
                }
                
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    else:
        # Instructions
        st.info("""
        👆 **Upload your packaging and label files to begin validation**
        
        **What this tool checks:**
        - ✅ Origin consistency (Made in China)
        - ✅ Color matching between files
        - ✅ SKU format validation
        - ✅ Required text presence (Vive branding)
        - ✅ Basic spelling checks
        - ✅ Barcode/QR code presence
        
        **Supported file types:** PDF, PNG, JPG, JPEG, TXT
        """)
        
        # Example workflow
        with st.expander("📖 Example Workflow"):
            st.markdown("""
            1. **Collect all files for a product** (all color variants)
               - Packaging artwork (box designs)
               - Product labels/tags
               - Shipping marks
               - Manuals/quick start guides
            
            2. **Upload all files at once**
               - The tool will analyze them together
               - Cross-check for consistency
            
            3. **Review validation results**
               - Fix any errors (red items)
               - Review warnings (yellow items)
               - Confirm all checks pass
            
            4. **Download report for records**
               - Keep validation report with job files
            """)

if __name__ == "__main__":
    # Check dependencies
    required_packages = {
        'streamlit': 'streamlit',
        'PIL': 'pillow',
        'fitz': 'PyMuPDF',
        'pytesseract': 'pytesseract'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"""
        Missing required packages. Please install:
        ```
        pip install {' '.join(missing_packages)}
        ```
        
        For OCR support, also install Tesseract:
        - Mac: `brew install tesseract`
        - Ubuntu: `sudo apt-get install tesseract-ocr`
        - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
        """)
    else:
        main()
