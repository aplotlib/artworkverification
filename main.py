"""
Vive Health Artwork Verification System - Enhanced Version
Based on ISO 13485 requirements with improved stability and validation
"""

import streamlit as st
import re
import pandas as pd
import logging
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
import base64
import gc
import sys
import os
import tempfile
from contextlib import contextmanager
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vive Health Artwork Verification", 
    page_icon="✅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main-header { 
        background-color: #00466B; 
        color: white; 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        margin-bottom: 30px; 
    }
    .validation-pass {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .validation-fail {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .validation-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .stDataFrame {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants and Configuration ---

# Updated artwork classification based on requirements document
ARTWORK_CLASSIFICATION = {
    'unique_per_variant': {
        'inner_tag': ['inner_tag', 'inner tag', 'innertag'],
        'wash_tag': ['wash_tag', 'wash tag', 'washtag', 'care_tag'],
        'udi_qr_code': ['udi_qr', 'udi_product_qr', 'product_qr', 'qr_code'],
        'shipping_mark': ['shipping_mark', 'shipping mark', 'shipping', 'carton_mark']
    },
    'shared_across_all': {
        'logo_tag': ['logo_tag', 'logo tag', 'logotag'],
        'logo': ['logo', 'brand_logo', 'company_logo'],
        'made_in_china': ['made_in_china', 'china_sticker', 'made_china', 'china sticker'],
        'manual': ['manual', 'instruction', 'guide', 'user_manual'],
        'thank_you_card': ['thank', 'ty_card', 'thank_you', 'thankyou']
    }
}

# File naming patterns for better detection
FILE_NAMING_PATTERNS = {
    'packaging': r'.*(_packaging|_box|_package)\.pdf',
    'manual': r'.*(_manual|_instruction|_guide)\.pdf',
    'inner_tag': r'.*(_inner_tag|innertag)\.pdf',
    'wash_tag': r'.*(_wash_tag|washtag)\.pdf',
    'logo_tag': r'.*(_logo_tag|logotag)\.pdf',
    'logo': r'.*(_logo)\.pdf',
    'thank_you': r'.*(_ty_card|thank)\.pdf',
    'shipping_mark': r'.*(_shipping|shipping_mark)\.pdf'
}

# --- Session State Management ---
def initialize_session_state():
    """Initialize session state variables"""
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    if 'serial_data' not in st.session_state:
        st.session_state.serial_data = {}
    if 'file_dimensions' not in st.session_state:
        st.session_state.file_dimensions = {}

# --- Utility Functions ---
def clean_value(value: Any) -> str:
    """Clean and format values for display, handling NaN and None"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return ""
        return str(value)
    return str(value).strip()

def detect_file_type(filename: str) -> str:
    """Detect file type based on filename patterns"""
    filename_lower = filename.lower()
    
    # Check against file naming patterns
    for file_type, pattern in FILE_NAMING_PATTERNS.items():
        if re.search(pattern, filename_lower):
            return file_type
    
    # Check against classification keywords
    for category in ['unique_per_variant', 'shared_across_all']:
        for file_type, keywords in ARTWORK_CLASSIFICATION[category].items():
            for keyword in keywords:
                if keyword.replace('_', '') in filename_lower.replace('_', '').replace(' ', ''):
                    return file_type
    
    return 'unknown'

def extract_serials_from_text(text: str) -> Dict[str, List[str]]:
    """Extract UPC and UDI serials from text"""
    serials = {
        'upc': [],
        'udi': [],
        'general_12_digit': []
    }
    
    # Pattern for 12-digit numbers
    twelve_digit_pattern = r'\b\d{12}\b'
    
    # Pattern for UDI format (01)XXXXXXXXXXXX(241)
    udi_pattern = r'\(01\)(\d{14})\(241\)'
    
    # Find all 12-digit numbers
    twelve_digit_matches = re.findall(twelve_digit_pattern, text)
    serials['general_12_digit'] = twelve_digit_matches
    
    # Find UDI patterns
    udi_matches = re.findall(udi_pattern, text)
    for match in udi_matches:
        # Extract the 12-digit portion from 14-digit UDI
        if len(match) >= 12:
            serials['udi'].append(match[:12])
    
    # Identify potential UPC codes (12-digit numbers not in UDI format)
    for num in twelve_digit_matches:
        if num not in serials['udi']:
            serials['upc'].append(num)
    
    return serials

def extract_dimensions_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extract dimensions from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count > 0:
            page = doc[0]
            width_in = page.rect.width / 72  # Convert points to inches
            height_in = page.rect.height / 72
            return {
                'width_inches': round(width_in, 2),
                'height_inches': round(height_in, 2),
                'width_points': page.rect.width,
                'height_points': page.rect.height,
                'page_count': doc.page_count
            }
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting dimensions: {e}")
    return {}

# --- OCR and Text Extraction ---
def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF using PyMuPDF and OCR if needed"""
    text = ""
    try:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Try text extraction first
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            else:
                # Fall back to OCR if no text found
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    img = Image.open(BytesIO(img_data))
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
                except Exception as ocr_error:
                    logger.warning(f"OCR failed for page {page_num}: {ocr_error}")
        
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    
    return text

# --- Validation Classes ---
class ArtworkValidator:
    """Main validation class for artwork verification"""
    
    def __init__(self):
        self.validation_results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def validate_file_classification(self, files: List[Dict]) -> Dict[str, Any]:
        """Validate that files are correctly classified as shared or unique"""
        results = {
            'status': 'pass',
            'details': [],
            'summary': {}
        }
        
        # Group files by type
        file_groups = {}
        for file in files:
            file_type = detect_file_type(file.get('name', ''))
            if file_type not in file_groups:
                file_groups[file_type] = []
            file_groups[file_type].append(file)
        
        # Check each file type
        for file_type, file_list in file_groups.items():
            # Determine if should be unique or shared
            is_unique = False
            for unique_type in ARTWORK_CLASSIFICATION['unique_per_variant']:
                if file_type == unique_type:
                    is_unique = True
                    break
            
            if is_unique and len(file_list) == 1:
                results['warnings'].append(f"{file_type} should have separate files per variant but only 1 found")
                results['status'] = 'warning'
            
            results['summary'][file_type] = {
                'count': len(file_list),
                'should_be_unique': is_unique
            }
        
        return results
    
    def validate_upc_udi_match(self, documents: Dict) -> Dict[str, Any]:
        """Validate UPC and UDI serial matching across documents"""
        results = {
            'status': 'pass',
            'serials_found': {},
            'matches': [],
            'mismatches': []
        }
        
        all_serials = {}
        
        # Extract serials from all documents
        for doc_type, doc_data in documents.items():
            if 'text' in doc_data:
                serials = extract_serials_from_text(doc_data['text'])
                if any(serials.values()):
                    all_serials[doc_type] = serials
                    results['serials_found'][doc_type] = serials
        
        # Check for 12-digit matches between UPC and UDI
        upc_numbers = set()
        udi_numbers = set()
        
        for doc_type, serials in all_serials.items():
            upc_numbers.update(serials.get('upc', []))
            udi_numbers.update(serials.get('udi', []))
        
        # Find matching 12-digit portions
        for upc in upc_numbers:
            for udi in udi_numbers:
                if upc == udi[:12]:  # Check if UPC matches first 12 digits of UDI
                    results['matches'].append({
                        'upc': upc,
                        'udi': udi,
                        'match_type': 'exact_12_digit'
                    })
        
        if not results['matches'] and (upc_numbers or udi_numbers):
            results['status'] = 'fail'
            results['mismatches'].append("No matching 12-digit serials found between UPC and UDI")
        
        return results
    
    def validate_all(self, documents: Dict, product_info: Dict = None) -> Dict[str, Any]:
        """Run all validations"""
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'product_info': product_info,
            'validations': {}
        }
        
        # File classification validation
        file_list = [{'name': k, 'data': v} for k, v in documents.items()]
        all_results['validations']['file_classification'] = self.validate_file_classification(file_list)
        
        # UPC/UDI validation
        all_results['validations']['upc_udi_match'] = self.validate_upc_udi_match(documents)
        
        # Overall status
        has_failure = any(
            v.get('status') == 'fail' 
            for v in all_results['validations'].values()
        )
        has_warning = any(
            v.get('status') == 'warning' 
            for v in all_results['validations'].values()
        )
        
        if has_failure:
            all_results['overall_status'] = 'fail'
        elif has_warning:
            all_results['overall_status'] = 'warning'
        else:
            all_results['overall_status'] = 'pass'
        
        return all_results

# --- Report Generation ---
def generate_serial_report(validation_results: Dict) -> pd.DataFrame:
    """Generate a report of serials found per variant"""
    data = []
    
    serials_data = validation_results.get('validations', {}).get('upc_udi_match', {}).get('serials_found', {})
    
    for doc_type, serials in serials_data.items():
        for serial_type, serial_list in serials.items():
            for serial in serial_list:
                data.append({
                    'Document': clean_value(doc_type),
                    'Serial Type': clean_value(serial_type.upper()),
                    'Serial Number': clean_value(serial),
                    'Digits': len(serial) if serial else 0
                })
    
    if data:
        df = pd.DataFrame(data)
        # Clean any NaN values
        df = df.fillna("")
        return df
    else:
        return pd.DataFrame(columns=['Document', 'Serial Type', 'Serial Number', 'Digits'])

def generate_dimension_report(file_dimensions: Dict) -> pd.DataFrame:
    """Generate a report of artwork dimensions"""
    data = []
    
    for filename, dims in file_dimensions.items():
        if dims:
            data.append({
                'File Name': clean_value(filename),
                'Width (inches)': clean_value(dims.get('width_inches', '')),
                'Height (inches)': clean_value(dims.get('height_inches', '')),
                'Page Count': clean_value(dims.get('page_count', ''))
            })
    
    if data:
        df = pd.DataFrame(data)
        df = df.fillna("")
        return df
    else:
        return pd.DataFrame(columns=['File Name', 'Width (inches)', 'Height (inches)', 'Page Count'])

# --- Main Application ---
def main():
    st.markdown('<div class="main-header"><h1>🎨 Vive Health Artwork Verification System</h1></div>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for product information
    with st.sidebar:
        st.header("📋 Product Information")
        
        product_info = {}
        product_info['sku'] = st.text_input("SKU/Product Code", help="Enter the product SKU")
        product_info['product_name'] = st.text_input("Product Name")
        product_info['brand'] = st.selectbox("Brand", ["Vive", "HealthSmart", "Equate", "Other"])
        product_info['variant'] = st.text_input("Variant/Size", help="e.g., Small, Medium, Large")
        
        st.markdown("---")
        st.header("⚙️ Settings")
        enable_ocr = st.checkbox("Enable OCR for text extraction", value=True)
        extract_dimensions = st.checkbox("Extract artwork dimensions", value=True)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📁 Upload Artwork Files")
        
        uploaded_files = st.file_uploader(
            "Choose artwork files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload all artwork files including packaging, tags, manuals, etc."
        )
        
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} files uploaded")
            
            # Process uploaded files
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    try:
                        # Detect file type
                        file_type = detect_file_type(file.name)
                        
                        # Read file content
                        file_content = file.read()
                        
                        # Store in session state
                        st.session_state.documents[file.name] = {
                            'type': file_type,
                            'content': file_content,
                            'size': len(file_content)
                        }
                        
                        # Extract text if PDF and OCR is enabled
                        if file.name.lower().endswith('.pdf') and enable_ocr:
                            text = extract_text_from_pdf(file_content)
                            st.session_state.documents[file.name]['text'] = text
                        
                        # Extract dimensions if enabled
                        if extract_dimensions and file.name.lower().endswith('.pdf'):
                            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                                tmp_file.write(file_content)
                                tmp_file.flush()
                                dims = extract_dimensions_from_pdf(tmp_file.name)
                                st.session_state.file_dimensions[file.name] = dims
                                os.unlink(tmp_file.name)
                        
                        st.success(f"✅ Processed: {file.name} (Type: {file_type})")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                        logger.error(f"Error processing {file.name}: {e}")
    
    with col2:
        st.header("📊 File Status")
        
        if st.session_state.documents:
            # Show file classification
            st.subheader("File Classification")
            
            unique_files = []
            shared_files = []
            unknown_files = []
            
            for filename, doc_data in st.session_state.documents.items():
                file_type = doc_data.get('type', 'unknown')
                
                # Determine classification
                is_unique = False
                is_shared = False
                
                for unique_type in ARTWORK_CLASSIFICATION['unique_per_variant']:
                    if file_type == unique_type:
                        is_unique = True
                        break
                
                for shared_type in ARTWORK_CLASSIFICATION['shared_across_all']:
                    if file_type == shared_type:
                        is_shared = True
                        break
                
                if is_unique:
                    unique_files.append(f"• {filename} ({file_type})")
                elif is_shared:
                    shared_files.append(f"• {filename} ({file_type})")
                else:
                    unknown_files.append(f"• {filename}")
            
            if unique_files:
                st.markdown("**📌 Unique per Variant:**")
                for f in unique_files:
                    st.markdown(f)
            
            if shared_files:
                st.markdown("**🔄 Shared Across All:**")
                for f in shared_files:
                    st.markdown(f)
            
            if unknown_files:
                st.markdown("**❓ Unclassified:**")
                for f in unknown_files:
                    st.markdown(f)
    
    # Validation Section
    st.markdown("---")
    
    if st.session_state.documents:
        if st.button("🔍 Run Validation", type="primary", use_container_width=True):
            with st.spinner("Running validations..."):
                validator = ArtworkValidator()
                validation_results = validator.validate_all(st.session_state.documents, product_info)
                st.session_state.validation_results = validation_results
                
                # Display validation results
                st.header("✅ Validation Results")
                
                # Overall status
                overall_status = validation_results.get('overall_status', 'unknown')
                if overall_status == 'pass':
                    st.success("🎉 All validations passed!")
                elif overall_status == 'warning':
                    st.warning("⚠️ Validation completed with warnings")
                else:
                    st.error("❌ Validation failed - please review issues below")
                
                # File Classification Results
                st.subheader("📁 File Classification Check")
                class_results = validation_results['validations'].get('file_classification', {})
                if class_results.get('status') == 'pass':
                    st.markdown('<div class="validation-pass">✅ File classification is correct</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-warning">⚠️ File classification needs review</div>', 
                               unsafe_allow_html=True)
                
                # UPC/UDI Match Results
                st.subheader("🔢 UPC/UDI Serial Matching")
                serial_results = validation_results['validations'].get('upc_udi_match', {})
                
                if serial_results.get('matches'):
                    st.markdown('<div class="validation-pass">✅ Matching serials found:</div>', 
                               unsafe_allow_html=True)
                    for match in serial_results['matches']:
                        st.write(f"• UPC: {match['upc']} matches UDI: {match['udi']}")
                elif serial_results.get('status') == 'fail':
                    st.markdown('<div class="validation-fail">❌ No matching serials found</div>', 
                               unsafe_allow_html=True)
                
                # Generate Reports
                st.markdown("---")
                st.header("📊 Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Serial Numbers Report")
                    serial_df = generate_serial_report(validation_results)
                    if not serial_df.empty:
                        st.dataframe(serial_df, use_container_width=True)
                        
                        # Download button for serial report
                        csv = serial_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Serial Report",
                            data=csv,
                            file_name=f"serial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No serial numbers found in documents")
                
                with col2:
                    st.subheader("Artwork Dimensions Report")
                    if st.session_state.file_dimensions:
                        dim_df = generate_dimension_report(st.session_state.file_dimensions)
                        if not dim_df.empty:
                            st.dataframe(dim_df, use_container_width=True)
                            
                            # Download button for dimensions report
                            csv = dim_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Dimensions Report",
                                data=csv,
                                file_name=f"dimensions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No dimension data available")
                
                # Save full validation report
                st.markdown("---")
                if st.button("💾 Save Full Validation Report", use_container_width=True):
                    report_json = json.dumps(validation_results, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=report_json,
                        file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    else:
        st.info("👆 Please upload artwork files to begin validation")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>Vive Health Artwork Verification System v2.0 | ISO 13485 Compliant</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
