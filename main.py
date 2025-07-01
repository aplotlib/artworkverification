"""
Vive Health Artwork Verification System
A comprehensive Streamlit app for validating packaging artwork against company standards
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
from typing import Dict, List, Tuple, Optional
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Vive Health Artwork Verification", 
    page_icon="✅", 
    layout="wide"
)

# --- Constants and Configuration ---
SKU_COLOR_MAPPING = {
    'BLK': 'black',
    'PUR': 'purple floral',
    'BGES': 'beige',
    'S': 'small',
    'M': 'medium',
    'L': 'large'
}

REQUIRED_DOCUMENTS = {
    'packaging_artwork': {
        'keywords': ['packaging', 'box', 'package', 'artwork'],
        'required': True,
        'description': 'Main product packaging artwork'
    },
    'manual': {
        'keywords': ['manual', 'instructions', 'guide', 'qsg', 'quickstart'],
        'required': False,
        'description': 'Product manual or quick start guide'
    },
    'washtag': {
        'keywords': ['washtag', 'wash tag', 'care instructions'],
        'required': False,
        'description': 'Washtag with care instructions'
    },
    'shipping_mark': {
        'keywords': ['shipping', 'mark', 'carton'],
        'required': True,
        'description': 'Shipping mark with SKU and quantity'
    },
    'thank_you_card': {
        'keywords': ['thank', 'thankyou'],
        'required': False,  # Only for Vive brand
        'description': 'Thank you card for Vive brand products'
    },
    'qc_sheet': {
        'keywords': ['qc', 'quality', 'sheet', 'specs'],
        'required': True,
        'description': 'QC sheet with specifications'
    },
    'made_in_china': {
        'keywords': ['made_in_china', 'china_sticker'],
        'required': False,  # Conditional
        'description': 'Made in China sticker'
    }
}

# --- Helper Classes ---
class DocumentExtractor:
    """Handles text and image extraction from various file types"""
    
    @staticmethod
    def extract_from_pdf(file_buffer, filename):
        """Extract text and images from PDF using OCR"""
        try:
            text = ""
            images = []
            file_buffer.seek(0)
            pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # First try to extract text directly
                page_text = page.get_text()
                
                # If no text found, use OCR
                if not page_text.strip():
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    image = Image.open(BytesIO(img_bytes))
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    images.append(image)
                
                text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
            
            pdf_document.close()
            return {'success': True, 'text': text, 'images': images}
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {filename}: {e}")
            return {'success': False, 'text': '', 'images': [], 'error': str(e)}
    
    @staticmethod
    def extract_from_image(file_buffer, filename):
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_buffer)
            text = pytesseract.image_to_string(image, lang='eng')
            return {'success': True, 'text': text, 'images': [image]}
        except Exception as e:
            logger.error(f"Error extracting from image {filename}: {e}")
            return {'success': False, 'text': '', 'images': [], 'error': str(e)}

class ArtworkValidator:
    """Validates artwork against Vive Health requirements"""
    
    def __init__(self):
        self.validation_results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'info': []
        }
    
    def validate_packaging_artwork(self, text, filename, product_info):
        """Validate main packaging artwork requirements"""
        results = []
        
        # Check for consistent product name
        if product_info.get('product_name'):
            name_count = text.lower().count(product_info['product_name'].lower())
            if name_count == 0:
                results.append(('failed', f'Product name "{product_info["product_name"]}" not found in {filename}'))
            elif name_count == 1:
                results.append(('warning', f'Product name appears only once in {filename}, verify consistency'))
            else:
                results.append(('passed', f'Product name consistent in {filename}'))
        
        # Check for Made in China
        if 'made in china' not in text.lower() and 'made in taiwan' not in text.lower():
            results.append(('failed', f'Missing country of origin in {filename}'))
        else:
            results.append(('passed', f'Country of origin present in {filename}'))
        
        # Check SKU and color matching
        if product_info.get('sku'):
            sku = product_info['sku']
            if sku in text:
                results.append(('passed', f'SKU {sku} found in {filename}'))
                
                # Check color suffix matching
                for suffix, color in SKU_COLOR_MAPPING.items():
                    if sku.upper().endswith(suffix) and color not in text.lower():
                        results.append(('failed', f'SKU indicates {color} but color not mentioned in {filename}'))
                        break
            else:
                results.append(('failed', f'SKU {sku} not found in {filename}'))
        
        # Check for UPC/UDI
        upc_pattern = r'\b\d{12,14}\b'
        udi_pattern = r'\(01\)\s*\d{14}'
        
        if re.search(upc_pattern, text):
            results.append(('passed', 'UPC barcode found'))
        else:
            results.append(('warning', 'UPC barcode not detected (may need manual verification)'))
            
        if re.search(udi_pattern, text):
            results.append(('passed', 'UDI format found'))
        else:
            results.append(('info', 'UDI not found (only required if specified in R&D)'))
        
        # Check for California Prop 65 warning
        if 'warning' in text.lower() and 'california' in text.lower():
            results.append(('passed', 'California Prop 65 warning found'))
        else:
            results.append(('info', 'No Prop 65 warning found (only required if specified in R&D)'))
        
        return results
    
    def validate_manual(self, text, filename):
        """Validate manual/quick start guide"""
        results = []
        
        # Basic spell check indicators (common misspellings)
        common_errors = ['recieve', 'occured', 'seperate', 'definately', 'accomodate']
        found_errors = [error for error in common_errors if error in text.lower()]
        
        if found_errors:
            results.append(('failed', f'Potential spelling errors in {filename}: {", ".join(found_errors)}'))
        else:
            results.append(('passed', f'No common spelling errors detected in {filename}'))
        
        return results
    
    def validate_shipping_mark(self, text, filename, product_info):
        """Validate shipping mark requirements"""
        results = []
        
        # Check format SKU - QTY
        sku_qty_pattern = r'([A-Z]{3}\d{4}[A-Z]{0,4})\s*[-–]\s*(\d+)'
        match = re.search(sku_qty_pattern, text.upper())
        
        if match:
            results.append(('passed', f'Shipping mark format correct: {match.group(0)}'))
        else:
            results.append(('failed', f'Shipping mark format incorrect in {filename}. Expected: SKU - QTY'))
        
        # Check for Made in China
        if 'made in china' not in text.lower():
            results.append(('warning', f'Made in China not found in shipping mark {filename}'))
        else:
            results.append(('passed', f'Made in China present in shipping mark'))
        
        return results
    
    def validate_washtag(self, text, filename):
        """Validate washtag requirements"""
        results = []
        
        # Check for care instructions
        care_keywords = ['machine wash', 'wash cold', 'air dry', 'do not bleach', 'do not iron']
        found_keywords = [kw for kw in care_keywords if kw in text.lower()]
        
        if len(found_keywords) >= 3:
            results.append(('passed', f'Care instructions found in {filename}'))
        else:
            results.append(('warning', f'Limited care instructions in {filename}. Found: {", ".join(found_keywords)}'))
        
        # Check for Made in China
        if 'made in china' in text.lower():
            results.append(('passed', 'Made in China present on washtag'))
        else:
            results.append(('failed', 'Made in China missing from washtag'))
        
        # Check for material composition
        if '%' in text and any(word in text.lower() for word in ['polyester', 'cotton', 'pvc', 'ldpe']):
            results.append(('passed', 'Material composition found'))
        else:
            results.append(('warning', 'Material composition may be missing'))
        
        return results
    
    def validate_all(self, documents, product_info):
        """Run all validations and compile results"""
        all_results = []
        
        for doc_type, doc_data in documents.items():
            if not doc_data:
                continue
                
            text = doc_data.get('text', '')
            filename = doc_data.get('filename', '')
            
            if doc_type == 'packaging_artwork':
                results = self.validate_packaging_artwork(text, filename, product_info)
            elif doc_type == 'manual':
                results = self.validate_manual(text, filename)
            elif doc_type == 'shipping_mark':
                results = self.validate_shipping_mark(text, filename, product_info)
            elif doc_type == 'washtag':
                results = self.validate_washtag(text, filename)
            else:
                results = [('info', f'{doc_type} uploaded but no specific validation rules')]
            
            all_results.extend(results)
        
        # Check for required documents
        if product_info.get('brand', '').lower() == 'vive' and 'thank_you_card' not in documents:
            all_results.append(('failed', 'Thank you card required for Vive brand products but not found'))
        
        # Check Made in China sticker requirement
        has_washtag = 'washtag' in documents
        has_rating_label = any('rating' in doc.get('text', '').lower() for doc in documents.values() if doc)
        
        if not has_washtag and not has_rating_label and 'made_in_china' not in documents:
            all_results.append(('warning', 'Made in China sticker may be required (no washtag or rating label found)'))
        
        return all_results

class DocumentClassifier:
    """Classifies uploaded documents by type"""
    
    @staticmethod
    def classify_document(filename, text):
        """Determine document type based on filename and content"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        for doc_type, config in REQUIRED_DOCUMENTS.items():
            # Check filename
            if any(keyword in filename_lower for keyword in config['keywords']):
                return doc_type
            
            # Check content
            if doc_type == 'packaging_artwork' and 'distributed by' in text_lower:
                return doc_type
            elif doc_type == 'shipping_mark' and 'item name:' in text_lower and 'po #:' in text_lower:
                return doc_type
            elif doc_type == 'washtag' and 'machine wash' in text_lower:
                return doc_type
            elif doc_type == 'manual' and ('quick start guide' in text_lower or 'instructions' in text_lower):
                return doc_type
        
        return 'unknown'

# --- UI Functions ---
def display_header():
    """Display app header with styling"""
    st.markdown("""
    <style>
        .main-header {
            padding: 2rem;
            background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%);
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        .stButton>button {
            background-color: #0891b2;
            color: white;
        }
        .success-box {
            padding: 1rem;
            background-color: #d1fae5;
            border-left: 4px solid #10b981;
            margin: 0.5rem 0;
        }
        .error-box {
            padding: 1rem;
            background-color: #fee2e2;
            border-left: 4px solid #ef4444;
            margin: 0.5rem 0;
        }
        .warning-box {
            padding: 1rem;
            background-color: #fef3c7;
            border-left: 4px solid #f59e0b;
            margin: 0.5rem 0;
        }
        .info-box {
            padding: 1rem;
            background-color: #dbeafe;
            border-left: 4px solid #3b82f6;
            margin: 0.5rem 0;
        }
    </style>
    <div class="main-header">
        <h1>✅ Vive Health Artwork Verification System</h1>
        <p>Automated validation against packaging standards and requirements</p>
    </div>
    """, unsafe_allow_html=True)

def display_validation_result(status, message):
    """Display validation result with appropriate styling"""
    if status == 'passed':
        st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
    elif status == 'failed':
        st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
    elif status == 'warning':
        st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)
    else:  # info
        st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)

def generate_report(validation_results, documents, product_info):
    """Generate a comprehensive validation report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'product_info': product_info,
        'documents_reviewed': list(documents.keys()),
        'validation_summary': {
            'total_checks': len(validation_results),
            'passed': len([r for r in validation_results if r[0] == 'passed']),
            'failed': len([r for r in validation_results if r[0] == 'failed']),
            'warnings': len([r for r in validation_results if r[0] == 'warning']),
            'info': len([r for r in validation_results if r[0] == 'info'])
        },
        'detailed_results': validation_results
    }
    
    return report

def export_report_to_csv(report):
    """Convert report to CSV format"""
    rows = []
    for result in report['detailed_results']:
        rows.append({
            'Status': result[0].upper(),
            'Message': result[1],
            'Product': report['product_info'].get('product_name', 'Unknown'),
            'SKU': report['product_info'].get('sku', 'Unknown'),
            'Timestamp': report['timestamp']
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

# --- Main Application ---
def main():
    if 'validation_complete' not in st.session_state:
        st.session_state.validation_complete = False
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = []
    
    display_header()
    
    # Sidebar for product information
    with st.sidebar:
        st.header("📋 Product Information")
        
        product_info = {}
        product_info['product_name'] = st.text_input("Product Name", placeholder="e.g., Wheelchair Bag Advanced")
        product_info['sku'] = st.text_input("SKU", placeholder="e.g., LVA3100BLK")
        product_info['brand'] = st.selectbox("Brand", ["Vive", "Other"])
        
        st.markdown("---")
        st.header("📚 Reference Documents")
        st.info("Upload your R&D document or paste the link for reference")
        rd_link = st.text_input("R&D Document Link", placeholder="Google Docs link")
        
        if st.button("Clear All", type="secondary"):
            st.session_state.documents = {}
            st.session_state.validation_complete = False
            st.session_state.validation_results = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Artwork Files")
        
        uploaded_files = st.file_uploader(
            "Upload all artwork files (PDFs, images)",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload packaging artwork, manuals, shipping marks, washtags, etc."
        )
        
        if uploaded_files:
            with st.spinner("Processing files..."):
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    # Extract content based on file type
                    if file.type == "application/pdf":
                        extraction = DocumentExtractor.extract_from_pdf(file, file.name)
                    else:
                        extraction = DocumentExtractor.extract_from_image(file, file.name)
                    
                    if extraction['success']:
                        # Classify document
                        doc_type = DocumentClassifier.classify_document(file.name, extraction['text'])
                        
                        # Store document
                        st.session_state.documents[doc_type] = {
                            'filename': file.name,
                            'text': extraction['text'],
                            'images': extraction.get('images', []),
                            'type': doc_type
                        }
                
                progress_bar.empty()
                st.success(f"✅ Processed {len(uploaded_files)} files")
    
    with col2:
        st.header("📊 Document Status")
        
        # Show which documents have been uploaded
        for doc_type, config in REQUIRED_DOCUMENTS.items():
            if doc_type in st.session_state.documents:
                st.success(f"✅ {config['description']}")
            elif config['required']:
                st.error(f"❌ {config['description']} (Required)")
            else:
                st.info(f"➖ {config['description']} (Optional)")
    
    # Validation section
    st.markdown("---")
    
    if st.button("🔍 Run Validation", type="primary", disabled=not uploaded_files):
        with st.spinner("Validating artwork..."):
            validator = ArtworkValidator()
            validation_results = validator.validate_all(st.session_state.documents, product_info)
            st.session_state.validation_results = validation_results
            st.session_state.validation_complete = True
    
    # Display results
    if st.session_state.validation_complete:
        st.header("📋 Validation Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        passed = len([r for r in st.session_state.validation_results if r[0] == 'passed'])
        failed = len([r for r in st.session_state.validation_results if r[0] == 'failed'])
        warnings = len([r for r in st.session_state.validation_results if r[0] == 'warning'])
        info = len([r for r in st.session_state.validation_results if r[0] == 'info'])
        
        with col1:
            st.metric("Passed", passed, delta=None, delta_color="off")
        with col2:
            st.metric("Failed", failed, delta=None, delta_color="off")
        with col3:
            st.metric("Warnings", warnings, delta=None, delta_color="off")
        with col4:
            st.metric("Info", info, delta=None, delta_color="off")
        
        # Detailed results
        st.markdown("---")
        
        # Group results by status
        for status in ['failed', 'warning', 'passed', 'info']:
            results = [r for r in st.session_state.validation_results if r[0] == status]
            if results:
                st.subheader(f"{status.title()} Items")
                for _, message in results:
                    display_validation_result(status, message)
        
        # Export options
        st.markdown("---")
        st.header("📥 Export Report")
        
        report = generate_report(st.session_state.validation_results, st.session_state.documents, product_info)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_report_to_csv(report)
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name=f"artwork_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = json.dumps(report, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name=f"artwork_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Document preview section
    if st.session_state.documents:
        st.markdown("---")
        st.header("📄 Document Preview")
        
        doc_type = st.selectbox("Select document to preview", list(st.session_state.documents.keys()))
        
        if doc_type in st.session_state.documents:
            doc = st.session_state.documents[doc_type]
            
            with st.expander(f"View {doc['filename']}", expanded=True):
                st.text_area("Extracted Text", doc['text'], height=300)
                
                if doc.get('images'):
                    st.subheader("Images")
                    for idx, img in enumerate(doc['images']):
                        st.image(img, caption=f"Page {idx + 1}", use_container_width=True)

if __name__ == "__main__":
    main()
