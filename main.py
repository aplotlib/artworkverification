"""
Vive Health Artwork Verification System
A comprehensive Streamlit app for validating packaging artwork against company standards
Enhanced with stability improvements and resource management
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
import gc
import sys
import os
import tempfile
import weakref
from contextlib import contextmanager
import time

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('artwork_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if not PSUTIL_AVAILABLE:
    logger.warning("psutil not available - memory monitoring disabled")

# Page configuration
st.set_page_config(
    page_title="Vive Health Artwork Verification", 
    page_icon="✅", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Memory Management ---
class MemoryManager:
    """Manages memory usage and cleanup"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024
            except Exception as e:
                logger.warning(f"Error getting memory usage: {e}")
                return 0
        else:
            # Return a placeholder value when psutil is not available
            return 0
    
    @staticmethod
    def cleanup():
        """Force garbage collection and log memory usage"""
        if PSUTIL_AVAILABLE:
            before = MemoryManager.get_memory_usage()
            gc.collect()
            after = MemoryManager.get_memory_usage()
            logger.info(f"Memory cleanup: {before:.1f}MB -> {after:.1f}MB (freed {before-after:.1f}MB)")
        else:
            gc.collect()
            logger.info("Memory cleanup performed (psutil not available for metrics)")
    
    @staticmethod
    def check_memory_threshold(threshold_mb=500):
        """Check if memory usage exceeds threshold"""
        if PSUTIL_AVAILABLE:
            usage = MemoryManager.get_memory_usage()
            if usage > threshold_mb:
                logger.warning(f"High memory usage: {usage:.1f}MB")
                MemoryManager.cleanup()
                return True
        else:
            # Still perform periodic cleanup even without metrics
            MemoryManager.cleanup()
        return False

# --- Session State Manager ---
class SessionStateManager:
    """Manages session state with size limits and cleanup"""
    
    MAX_DOCUMENTS = 20  # Maximum documents to keep in session
    MAX_VALIDATION_HISTORY = 10  # Maximum validation results to keep
    
    @staticmethod
    def initialize():
        """Initialize session state with defaults"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.validation_complete = False
            st.session_state.documents = {}
            st.session_state.validation_results = []
            st.session_state.review_states = {}
            st.session_state.last_activity = time.time()
            st.session_state.error_count = 0
            logger.info("Session state initialized")
    
    @staticmethod
    def cleanup_old_data():
        """Remove old data from session state"""
        try:
            # Clean up old documents
            if len(st.session_state.documents) > SessionStateManager.MAX_DOCUMENTS:
                # Keep only the most recent documents
                sorted_docs = sorted(
                    st.session_state.documents.items(),
                    key=lambda x: x[1].get('timestamp', 0),
                    reverse=True
                )
                st.session_state.documents = dict(sorted_docs[:SessionStateManager.MAX_DOCUMENTS])
                logger.info(f"Cleaned up old documents, kept {SessionStateManager.MAX_DOCUMENTS} most recent")
            
            # Clean up validation history
            if len(st.session_state.validation_results) > SessionStateManager.MAX_VALIDATION_HISTORY:
                st.session_state.validation_results = st.session_state.validation_results[-SessionStateManager.MAX_VALIDATION_HISTORY:]
                logger.info("Cleaned up old validation results")
            
            # Update last activity
            st.session_state.last_activity = time.time()
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    @staticmethod
    def reset_on_error():
        """Reset session state after multiple errors"""
        if st.session_state.get('error_count', 0) > 3:
            logger.warning("Too many errors, resetting session state")
            for key in list(st.session_state.keys()):
                if key != 'initialized':
                    del st.session_state[key]
            SessionStateManager.initialize()

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

# --- Context Managers for Resource Management ---
@contextmanager
def managed_pdf_document(file_buffer, filename):
    """Context manager for PDF documents to ensure proper cleanup"""
    pdf_document = None
    try:
        file_buffer.seek(0)
        pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
        yield pdf_document
    except Exception as e:
        logger.error(f"Error with PDF {filename}: {e}")
        raise
    finally:
        if pdf_document:
            pdf_document.close()
            del pdf_document
        gc.collect()

@contextmanager
def managed_image(file_buffer):
    """Context manager for images to ensure proper cleanup"""
    image = None
    try:
        image = Image.open(file_buffer)
        yield image
    finally:
        if image:
            image.close()
            del image
        gc.collect()

# --- Helper Classes ---
class DocumentExtractor:
    """Handles text and image extraction from various file types with resource management"""
    
    @staticmethod
    def extract_from_pdf(file_buffer, filename):
        """Extract text and images from PDF using OCR with proper resource management"""
        try:
            text = ""
            images = []
            
            with managed_pdf_document(file_buffer, filename) as pdf_document:
                total_pages = len(pdf_document)
                
                # Limit pages to prevent memory issues
                max_pages = min(total_pages, 50)
                if total_pages > max_pages:
                    logger.warning(f"PDF has {total_pages} pages, processing only first {max_pages}")
                
                for page_num in range(max_pages):
                    try:
                        page = pdf_document.load_page(page_num)
                        
                        # First try to extract text directly
                        page_text = page.get_text()
                        
                        # If no text found, use OCR
                        if not page_text.strip():
                            # Use lower DPI for better performance
                            pix = page.get_pixmap(dpi=150)
                            img_bytes = pix.tobytes("png")
                            
                            with BytesIO(img_bytes) as img_buffer:
                                with managed_image(img_buffer) as image:
                                    # Resize if too large
                                    max_size = (2000, 2000)
                                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                                        image.thumbnail(max_size, Image.Resampling.LANCZOS)
                                    
                                    page_text = pytesseract.image_to_string(image, lang='eng', timeout=30)
                                    
                                    # Store only first few images to prevent memory issues
                                    if len(images) < 5:
                                        images.append(image.copy())
                            
                            # Clean up pixmap
                            pix = None
                        
                        text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        # Check memory periodically
                        if page_num % 10 == 0:
                            MemoryManager.check_memory_threshold()
                        
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        continue
            
            return {'success': True, 'text': text, 'images': images}
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {filename}: {e}")
            return {'success': False, 'text': '', 'images': [], 'error': str(e)}
        finally:
            MemoryManager.cleanup()
    
    @staticmethod
    def extract_from_image(file_buffer, filename):
        """Extract text from image using OCR with resource management"""
        try:
            text = ""
            images = []
            
            with managed_image(file_buffer) as image:
                # Resize if too large
                max_size = (2000, 2000)
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                text = pytesseract.image_to_string(image, lang='eng', timeout=30)
                images = [image.copy()]
            
            return {'success': True, 'text': text, 'images': images}
            
        except Exception as e:
            logger.error(f"Error extracting from image {filename}: {e}")
            return {'success': False, 'text': '', 'images': [], 'error': str(e)}
        finally:
            MemoryManager.cleanup()
    
    @staticmethod
    def extract_from_excel(file_buffer, filename):
        """Extract text from Excel file with resource management"""
        try:
            text = ""
            
            # Use context manager for Excel file
            with pd.ExcelFile(file_buffer) as excel_file:
                # Limit sheets to prevent memory issues
                max_sheets = min(len(excel_file.sheet_names), 10)
                
                for idx, sheet_name in enumerate(excel_file.sheet_names[:max_sheets]):
                    try:
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=1000)
                        text += f"\n\n--- Sheet: {sheet_name} ---\n"
                        text += df.to_string(index=False, header=False, max_rows=100)
                        
                        # Clean up dataframe
                        del df
                        
                    except Exception as e:
                        logger.error(f"Error reading sheet {sheet_name}: {e}")
                        continue
            
            return {'success': True, 'text': text, 'images': []}
            
        except Exception as e:
            logger.error(f"Error extracting from Excel {filename}: {e}")
            return {'success': False, 'text': '', 'images': [], 'error': str(e)}
        finally:
            MemoryManager.cleanup()

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
        
        try:
            # Check for consistent product name
            if product_info.get('product_name'):
                name_count = text.lower().count(product_info['product_name'].lower())
                if name_count == 0:
                    results.append(('failed', f'Product name "{product_info["product_name"]}" not found in {filename}', 'product_name_missing'))
                elif name_count == 1:
                    results.append(('warning', f'Product name appears only once in {filename}, verify consistency', 'product_name_single'))
                else:
                    results.append(('passed', f'Product name consistent in {filename}', 'product_name_ok'))
            
            # Check for Made in China
            if 'made in china' not in text.lower() and 'made in taiwan' not in text.lower():
                results.append(('failed', f'Missing country of origin in {filename}', 'origin_missing'))
            else:
                results.append(('passed', f'Country of origin present in {filename}', 'origin_ok'))
            
            # Check SKU and color matching
            if product_info.get('sku'):
                sku = product_info['sku']
                if sku in text:
                    results.append(('passed', f'SKU {sku} found in {filename}', 'sku_found'))
                    
                    # Check color suffix matching
                    for suffix, color in SKU_COLOR_MAPPING.items():
                        if sku.upper().endswith(suffix) and color not in text.lower():
                            results.append(('failed', f'SKU indicates {color} but color not mentioned in {filename}', 'color_mismatch'))
                            break
                else:
                    results.append(('failed', f'SKU {sku} not found in {filename}', 'sku_missing'))
            
            # Check for UPC/UDI
            upc_pattern = r'\b\d{12,14}\b'
            udi_pattern = r'\(01\)\s*\d{14}'
            
            if re.search(upc_pattern, text):
                results.append(('passed', 'UPC barcode found', 'upc_found'))
            else:
                results.append(('warning', 'UPC barcode not detected (may need manual verification)', 'upc_not_detected'))
                
            if re.search(udi_pattern, text):
                results.append(('passed', 'UDI format found', 'udi_found'))
            else:
                results.append(('info', 'UDI not found (only required if specified in R&D)', 'udi_not_found'))
            
            # Check for California Prop 65 warning
            if 'warning' in text.lower() and 'california' in text.lower():
                results.append(('passed', 'California Prop 65 warning found', 'prop65_found'))
            else:
                results.append(('info', 'No Prop 65 warning found (only required if specified in R&D)', 'prop65_not_found'))
            
        except Exception as e:
            logger.error(f"Error validating packaging artwork: {e}")
            results.append(('failed', f'Error during validation: {str(e)}', 'validation_error'))
        
        return results
    
    def validate_manual(self, text, filename):
        """Validate manual/quick start guide"""
        results = []
        
        try:
            # Basic spell check indicators (common misspellings)
            common_errors = ['recieve', 'occured', 'seperate', 'definately', 'accomodate']
            found_errors = [error for error in common_errors if error in text.lower()]
            
            if found_errors:
                results.append(('failed', f'Potential spelling errors in {filename}: {", ".join(found_errors)}', 'spelling_errors'))
            else:
                results.append(('passed', f'No common spelling errors detected in {filename}', 'spelling_ok'))
            
        except Exception as e:
            logger.error(f"Error validating manual: {e}")
            results.append(('failed', f'Error during validation: {str(e)}', 'validation_error'))
        
        return results
    
    def validate_shipping_mark(self, text, filename, product_info):
        """Validate shipping mark requirements"""
        results = []
        
        try:
            # Check format SKU - QTY
            sku_qty_pattern = r'([A-Z]{3}\d{4}[A-Z]{0,4})\s*[-–]\s*(\d+)'
            match = re.search(sku_qty_pattern, text.upper())
            
            if match:
                results.append(('passed', f'Shipping mark format correct: {match.group(0)}', 'format_ok'))
            else:
                results.append(('failed', f'Shipping mark format incorrect in {filename}. Expected: SKU - QTY', 'format_error'))
            
            # Check for Made in China
            if 'made in china' not in text.lower():
                results.append(('warning', f'Made in China not found in shipping mark {filename}', 'origin_missing'))
            else:
                results.append(('passed', f'Made in China present in shipping mark', 'origin_ok'))
            
        except Exception as e:
            logger.error(f"Error validating shipping mark: {e}")
            results.append(('failed', f'Error during validation: {str(e)}', 'validation_error'))
        
        return results
    
    def validate_washtag(self, text, filename):
        """Validate washtag requirements"""
        results = []
        
        try:
            # Check for care instructions
            care_keywords = ['machine wash', 'wash cold', 'air dry', 'do not bleach', 'do not iron']
            found_keywords = [kw for kw in care_keywords if kw in text.lower()]
            
            if len(found_keywords) >= 3:
                results.append(('passed', f'Care instructions found in {filename}', 'care_ok'))
            else:
                results.append(('warning', f'Limited care instructions in {filename}. Found: {", ".join(found_keywords)}', 'care_limited'))
            
            # Check for Made in China
            if 'made in china' in text.lower():
                results.append(('passed', 'Made in China present on washtag', 'origin_ok'))
            else:
                results.append(('failed', 'Made in China missing from washtag', 'origin_missing'))
            
            # Check for material composition
            if '%' in text and any(word in text.lower() for word in ['polyester', 'cotton', 'pvc', 'ldpe']):
                results.append(('passed', 'Material composition found', 'material_ok'))
            else:
                results.append(('warning', 'Material composition may be missing', 'material_missing'))
            
        except Exception as e:
            logger.error(f"Error validating washtag: {e}")
            results.append(('failed', f'Error during validation: {str(e)}', 'validation_error'))
        
        return results
    
    def validate_qc_sheet(self, text, filename):
        """Validate QC sheet requirements"""
        results = []
        
        try:
            # Check for key QC sheet elements
            if 'product name' in text.lower() or 'product sku code' in text.lower():
                results.append(('passed', f'Product information found in {filename}', 'product_info_ok'))
            else:
                results.append(('warning', f'Product information may be missing in {filename}', 'product_info_missing'))
            
            # Check for specifications
            if 'print style' in text.lower() or 'paper type' in text.lower() or 'dimensions' in text.lower():
                results.append(('passed', f'Specifications found in {filename}', 'specs_ok'))
            else:
                results.append(('warning', f'Specifications may be incomplete in {filename}', 'specs_incomplete'))
            
            # Check for thank you card specs if mentioned
            if 'thank you card' in text.lower():
                if '88.9mm' in text or '50.8mm' in text:
                    results.append(('passed', 'Thank you card dimensions correct', 'thankyou_dims_ok'))
                else:
                    results.append(('warning', 'Thank you card dimensions not standard (should be 88.9mm x 50.8mm)', 'thankyou_dims_nonstandard'))
            
            # Check for QR code specs
            if 'qr code' in text.lower():
                if '19mm x 19mm' in text:
                    results.append(('passed', 'QR code dimensions correct', 'qr_dims_ok'))
                else:
                    results.append(('warning', 'QR code dimensions not standard (should be 19mm x 19mm)', 'qr_dims_nonstandard'))
            
        except Exception as e:
            logger.error(f"Error validating QC sheet: {e}")
            results.append(('failed', f'Error during validation: {str(e)}', 'validation_error'))
        
        return results
    
    def validate_all(self, documents, product_info):
        """Run all validations and compile results"""
        all_results = []
        
        try:
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
                elif doc_type == 'qc_sheet':
                    results = self.validate_qc_sheet(text, filename)
                else:
                    results = [('info', f'{doc_type} uploaded but no specific validation rules', f'{doc_type}_no_rules')]
                
                # Add document type to results
                for result in results:
                    all_results.append((*result, doc_type))
            
            # Check for required documents
            if product_info.get('brand', '').lower() == 'vive' and 'thank_you_card' not in documents:
                all_results.append(('failed', 'Thank you card required for Vive brand products but not found', 'thank_you_missing', 'general'))
            
            # Check Made in China sticker requirement
            has_washtag = 'washtag' in documents
            has_rating_label = any('rating' in doc.get('text', '').lower() for doc in documents.values() if doc)
            
            if not has_washtag and not has_rating_label and 'made_in_china' not in documents:
                all_results.append(('warning', 'Made in China sticker may be required (no washtag or rating label found)', 'china_sticker_maybe', 'general'))
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            all_results.append(('failed', f'Validation process error: {str(e)}', 'process_error', 'general'))
        
        return all_results

class DocumentClassifier:
    """Classifies uploaded documents by type"""
    
    @staticmethod
    def classify_document(filename, text):
        """Determine document type based on filename and content"""
        try:
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
                elif doc_type == 'qc_sheet' and ('product sku code' in text_lower or 'paper type' in text_lower or 'print style' in text_lower):
                    return doc_type
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return 'unknown'

# --- UI Functions ---
def display_header():
    """Display app header with styling and health status"""
    memory_usage = MemoryManager.get_memory_usage()
    
    # Format health indicator based on psutil availability
    if PSUTIL_AVAILABLE and memory_usage > 0:
        health_html = f'<div class="health-indicator">🟢 System OK | Memory: {memory_usage:.0f}MB</div>'
    else:
        health_html = '<div class="health-indicator">🟢 System OK</div>'
    
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
        .health-indicator {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8rem;
            background: #10b981;
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
        .review-box {
            padding: 1rem;
            background-color: #f3f4f6;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
    </style>
    <div class="main-header">
        """ + health_html + """
        <h1>✅ Vive Health Artwork Verification System</h1>
        <p>Automated validation against packaging standards and requirements</p>
    </div>
    """, unsafe_allow_html=True)

def display_validation_result_with_review(status, message, result_id, doc_type):
    """Display validation result with review checkboxes"""
    try:
        # Initialize review state if not exists
        if 'review_states' not in st.session_state:
            st.session_state.review_states = {}
        
        if result_id not in st.session_state.review_states:
            st.session_state.review_states[result_id] = {
                'reviewed': False,
                'finding_accurate': None,
                'action_taken': None,
                'notes': ''
            }
        
        # Display the validation result
        if status == 'passed':
            st.markdown(f'<div class="success-box">✅ {message}</div>', unsafe_allow_html=True)
        elif status == 'failed':
            st.markdown(f'<div class="error-box">❌ {message}</div>', unsafe_allow_html=True)
        elif status == 'warning':
            st.markdown(f'<div class="warning-box">⚠️ {message}</div>', unsafe_allow_html=True)
        else:  # info
            st.markdown(f'<div class="info-box">ℹ️ {message}</div>', unsafe_allow_html=True)
        
        # Only show review options for failed and warning items
        if status in ['failed', 'warning']:
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 3])
                
                with col1:
                    reviewed = st.checkbox(
                        "Reviewed",
                        key=f"reviewed_{result_id}",
                        value=st.session_state.review_states[result_id]['reviewed']
                    )
                    st.session_state.review_states[result_id]['reviewed'] = reviewed
                
                with col2:
                    if reviewed:
                        finding_accurate = st.radio(
                            "Finding accurate?",
                            ["Yes", "No"],
                            key=f"accurate_{result_id}",
                            horizontal=True,
                            index=0 if st.session_state.review_states[result_id]['finding_accurate'] == 'Yes' else 
                                  1 if st.session_state.review_states[result_id]['finding_accurate'] == 'No' else None
                        )
                        st.session_state.review_states[result_id]['finding_accurate'] = finding_accurate
                
                with col3:
                    if reviewed and st.session_state.review_states[result_id]['finding_accurate'] == 'Yes':
                        action = st.radio(
                            "Action taken:",
                            ["Fixed", "Will Fix Later", "No Action Needed"],
                            key=f"action_{result_id}",
                            horizontal=True,
                            index=["Fixed", "Will Fix Later", "No Action Needed"].index(st.session_state.review_states[result_id]['action_taken']) 
                                  if st.session_state.review_states[result_id]['action_taken'] in ["Fixed", "Will Fix Later", "No Action Needed"] else None
                        )
                        st.session_state.review_states[result_id]['action_taken'] = action
                
                # Notes field for additional comments
                if reviewed:
                    notes = st.text_input(
                        "Notes (optional):",
                        key=f"notes_{result_id}",
                        value=st.session_state.review_states[result_id]['notes'],
                        placeholder="e.g., Updated artwork file version 2"
                    )
                    st.session_state.review_states[result_id]['notes'] = notes
                    
    except Exception as e:
        logger.error(f"Error displaying validation result: {e}")
        st.error(f"Display error: {str(e)}")

def generate_report(validation_results, documents, product_info):
    """Generate a comprehensive validation report with review status"""
    try:
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
            'review_summary': {
                'total_issues': len([r for r in validation_results if r[0] in ['failed', 'warning']]),
                'reviewed': 0,
                'fixed': 0,
                'will_fix_later': 0,
                'no_action_needed': 0,
                'disputed': 0
            },
            'detailed_results': []
        }
        
        # Count review states
        for result in validation_results:
            status, message, result_id, doc_type = result
            
            review_state = st.session_state.review_states.get(result_id, {})
            
            if review_state.get('reviewed'):
                report['review_summary']['reviewed'] += 1
                
                if review_state.get('finding_accurate') == 'No':
                    report['review_summary']['disputed'] += 1
                elif review_state.get('action_taken') == 'Fixed':
                    report['review_summary']['fixed'] += 1
                elif review_state.get('action_taken') == 'Will Fix Later':
                    report['review_summary']['will_fix_later'] += 1
                elif review_state.get('action_taken') == 'No Action Needed':
                    report['review_summary']['no_action_needed'] += 1
            
            # Add to detailed results
            report['detailed_results'].append({
                'status': status,
                'message': message,
                'document_type': doc_type,
                'reviewed': review_state.get('reviewed', False),
                'finding_accurate': review_state.get('finding_accurate', 'N/A'),
                'action_taken': review_state.get('action_taken', 'N/A'),
                'notes': review_state.get('notes', '')
            })
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None

def export_report_to_csv(report):
    """Convert report to CSV format with review status"""
    try:
        rows = []
        for result in report['detailed_results']:
            rows.append({
                'Status': result['status'].upper(),
                'Message': result['message'],
                'Document Type': result['document_type'],
                'Product': report['product_info'].get('product_name', 'Unknown'),
                'SKU': report['product_info'].get('sku', 'Unknown'),
                'Reviewed': 'Yes' if result['reviewed'] else 'No',
                'Finding Accurate': result['finding_accurate'],
                'Action Taken': result['action_taken'],
                'Notes': result['notes'],
                'Timestamp': report['timestamp']
            })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return ""

# --- Error Recovery Functions ---
def safe_file_upload(file, max_retries=3):
    """Safely process uploaded file with retries"""
    for attempt in range(max_retries):
        try:
            # Create a copy of the file buffer to avoid issues
            file_copy = BytesIO(file.read())
            file.seek(0)  # Reset original file pointer
            
            return file_copy
            
        except Exception as e:
            logger.error(f"File upload attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

# --- Main Application ---
def main():
    try:
        # Initialize session state
        SessionStateManager.initialize()
        
        # Periodic cleanup
        if time.time() - st.session_state.get('last_cleanup', 0) > 300:  # Every 5 minutes
            SessionStateManager.cleanup_old_data()
            st.session_state.last_cleanup = time.time()
        
        # Display header with health status
        display_header()
        
        # Error recovery check
        SessionStateManager.reset_on_error()
        
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
            
            st.markdown("---")
            st.header("🔧 System Status")
            
            if PSUTIL_AVAILABLE:
                memory = MemoryManager.get_memory_usage()
                st.metric("Memory Usage", f"{memory:.0f} MB", 
                         delta=f"{memory - 100:.0f} MB" if memory > 100 else None)
            else:
                st.info("Memory monitoring unavailable")
                st.caption("Install psutil for memory metrics")
            
            if st.button("Clear All", type="secondary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key != 'initialized':
                        del st.session_state[key]
                SessionStateManager.initialize()
                MemoryManager.cleanup()
                st.rerun()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📤 Upload Artwork Files")
            
            uploaded_files = st.file_uploader(
                "Upload all artwork files (PDFs, images, Excel files)",
                type=['pdf', 'png', 'jpg', 'jpeg', 'xlsx', 'xls'],
                accept_multiple_files=True,
                help="Upload packaging artwork, manuals, shipping marks, washtags, QC sheets, etc."
            )
            
            if uploaded_files:
                with st.spinner("Processing files..."):
                    progress_bar = st.progress(0)
                    
                    for idx, file in enumerate(uploaded_files):
                        try:
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                            
                            # Increment error count if needed
                            st.session_state.error_count = 0
                            
                            # Safe file processing
                            file_buffer = safe_file_upload(file)
                            
                            # Extract content based on file type
                            if file.type == "application/pdf":
                                extraction = DocumentExtractor.extract_from_pdf(file_buffer, file.name)
                            elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                                extraction = DocumentExtractor.extract_from_excel(file_buffer, file.name)
                            else:
                                extraction = DocumentExtractor.extract_from_image(file_buffer, file.name)
                            
                            if extraction['success']:
                                # Classify document
                                doc_type = DocumentClassifier.classify_document(file.name, extraction['text'])
                                
                                # Store document with timestamp
                                st.session_state.documents[doc_type] = {
                                    'filename': file.name,
                                    'text': extraction['text'],
                                    'images': extraction.get('images', []),
                                    'type': doc_type,
                                    'timestamp': time.time()
                                }
                            else:
                                st.error(f"Failed to process {file.name}: {extraction.get('error', 'Unknown error')}")
                                st.session_state.error_count += 1
                                
                        except Exception as e:
                            logger.error(f"Error processing {file.name}: {e}")
                            st.error(f"Error processing {file.name}: {str(e)}")
                            st.session_state.error_count += 1
                            continue
                    
                    progress_bar.empty()
                    
                    # Check memory after processing
                    MemoryManager.check_memory_threshold()
                    
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
            try:
                with st.spinner("Validating artwork..."):
                    validator = ArtworkValidator()
                    validation_results = validator.validate_all(st.session_state.documents, product_info)
                    st.session_state.validation_results = validation_results
                    st.session_state.validation_complete = True
                    # Reset review states for new validation
                    st.session_state.review_states = {}
                    
                    # Clean up after validation
                    MemoryManager.cleanup()
                    
            except Exception as e:
                logger.error(f"Validation error: {e}")
                st.error(f"Validation failed: {str(e)}")
                st.session_state.error_count += 1
        
        # Display results
        if st.session_state.validation_complete:
            st.header("📋 Validation Results")
            
            try:
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
                
                # Review progress
                issues = [r for r in st.session_state.validation_results if r[0] in ['failed', 'warning']]
                if issues:
                    reviewed_count = sum(1 for r in issues if st.session_state.review_states.get(r[2], {}).get('reviewed', False))
                    st.progress(reviewed_count / len(issues), text=f"Review Progress: {reviewed_count}/{len(issues)} issues reviewed")
                
                # Detailed results
                st.markdown("---")
                
                # Group results by document type
                results_by_doc = {}
                for result in st.session_state.validation_results:
                    doc_type = result[3]
                    if doc_type not in results_by_doc:
                        results_by_doc[doc_type] = []
                    results_by_doc[doc_type].append(result)
                
                # Display results by document
                for doc_type, results in results_by_doc.items():
                    if doc_type in st.session_state.documents:
                        st.subheader(f"📄 {st.session_state.documents[doc_type]['filename']}")
                    else:
                        st.subheader(f"📋 {doc_type.replace('_', ' ').title()}")
                    
                    for result in results:
                        status, message, result_id, _ = result
                        display_validation_result_with_review(status, message, result_id, doc_type)
                    
                    st.markdown("---")
                
                # Export options
                st.header("📥 Export Report")
                
                # Check if all issues have been reviewed
                all_reviewed = all(
                    st.session_state.review_states.get(r[2], {}).get('reviewed', False)
                    for r in st.session_state.validation_results
                    if r[0] in ['failed', 'warning']
                )
                
                if not all_reviewed and issues:
                    st.warning("⚠️ Please review all failed and warning items before exporting the final report.")
                
                report = generate_report(st.session_state.validation_results, st.session_state.documents, product_info)
                
                if report:
                    # Show review summary
                    if issues:
                        st.subheader("Review Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Issues Reviewed", f"{report['review_summary']['reviewed']}/{report['review_summary']['total_issues']}")
                        with col2:
                            st.metric("Fixed", report['review_summary']['fixed'])
                        with col3:
                            st.metric("Will Fix Later", report['review_summary']['will_fix_later'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = export_report_to_csv(report)
                        st.download_button(
                            label="Download CSV Report",
                            data=csv_data,
                            file_name=f"artwork_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            disabled=(not all_reviewed and len(issues) > 0)
                        )
                    
                    with col2:
                        json_data = json.dumps(report, indent=2)
                        st.download_button(
                            label="Download JSON Report",
                            data=json_data,
                            file_name=f"artwork_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            disabled=(not all_reviewed and len(issues) > 0)
                        )
                        
            except Exception as e:
                logger.error(f"Error displaying results: {e}")
                st.error(f"Error displaying results: {str(e)}")
        
        # Document preview section
        if st.session_state.documents:
            st.markdown("---")
            st.header("📄 Document Preview")
            
            doc_type = st.selectbox("Select document to preview", list(st.session_state.documents.keys()))
            
            if doc_type in st.session_state.documents:
                doc = st.session_state.documents[doc_type]
                
                with st.expander(f"View {doc['filename']}", expanded=False):
                    st.text_area("Extracted Text", doc['text'], height=300)
                    
                    if doc.get('images'):
                        st.subheader("Images")
                        for idx, img in enumerate(doc['images'][:3]):  # Limit preview images
                            try:
                                st.image(img, caption=f"Page {idx + 1}", use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                                
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        st.error("🚨 A critical error occurred. Please refresh the page and try again.")
        st.error(f"Error details: {str(e)}")
        
        # Attempt recovery
        if st.button("🔄 Reset Application"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crash: {e}", exc_info=True)
        st.error("The application encountered a critical error and needs to restart.")
        st.error("Please refresh the page to continue.")
