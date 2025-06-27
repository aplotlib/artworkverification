"""
packaging_validator_enhanced.py - Enhanced AI-powered packaging and label validator
Optimized for Vive Health wheelchair bag and medical device packaging validation
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import json
import PyPDF2
from collections import defaultdict
import time
import base64
import io
import uuid

# Try to import chardet for encoding detection
try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress debug messages
logging.getLogger('PyPDF2').setLevel(logging.WARNING)
logging.getLogger('pdfplumber').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Page config
st.set_page_config(
    page_title="Vive Health Packaging Validator",
    page_icon="🏥",
    layout="wide"
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
    logger.warning("pdfplumber not available")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available")

# Try to import OCR libraries
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract OCR not available")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

# Vive Health specific validation rules
VIVE_PRODUCTS = {
    "wheelchair_bag": {
        "variants": ["BLACK", "PURPLE FLORAL"],
        "sku_prefix": "LVA3100",
        "required_text": ["Vive", "Wheelchair Bag Advanced", "Made in China"],
        "materials": ["60% Polyester", "20% PVC", "20% LDPE"],
        "care_instructions": ["Machine wash", "Wash cold", "Air dry", "Do not tumble dry"]
    }
}

# Enhanced validation checklist for Vive Health products
VIVE_VALIDATION_CHECKLIST = {
    "Packaging Artwork": [
        "Vive brand logo is present and correct",
        "Product name matches SKU (Wheelchair Bag Advanced)",
        "Color variant matches filename and content",
        "Made in China is clearly displayed",
        "Barcode/UPC is present and readable",
        "SKU format is correct (LVA3100XX for wheelchair bag)",
        "Website URL is vivehealth.com",
        "Social media handles are correct",
        "California Proposition 65 warning if required",
        "Distributed by information is present"
    ],
    "Wash Tag/Care Label": [
        "Material composition is correct (60% Polyester, 20% PVC, 20% LDPE)",
        "Made in China is present",
        "Care instructions are complete and correct",
        "Temperature range specified (65°F to 85°F)",
        "All required care symbols are present"
    ],
    "Quick Start Guide": [
        "Vive logo is present",
        "Product name is consistent",
        "SKU (LVA3100) is displayed",
        "Website URL is correct (vhealth.link/fxv)",
        "Application instructions are clear",
        "Care instructions match wash tag",
        "Warranty information is included",
        "Contact information is accurate"
    ]
}

# Common issues specific to Vive Health
VIVE_COMMON_ISSUES = {
    "branding_errors": [
        ("vive health", "Vive Health"),  # Incorrect capitalization
        ("VIVE HEALTH", "Vive Health"),  # All caps
        ("Vive", "vive®"),  # Missing trademark
    ],
    "url_errors": [
        ("vivehealth.com", "www.vivehealth.com"),
        ("vhealth.link", "vhealth.link/fxv"),
    ],
    "required_elements": [
        "Vive",
        "vive®",
        "Made in China",
        "1-800-487-3808",
        "service@vivehealth.com"
    ]
}

def inject_css():
    """Inject enhanced CSS styling for Vive Health branding"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #55c4cf 0%, #3ba0a8 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .vive-logo {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .product-card {
            background: #f8f9fa;
            border: 2px solid #55c4cf;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .validation-result {
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }
        
        .error { 
            background: #fee; 
            border-color: #f44336;
            color: #c62828;
        }
        
        .warning { 
            background: #fff3cd; 
            border-color: #ffc107;
            color: #856404;
        }
        
        .success { 
            background: #d4edda; 
            border-color: #28a745;
            color: #155724;
        }
        
        .info {
            background: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
        
        .checklist-item {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .checklist-pass {
            background: #d4edda;
            color: #155724;
        }
        
        .checklist-fail {
            background: #f8d7da;
            color: #721c24;
        }
        
        .checklist-warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .debug-box {
            background: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .ocr-status {
            background: #e3f2fd;
            border: 1px solid #1976d2;
            border-radius: 4px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            color: #0d47a1;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from secrets or environment"""
    keys = {}
    
    try:
        if hasattr(st, 'secrets'):
            for key_name in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                if key_name in st.secrets:
                    keys['openai'] = st.secrets[key_name]
                    break
            
            for key_name in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key']:
                if key_name in st.secrets:
                    keys['claude'] = st.secrets[key_name]
                    break
    except:
        pass
    
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

def detect_vive_product_type(filename, text=""):
    """Detect specific Vive Health product type from filename and content"""
    product_info = {
        'product': 'unknown',
        'variant': '',
        'color': '',
        'type': '',
        'sku_detected': ''
    }
    
    name_lower = filename.lower()
    text_lower = text.lower() if text else ""
    
    # Detect wheelchair bag
    if 'wheelchair' in name_lower or 'lva3100' in text_lower:
        product_info['product'] = 'wheelchair_bag'
        
        # Detect color variant
        if 'black' in name_lower or 'black' in text_lower:
            product_info['color'] = 'BLACK'
            product_info['variant'] = 'BLACK'
            product_info['sku_detected'] = 'LVA3100BLK'
        elif 'purple' in name_lower or 'floral' in name_lower or 'purple floral' in text_lower:
            product_info['color'] = 'PURPLE FLORAL'
            product_info['variant'] = 'PURPLE FLORAL'
            product_info['sku_detected'] = 'LVA3100PUR'
    
    # Detect file type
    if 'packaging' in name_lower or 'package' in name_lower:
        product_info['type'] = 'packaging'
    elif 'wash' in name_lower or 'tag' in name_lower or 'label' in name_lower:
        product_info['type'] = 'washtag'
    elif 'quick' in name_lower or 'guide' in name_lower or 'manual' in name_lower:
        product_info['type'] = 'quickstart'
    elif 'qc' in name_lower:
        product_info['type'] = 'qc_sheet'
    
    return product_info

def extract_text_from_pdf_enhanced(file_bytes, filename="", use_ocr=False):
    """Enhanced PDF text extraction with better handling for packaging files"""
    extracted_text = ""
    method_used = ""
    extraction_details = {
        'pages': 0,
        'images': 0,
        'text_chars': 0,
        'extraction_method': '',
        'is_image_based': False
    }
    
    try:
        # First analyze PDF structure
        file_bytes.seek(0)
        pdf_info = check_pdf_content_type(file_bytes)
        extraction_details.update(pdf_info)
        
        logger.info(f"Analyzing {filename}: {pdf_info['pdf_type']} - Pages: {pdf_info['num_pages']}, Images: {pdf_info['image_count']}, Text: {pdf_info['text_length']} chars")
        
        # If empty PDF
        if pdf_info['num_pages'] == 0:
            return "[Empty PDF - no pages found]", "error", extraction_details
        
        # Try text extraction methods
        if pdf_info['has_text'] or pdf_info['pdf_type'] == 'mixed':
            # Try pdfplumber first
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    file_bytes.seek(0)
                    with pdfplumber.open(file_bytes) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text += page_text + "\n"
                    
                    if extracted_text.strip():
                        method_used = "pdfplumber"
                        extraction_details['extraction_method'] = method_used
                        extraction_details['text_chars'] = len(extracted_text)
                except Exception as e:
                    logger.warning(f"pdfplumber failed: {e}")
            
            # Try PyMuPDF if needed
            if not extracted_text.strip() and PYMUPDF_AVAILABLE:
                try:
                    import fitz
                    file_bytes.seek(0)
                    pdf_document = fitz.open(stream=file_bytes.read(), filetype="pdf")
                    file_bytes.seek(0)
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        page_text = page.get_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                    
                    pdf_document.close()
                    
                    if extracted_text.strip():
                        method_used = "PyMuPDF"
                        extraction_details['extraction_method'] = method_used
                        extraction_details['text_chars'] = len(extracted_text)
                except Exception as e:
                    logger.warning(f"PyMuPDF failed: {e}")
            
            # Fall back to PyPDF2
            if not extracted_text.strip():
                try:
                    file_bytes.seek(0)
                    pdf_reader = PyPDF2.PdfReader(file_bytes)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
                    
                    if extracted_text.strip():
                        method_used = "PyPDF2"
                        extraction_details['extraction_method'] = method_used
                        extraction_details['text_chars'] = len(extracted_text)
                except Exception as e:
                    logger.warning(f"PyPDF2 failed: {e}")
        
        # Clean extracted text
        if extracted_text.strip():
            extracted_text = re.sub(r'\s+', ' ', extracted_text)
            extracted_text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', extracted_text)
            return extracted_text.strip(), method_used, extraction_details
        
        # If no text extracted and it's image-based, try OCR
        if pdf_info['pdf_type'] == 'image-based' and use_ocr:
            logger.info(f"Attempting OCR on {filename}...")
            file_bytes.seek(0)
            ocr_text, ocr_method = extract_text_with_ocr_enhanced(file_bytes, filename)
            if ocr_text and not ocr_text.startswith("["):
                extraction_details['extraction_method'] = ocr_method
                extraction_details['text_chars'] = len(ocr_text)
                return ocr_text, ocr_method, extraction_details
        
        # Return appropriate message
        if pdf_info['pdf_type'] == 'image-based':
            extraction_details['is_image_based'] = True
            return f"[Image-based PDF - {pdf_info['num_pages']} pages, {pdf_info['image_count']} images. Enable OCR to extract text]", "image-based", extraction_details
        else:
            return f"[No text extracted - {pdf_info['num_pages']} pages analyzed]", "no-text", extraction_details
            
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"[Extraction error: {str(e)}]", "error", extraction_details

def check_pdf_content_type(file_bytes):
    """Analyze PDF to determine its content type"""
    result = {
        'has_text': False,
        'has_images': False,
        'num_pages': 0,
        'image_count': 0,
        'text_length': 0,
        'pdf_type': 'empty'
    }
    
    try:
        file_bytes.seek(0)
        
        # Check with PyMuPDF if available
        if PYMUPDF_AVAILABLE:
            try:
                import fitz
                doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
                file_bytes.seek(0)
                
                result['num_pages'] = len(doc)
                
                for page in doc:
                    text = page.get_text()
                    if text and text.strip():
                        result['has_text'] = True
                        result['text_length'] += len(text)
                    
                    image_list = page.get_images()
                    if image_list:
                        result['has_images'] = True
                        result['image_count'] += len(image_list)
                
                doc.close()
            except Exception as e:
                logger.debug(f"PyMuPDF check failed: {e}")
        
        # Fallback to PyPDF2
        if not result['has_images'] and not result['has_text']:
            try:
                pdf_reader = PyPDF2.PdfReader(file_bytes)
                result['num_pages'] = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        result['has_text'] = True
                        result['text_length'] += len(text)
                    
                    # Check for images
                    if '/Resources' in page and '/XObject' in page['/Resources']:
                        xobject = page['/Resources']['/XObject'].get_object()
                        if xobject:
                            for obj in xobject:
                                if xobject[obj]['/Subtype'] == '/Image':
                                    result['has_images'] = True
                                    result['image_count'] += 1
            except:
                pass
        
        # Determine PDF type
        if result['has_text'] and result['has_images']:
            result['pdf_type'] = 'mixed'
        elif result['has_text']:
            result['pdf_type'] = 'text-based'
        elif result['has_images']:
            result['pdf_type'] = 'image-based'
        
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
    
    return result

def extract_text_with_ocr_enhanced(file_bytes, filename="", page_limit=5):
    """Enhanced OCR extraction with progress feedback"""
    if not PYMUPDF_AVAILABLE:
        return "[OCR requires PyMuPDF library]", "error"
    
    try:
        import fitz
        from PIL import Image
        import io
        
        doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
        file_bytes.seek(0)
        
        extracted_text = ""
        pages_to_process = min(len(doc), page_limit)
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        
        for page_num in range(pages_to_process):
            progress_placeholder.info(f"🔍 OCR Processing page {page_num + 1}/{pages_to_process} of {filename}...")
            
            page = doc[page_num]
            
            # Convert page to image with higher resolution
            mat = fitz.Matrix(3, 3)  # 3x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.pil_tobytes(format="PNG")
            img = Image.open(io.BytesIO(img_data))
            
            # Try OCR
            page_text = ""
            
            if TESSERACT_AVAILABLE:
                try:
                    import pytesseract
                    # Configure Tesseract for better results
                    custom_config = r'--oem 3 --psm 6'
                    page_text = pytesseract.image_to_string(img, config=custom_config)
                    if page_text.strip():
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    logger.warning(f"Tesseract failed on page {page_num + 1}: {e}")
            
            elif EASYOCR_AVAILABLE:
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'])
                    results = reader.readtext(img_data)
                    page_text = ' '.join([text[1] for text in results])
                    if page_text.strip():
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
        
        doc.close()
        progress_placeholder.empty()
        
        if extracted_text.strip():
            return extracted_text.strip(), "OCR"
        else:
            return "[OCR completed but no text found]", "ocr-no-text"
            
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return f"[OCR error: {str(e)}]", "error"

def create_vive_specific_prompt(text, filename, product_info, checklist_items):
    """Create Vive Health specific validation prompt"""
    if not text or text.startswith("["):
        return None
    
    prompt = f"""You are a quality control expert for Vive Health medical devices, specifically reviewing packaging files for the Wheelchair Bag Advanced product line.

COMPANY STANDARDS:
- Brand: Vive Health (registered trademark: vive®)
- Website: vivehealth.com
- Support: 1-800-487-3808
- Email: service@vivehealth.com
- All products MUST say "Made in China"

FILE INFORMATION:
- Filename: {filename}
- Product: {product_info.get('product', 'unknown')} 
- Type: {product_info.get('type', 'unknown')}
- Color variant: {product_info.get('color', 'unknown')}
- Expected SKU: {product_info.get('sku_detected', 'unknown')}

EXTRACTED TEXT (first 3000 chars):
{text[:3000]}

SPECIFIC VALIDATION REQUIREMENTS:
1. For Wheelchair Bag Advanced:
   - SKU format: LVA3100XXX (BLK for Black, PUR for Purple Floral)
   - Materials MUST be: 60% Polyester, 20% PVC, 20% LDPE
   - Temperature range: 65°F to 85°F
   - Must include warranty information (1 year)

2. Critical checks:
   - Origin MUST be "Made in China" (not Taiwan, Vietnam, etc.)
   - Vive or vive® branding must be present
   - Website URL must be exact: vivehealth.com or vhealth.link/fxv
   - California Prop 65 warning if applicable

VALIDATION CHECKLIST:
{json.dumps(checklist_items, indent=2)}

RESPONSE FORMAT - You must respond with valid JSON:
{{
    "overall_assessment": "APPROVED" or "NEEDS_REVISION" or "REVIEW_REQUIRED",
    "product_identified": "wheelchair_bag" or "unknown",
    "color_variant": "BLACK" or "PURPLE FLORAL" or "unknown",
    "checklist_validation": {{
        "item_name": {{
            "status": "PASS" or "FAIL" or "UNSURE",
            "explanation": "specific details"
        }}
    }},
    "critical_issues": ["list of critical issues"],
    "warnings": ["list of warnings"],
    "spelling_errors": ["list of spelling/branding errors"],
    "missing_elements": ["list of required but missing elements"],
    "sku_validation": {{
        "expected": "LVA3100XXX",
        "found": "actual SKU if found",
        "correct": true/false
    }}
}}

Analyze thoroughly and provide detailed feedback."""

    return prompt

def parse_ai_response(response_text, provider):
    """Parse AI response with robust error handling"""
    logger.info(f"Parsing {provider} response")
    
    parsed = {
        'overall_assessment': 'UNKNOWN',
        'product_identified': 'unknown',
        'color_variant': 'unknown',
        'checklist_validation': {},
        'critical_issues': [],
        'warnings': [],
        'spelling_errors': [],
        'missing_elements': [],
        'sku_validation': {}
    }
    
    try:
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_text = json_match.group(0)
            result = json.loads(json_text)
            
            # Map fields
            for key in parsed.keys():
                if key in result:
                    parsed[key] = result[key]
            
            return parsed
    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}")
    
    # Fallback pattern extraction
    try:
        # Extract assessment
        assessment_match = re.search(r'overall[_\s]assessment["\s:]+([A-Z_\s]+)', response_text, re.IGNORECASE)
        if assessment_match:
            parsed['overall_assessment'] = assessment_match.group(1).strip().upper().replace(' ', '_')
        
        # Extract issues
        critical_match = re.search(r'critical[_\s]issues["\s:]+\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
        if critical_match:
            issues = re.findall(r'"([^"]+)"', critical_match.group(1))
            parsed['critical_issues'] = issues
        
        return parsed
        
    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
    
    parsed['error'] = f"Failed to parse {provider} response"
    return parsed

def call_claude(prompt, api_key):
    """Call Claude API"""
    if not CLAUDE_AVAILABLE or not api_key:
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            temperature=0.1,
            system="You are a quality control expert for Vive Health. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return parse_ai_response(response.content[0].text, 'Claude')
            
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def call_openai(prompt, api_key):
    """Call OpenAI API"""
    if not OPENAI_AVAILABLE or not api_key:
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a quality control expert for Vive Health. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        return parse_ai_response(response.choices[0].message.content, 'OpenAI')
            
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def display_vive_validation_results(results, provider, filename):
    """Display validation results with Vive Health specific formatting"""
    
    if "error" in results:
        st.error(f"**{provider} Error:** {results['error']}")
        return
    
    # Overall assessment with color coding
    assessment = results.get('overall_assessment', 'UNKNOWN')
    assessment_color = {
        'APPROVED': 'success',
        'NEEDS_REVISION': 'error',
        'REVIEW_REQUIRED': 'warning',
        'ERROR': 'error',
        'UNKNOWN': 'info'
    }.get(assessment, 'info')
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f'<div class="validation-result {assessment_color}"><strong>Assessment:</strong> {assessment}</div>', unsafe_allow_html=True)
    
    with col2:
        product = results.get('product_identified', 'unknown')
        if product == 'wheelchair_bag':
            st.success(f"✅ Product: Wheelchair Bag")
        else:
            st.warning(f"❓ Product: {product}")
    
    with col3:
        color = results.get('color_variant', 'unknown')
        if color in ['BLACK', 'PURPLE FLORAL']:
            st.success(f"✅ Color: {color}")
        else:
            st.warning(f"❓ Color: {color}")
    
    # SKU Validation
    sku_info = results.get('sku_validation', {})
    if sku_info:
        expected = sku_info.get('expected', 'N/A')
        found = sku_info.get('found', 'Not found')
        correct = sku_info.get('correct', False)
        
        if correct:
            st.success(f"✅ SKU Correct: {found}")
        else:
            st.error(f"❌ SKU Issue - Expected: {expected}, Found: {found}")
    
    # Critical Issues
    critical = results.get('critical_issues', [])
    if critical:
        st.markdown("#### 🚨 Critical Issues")
        for issue in critical:
            st.markdown(f'<div class="validation-result error">❌ {issue}</div>', unsafe_allow_html=True)
    
    # Missing Elements
    missing = results.get('missing_elements', [])
    if missing:
        st.markdown("#### ⚠️ Missing Required Elements")
        for element in missing:
            st.markdown(f"- {element}")
    
    # Checklist validation
    checklist = results.get('checklist_validation', {})
    if checklist:
        with st.expander("📋 Detailed Checklist Results", expanded=True):
            for item, details in checklist.items():
                if isinstance(details, dict):
                    status = details.get('status', 'UNKNOWN')
                    explanation = details.get('explanation', '')
                    
                    status_symbol = {'PASS': '✅', 'FAIL': '❌', 'UNSURE': '⚠️'}.get(status, '❓')
                    status_class = {'PASS': 'checklist-pass', 'FAIL': 'checklist-fail', 'UNSURE': 'checklist-warning'}.get(status, 'checklist-warning')
                    
                    st.markdown(f'<div class="checklist-item {status_class}">{status_symbol} {item}: {explanation}</div>', unsafe_allow_html=True)
    
    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in warnings:
                st.markdown(f"- {warning}")

def generate_vive_summary_report(results):
    """Generate Vive Health specific summary report"""
    report = []
    report.append("VIVE HEALTH PACKAGING VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    
    # Count results by product
    product_summary = defaultdict(lambda: {'total': 0, 'approved': 0, 'needs_revision': 0, 'issues': []})
    
    for filename, file_results in results.items():
        product = file_results.get('product_info', {}).get('product', 'unknown')
        product_summary[product]['total'] += 1
        
        for provider in ['claude', 'openai']:
            if provider in file_results and isinstance(file_results[provider], dict):
                assessment = file_results[provider].get('overall_assessment', 'UNKNOWN')
                if assessment == 'APPROVED':
                    product_summary[product]['approved'] += 1
                elif assessment == 'NEEDS_REVISION':
                    product_summary[product]['needs_revision'] += 1
                
                # Collect critical issues
                critical = file_results[provider].get('critical_issues', [])
                product_summary[product]['issues'].extend(critical)
    
    # Product summary
    report.append("PRODUCT SUMMARY:")
    for product, stats in product_summary.items():
        report.append(f"\n{product.upper().replace('_', ' ')}:")
        report.append(f"  Total Files: {stats['total']}")
        report.append(f"  Approved: {stats['approved']}")
        report.append(f"  Needs Revision: {stats['needs_revision']}")
        if stats['issues']:
            report.append(f"  Common Issues:")
            for issue in set(stats['issues']):
                report.append(f"    - {issue}")
    
    report.append("\n" + "=" * 60 + "\n")
    
    # Detailed file results
    report.append("DETAILED FILE RESULTS:\n")
    
    for filename, file_results in results.items():
        report.append(f"FILE: {filename}")
        report.append("-" * 40)
        
        product_info = file_results.get('product_info', {})
        report.append(f"Product: {product_info.get('product', 'unknown')}")
        report.append(f"Type: {product_info.get('type', 'unknown')}")
        report.append(f"Color: {product_info.get('color', 'unknown')}")
        
        if file_results.get('extraction_failed'):
            report.append(f"STATUS: Text extraction failed")
            report.append(f"REASON: {file_results.get('skip_reason', 'Unknown')}")
        else:
            for provider in ['claude', 'openai']:
                if provider in file_results and isinstance(file_results[provider], dict):
                    ai_results = file_results[provider]
                    report.append(f"\n{provider.upper()} Assessment: {ai_results.get('overall_assessment', 'UNKNOWN')}")
                    
                    # SKU validation
                    sku_info = ai_results.get('sku_validation', {})
                    if sku_info:
                        report.append(f"SKU: Expected {sku_info.get('expected', 'N/A')}, Found {sku_info.get('found', 'N/A')}")
                    
                    # Critical issues
                    critical = ai_results.get('critical_issues', [])
                    if critical:
                        report.append("\nCritical Issues:")
                        for issue in critical:
                            report.append(f"  - {issue}")
                    
                    # Missing elements
                    missing = ai_results.get('missing_elements', [])
                    if missing:
                        report.append("\nMissing Elements:")
                        for element in missing:
                            report.append(f"  - {element}")
        
        report.append("\n" + "=" * 60 + "\n")
    
    return "\n".join(report)

def main():
    inject_css()
    
    # Initialize session state
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'ai_providers' not in st.session_state:
        st.session_state.ai_providers = []
    if 'use_ocr' not in st.session_state:
        st.session_state.use_ocr = False
    if 'ocr_reader' not in st.session_state:
        st.session_state.ocr_reader = None
    
    # Header with Vive branding
    st.markdown("""
    <div class="main-header">
        <div class="vive-logo">vive®</div>
        <h1>Packaging & Label Validator</h1>
        <p>AI-powered quality control for Vive Health products</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API availability
    api_keys = get_api_keys()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏥 Vive Health QC System")
        
        st.markdown("#### 🤖 AI Providers")
        
        available_providers = []
        if 'claude' in api_keys and CLAUDE_AVAILABLE:
            available_providers.append('claude')
            st.success("✅ Claude Available")
        else:
            st.error("❌ Claude Not Available")
        
        if 'openai' in api_keys and OPENAI_AVAILABLE:
            available_providers.append('openai')
            st.success("✅ OpenAI Available")
        else:
            st.error("❌ OpenAI Not Available")
        
        if not available_providers:
            st.error("No AI providers configured!")
            with st.expander("Setup Instructions"):
                st.markdown("""
                Add to `.streamlit/secrets.toml`:
                ```
                OPENAI_API_KEY = "sk-..."
                ANTHROPIC_API_KEY = "sk-ant-..."
                ```
                """)
        
        st.markdown("---")
        
        # PDF Libraries
        st.markdown("#### 📄 PDF Processing")
        
        pdf_status = []
        if PDFPLUMBER_AVAILABLE:
            pdf_status.append("✅ pdfplumber")
        if PYMUPDF_AVAILABLE:
            pdf_status.append("✅ PyMuPDF")
        pdf_status.append("✅ PyPDF2")
        
        st.success(" | ".join(pdf_status))
        
        # OCR Options
        st.markdown("#### 🔍 OCR Options")
        
        ocr_available = False
        if TESSERACT_AVAILABLE:
            st.success("✅ Tesseract OCR")
            ocr_available = True
        else:
            st.info("❌ Tesseract not installed")
        
        if EASYOCR_AVAILABLE:
            st.success("✅ EasyOCR")
            ocr_available = True
        else:
            st.info("❌ EasyOCR not installed")
        
        # OCR toggle with auto-enable suggestion
        if ocr_available:
            use_ocr = st.checkbox(
                "Enable OCR for image PDFs", 
                value=st.session_state.get('use_ocr', True),
                help="Recommended for packaging artwork PDFs"
            )
            st.session_state.use_ocr = use_ocr
            
            if use_ocr:
                st.info("🔍 OCR is enabled - will process image-based PDFs automatically")
        else:
            st.warning("⚠️ Install OCR tools to process packaging artwork PDFs")
            st.session_state.use_ocr = False
        
        st.markdown("---")
        
        # Product reference
        st.markdown("### 📦 Product Standards")
        
        with st.expander("Wheelchair Bag Advanced"):
            st.markdown("""
            **SKUs:**
            - LVA3100BLK (Black)
            - LVA3100PUR (Purple Floral)
            
            **Materials:**
            - 60% Polyester
            - 20% PVC
            - 20% LDPE
            
            **Requirements:**
            - Made in China
            - Temperature: 65°F-85°F
            - 1 year warranty
            - Website: vivehealth.com
            """)
        
        with st.expander("Validation Checklist"):
            for category, items in VIVE_VALIDATION_CHECKLIST.items():
                st.markdown(f"**{category}:**")
                for item in items:
                    st.markdown(f"• {item}")
    
    # Main content area
    if not available_providers:
        st.warning("⚠️ Please configure at least one AI provider to use this tool.")
        return
    
    # Provider selection
    col1, col2 = st.columns(2)
    
    providers = []
    with col1:
        if 'claude' in available_providers:
            use_claude = st.checkbox("**Use Claude AI**", value=True)
            if use_claude:
                providers.append('claude')
    
    with col2:
        if 'openai' in available_providers:
            use_openai = st.checkbox("**Use OpenAI**", value=True)
            if use_openai:
                providers.append('openai')
    
    st.session_state.ai_providers = providers
    
    # File upload
    st.markdown("### 📤 Upload Packaging Files")
    
    uploaded_files = st.file_uploader(
        "Select PDF files (packaging, wash tags, quick start guides)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload all wheelchair bag packaging files for validation"
    )
    
    # Show current OCR status
    if uploaded_files and st.session_state.use_ocr:
        st.markdown('<div class="ocr-status">🔍 OCR is enabled - image-based PDFs will be processed automatically</div>', unsafe_allow_html=True)
    
    if uploaded_files and providers:
        if st.button("🚀 Start Validation", type="primary", use_container_width=True):
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Extract text with enhanced method
                file.seek(0)
                text, method, extraction_details = extract_text_from_pdf_enhanced(
                    file, 
                    file.name, 
                    use_ocr=st.session_state.use_ocr
                )
                
                # Get product info
                product_info = detect_vive_product_type(file.name, text)
                
                # Store file results
                file_results = {
                    'product_info': product_info,
                    'extraction_details': extraction_details,
                    'text_preview': text[:500] if not text.startswith("[") else text
                }
                
                # Check if extraction failed
                if text.startswith("[") and text.endswith("]"):
                    file_results['extraction_failed'] = True
                    file_results['skip_reason'] = text
                    
                    # Show specific message for image-based PDFs
                    if "Image-based PDF" in text and not st.session_state.use_ocr:
                        st.warning(f"⚠️ {file.name} is an image-based PDF. Enable OCR in the sidebar to extract text.")
                else:
                    # Determine checklist based on file type
                    file_type = product_info.get('type', 'packaging')
                    checklist_key = {
                        'packaging': 'Packaging Artwork',
                        'washtag': 'Wash Tag/Care Label',
                        'quickstart': 'Quick Start Guide'
                    }.get(file_type, 'Packaging Artwork')
                    
                    checklist_items = VIVE_VALIDATION_CHECKLIST.get(checklist_key, [])
                    
                    # Create prompt
                    prompt = create_vive_specific_prompt(text, file.name, product_info, checklist_items)
                    
                    if prompt:
                        # Call AI providers
                        if 'claude' in providers:
                            with st.spinner(f"🤖 Claude reviewing {file.name}..."):
                                claude_result = call_claude(prompt, api_keys['claude'])
                                file_results['claude'] = claude_result
                                time.sleep(0.5)
                        
                        if 'openai' in providers:
                            with st.spinner(f"🤖 OpenAI reviewing {file.name}..."):
                                openai_result = call_openai(prompt, api_keys['openai'])
                                file_results['openai'] = openai_result
                                time.sleep(0.5)
                
                results[file.name] = file_results
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Validation complete!")
            st.session_state.validation_results = results
            st.balloons()
    
    # Display results
    if st.session_state.validation_results:
        st.markdown("---")
        st.markdown("## 📊 Validation Results")
        
        # Summary metrics
        total_files = len(st.session_state.validation_results)
        approved_count = sum(1 for r in st.session_state.validation_results.values() 
                           for p in ['claude', 'openai'] 
                           if p in r and r[p].get('overall_assessment') == 'APPROVED')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Approved", approved_count)
        with col3:
            st.metric("Issues", total_files - approved_count)
        
        # Individual file results
        for filename, file_results in st.session_state.validation_results.items():
            with st.expander(f"📄 {filename}", expanded=True):
                # Product info card
                product_info = file_results['product_info']
                extraction_details = file_results.get('extraction_details', {})
                
                st.markdown(f"""
                <div class="product-card">
                    <strong>Product:</strong> {product_info.get('product', 'unknown').replace('_', ' ').title()}<br>
                    <strong>Type:</strong> {product_info.get('type', 'unknown')}<br>
                    <strong>Color:</strong> {product_info.get('color', 'unknown')}<br>
                    <strong>PDF Info:</strong> {extraction_details.get('pages', 0)} pages, 
                    {extraction_details.get('images', 0)} images, 
                    {extraction_details.get('text_chars', 0)} text chars
                    {' (Image-based)' if extraction_details.get('is_image_based') else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Check if extraction failed
                if file_results.get('extraction_failed'):
                    st.error("**Text Extraction Failed**")
                    reason = file_results.get('skip_reason', 'Unknown error')
                    st.info(f"**Reason:** {reason}")
                    
                    if "Image-based PDF" in reason and not st.session_state.use_ocr:
                        st.markdown("""
                        ### 💡 Solution:
                        1. Enable OCR in the sidebar
                        2. Re-upload and process the file
                        
                        Or use Adobe Acrobat to add text layer to the PDF.
                        """)
                    continue
                
                # Display AI results
                if 'claude' in file_results:
                    st.markdown("#### 🤖 Claude Validation")
                    display_vive_validation_results(file_results['claude'], 'Claude', filename)
                
                if 'openai' in file_results:
                    if 'claude' in file_results:
                        st.markdown("---")
                    st.markdown("#### 🤖 OpenAI Validation")
                    display_vive_validation_results(file_results['openai'], 'OpenAI', filename)
        
        # Export options
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'company': 'Vive Health',
                'providers': st.session_state.ai_providers,
                'results': st.session_state.validation_results
            }
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"vive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Summary report
            summary = generate_vive_summary_report(st.session_state.validation_results)
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary,
                file_name=f"vive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
