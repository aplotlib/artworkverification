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

# Manual validation templates
MANUAL_TEMPLATES = {
    "packaging": {
        "title": "Packaging Artwork Manual Check",
        "fields": [
            {"name": "vive_logo", "label": "Vive Logo Present", "type": "checkbox"},
            {"name": "product_name", "label": "Product Name", "type": "text", "default": "Wheelchair Bag Advanced"},
            {"name": "color_variant", "label": "Color Variant", "type": "select", "options": ["BLACK", "PURPLE FLORAL"]},
            {"name": "sku", "label": "SKU Number", "type": "text", "placeholder": "LVA3100XXX"},
            {"name": "made_in_china", "label": "Made in China Text", "type": "checkbox"},
            {"name": "website_url", "label": "Website URL", "type": "text", "default": "vivehealth.com"},
            {"name": "barcode_present", "label": "Barcode/UPC Present", "type": "checkbox"},
            {"name": "ca_warning", "label": "CA Prop 65 Warning", "type": "checkbox"},
            {"name": "notes", "label": "Additional Notes", "type": "textarea"}
        ]
    },
    "washtag": {
        "title": "Wash Tag/Care Label Manual Check",
        "fields": [
            {"name": "materials", "label": "Material Composition", "type": "text", "default": "60% Polyester, 20% PVC, 20% LDPE"},
            {"name": "made_in_china", "label": "Made in China Text", "type": "checkbox"},
            {"name": "machine_wash", "label": "Machine Wash Instructions", "type": "checkbox"},
            {"name": "temp_range", "label": "Temperature Range", "type": "text", "default": "65°F to 85°F"},
            {"name": "care_symbols", "label": "Care Symbols Present", "type": "checkbox"},
            {"name": "notes", "label": "Additional Notes", "type": "textarea"}
        ]
    },
    "quickstart": {
        "title": "Quick Start Guide Manual Check",
        "fields": [
            {"name": "vive_logo", "label": "Vive Logo Present", "type": "checkbox"},
            {"name": "product_name", "label": "Product Name Consistent", "type": "checkbox"},
            {"name": "sku_displayed", "label": "SKU Displayed", "type": "text", "placeholder": "LVA3100"},
            {"name": "website", "label": "Website URL", "type": "text", "default": "vhealth.link/fxv"},
            {"name": "instructions", "label": "Application Instructions Clear", "type": "checkbox"},
            {"name": "warranty", "label": "Warranty Information", "type": "checkbox"},
            {"name": "contact_info", "label": "Contact Information", "type": "checkbox"},
            {"name": "notes", "label": "Additional Notes", "type": "textarea"}
        ]
    }
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
        
        .manual-input-box {
            background: #f0f8ff;
            border: 2px solid #4682b4;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .pdf-status {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
            color: #856404;
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

def extract_text_from_pdf_simple(file_bytes, filename=""):
    """Simple PDF text extraction with basic fallback"""
    extracted_text = ""
    pages_found = 0
    
    try:
        # Try PyPDF2 first (always available)
        file_bytes.seek(0)
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        pages_found = len(pdf_reader.pages)
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except:
                continue
        
        # Try pdfplumber if available and no text found
        if not extracted_text.strip() and PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                file_bytes.seek(0)
                with pdfplumber.open(file_bytes) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
            except:
                pass
        
        # Return result
        if extracted_text.strip():
            return extracted_text.strip(), "success", pages_found
        else:
            return "", "no_text", pages_found
            
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return "", "error", 0

def create_manual_validation_form(file_type, filename):
    """Create manual validation form for files where text extraction failed"""
    template = MANUAL_TEMPLATES.get(file_type, MANUAL_TEMPLATES['packaging'])
    
    st.markdown(f"### {template['title']}")
    st.info(f"Please manually review **{filename}** and fill out this checklist:")
    
    form_data = {}
    
    for field in template['fields']:
        if field['type'] == 'checkbox':
            form_data[field['name']] = st.checkbox(field['label'], key=f"{filename}_{field['name']}")
        elif field['type'] == 'text':
            default = field.get('default', '')
            placeholder = field.get('placeholder', '')
            form_data[field['name']] = st.text_input(
                field['label'], 
                value=default, 
                placeholder=placeholder,
                key=f"{filename}_{field['name']}"
            )
        elif field['type'] == 'select':
            form_data[field['name']] = st.selectbox(
                field['label'], 
                options=field['options'],
                key=f"{filename}_{field['name']}"
            )
        elif field['type'] == 'textarea':
            form_data[field['name']] = st.text_area(
                field['label'],
                key=f"{filename}_{field['name']}"
            )
    
    return form_data

def validate_manual_input(form_data, file_type):
    """Validate manual input and generate assessment"""
    issues = []
    warnings = []
    
    if file_type == 'packaging':
        if not form_data.get('vive_logo'):
            issues.append("Vive logo missing")
        if not form_data.get('made_in_china'):
            issues.append("Made in China text missing")
        if not form_data.get('barcode_present'):
            warnings.append("Barcode/UPC not confirmed")
        
        # Check SKU format
        sku = form_data.get('sku', '')
        if not sku.startswith('LVA3100'):
            issues.append(f"Invalid SKU format: {sku}")
            
    elif file_type == 'washtag':
        if not form_data.get('made_in_china'):
            issues.append("Made in China text missing")
        if form_data.get('materials') != "60% Polyester, 20% PVC, 20% LDPE":
            issues.append("Material composition incorrect")
            
    elif file_type == 'quickstart':
        if not form_data.get('vive_logo'):
            issues.append("Vive logo missing")
        if not form_data.get('warranty'):
            warnings.append("Warranty information not confirmed")
    
    # Determine overall assessment
    if issues:
        assessment = "NEEDS_REVISION"
    elif warnings:
        assessment = "REVIEW_REQUIRED"
    else:
        assessment = "APPROVED"
    
    return {
        'overall_assessment': assessment,
        'critical_issues': issues,
        'warnings': warnings,
        'form_data': form_data,
        'manual_review': True
    }

def create_vive_specific_prompt(text, filename, product_info, checklist_items):
    """Create Vive Health specific validation prompt"""
    if not text:
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
    
    parsed['error'] = f"Failed to parse {provider} response"
    return parsed

def call_claude(prompt, api_key):
    """Call Claude API"""
    if not CLAUDE_AVAILABLE or not api_key:
        logger.warning("Claude API not available or no API key")
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
        
        result = parse_ai_response(response.content[0].text, 'Claude')
        return result if result else {"error": "Failed to parse response", "overall_assessment": "ERROR"}
            
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def call_openai(prompt, api_key):
    """Call OpenAI API"""
    if not OPENAI_AVAILABLE or not api_key:
        logger.warning("OpenAI API not available or no API key")
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
        
        result = parse_ai_response(response.choices[0].message.content, 'OpenAI')
        return result if result else {"error": "Failed to parse response", "overall_assessment": "ERROR"}
            
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def display_vive_validation_results(results, provider, filename):
    """Display validation results with Vive Health specific formatting"""
    
    if "error" in results:
        st.error(f"**{provider} Error:** {results['error']}")
        return
    
    # Check if manual review
    if results.get('manual_review'):
        st.info("📝 Manual Review Results")
    
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
        # Safely get product info
        if isinstance(file_results, dict) and 'product_info' in file_results:
            product = file_results.get('product_info', {}).get('product', 'unknown')
        else:
            product = 'unknown'
            
        product_summary[product]['total'] += 1
        
        for provider in ['claude', 'openai', 'manual']:
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
        
        # Safely get product info
        if isinstance(file_results, dict) and 'product_info' in file_results:
            product_info = file_results.get('product_info', {})
            report.append(f"Product: {product_info.get('product', 'unknown').replace('_', ' ').title()}")
            report.append(f"Type: {product_info.get('type', 'unknown')}")
            report.append(f"Color: {product_info.get('color', 'unknown')}")
        else:
            report.append("Product: unknown")
            report.append("Type: unknown")
            report.append("Color: unknown")
        
        if file_results.get('extraction_failed'):
            report.append(f"STATUS: Text extraction failed - Manual review used")
        
        for provider in ['claude', 'openai', 'manual']:
            if provider in file_results and isinstance(file_results[provider], dict):
                ai_results = file_results[provider]
                report.append(f"\n{provider.upper()} Assessment: {ai_results.get('overall_assessment', 'UNKNOWN')}")
                
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
    if 'manual_reviews' not in st.session_state:
        st.session_state.manual_reviews = {}
    
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
            st.warning("No AI providers configured")
            st.info("Manual review mode available")
        
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
    st.markdown("### 🚀 How to Use")
    st.info("""
    1. Upload your packaging PDFs below
    2. The system will attempt to extract text and validate automatically
    3. For image-based PDFs, you'll get a manual review form
    4. Review results and download reports
    """)
    
    # Provider selection
    if available_providers:
        st.markdown("### 🤖 Select AI Providers")
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
    else:
        providers = []
        st.info("📝 Manual review mode - AI providers not configured")
    
    st.session_state.ai_providers = providers
    
    # File upload
    st.markdown("### 📤 Upload Packaging Files")
    
    uploaded_files = st.file_uploader(
        "Select PDF files (packaging, wash tags, quick start guides)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload all wheelchair bag packaging files for validation"
    )
    
    if uploaded_files:
        if st.button("🚀 Start Validation", type="primary", use_container_width=True):
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Extract text
                file.seek(0)
                text, extraction_status, pages = extract_text_from_pdf_simple(file, file.name)
                
                # Get product info
                product_info = detect_vive_product_type(file.name, text)
                
                # Store file results
                file_results = {
                    'product_info': product_info,
                    'pages': pages,
                    'extraction_status': extraction_status
                }
                
                # Check if we got text
                if text and extraction_status == "success":
                    file_results['text_preview'] = text[:500]
                    
                    # Determine checklist based on file type
                    file_type = product_info.get('type', 'packaging')
                    checklist_key = {
                        'packaging': 'Packaging Artwork',
                        'washtag': 'Wash Tag/Care Label',
                        'quickstart': 'Quick Start Guide'
                    }.get(file_type, 'Packaging Artwork')
                    
                    checklist_items = VIVE_VALIDATION_CHECKLIST.get(checklist_key, [])
                    
                    # Create prompt and call AI providers
                    if providers:
                        prompt = create_vive_specific_prompt(text, file.name, product_info, checklist_items)
                        
                        if prompt:
                            if 'claude' in providers:
                                with st.spinner(f"🤖 Claude reviewing {file.name}..."):
                                    claude_result = call_claude(prompt, api_keys['claude'])
                                    if claude_result:
                                        file_results['claude'] = claude_result
                                    time.sleep(0.5)
                            
                            if 'openai' in providers:
                                with st.spinner(f"🤖 OpenAI reviewing {file.name}..."):
                                    openai_result = call_openai(prompt, api_keys['openai'])
                                    if openai_result:
                                        file_results['openai'] = openai_result
                                    time.sleep(0.5)
                else:
                    # Text extraction failed - mark for manual review
                    file_results['extraction_failed'] = True
                    file_results['needs_manual_review'] = True
                    
                    if extraction_status == "no_text":
                        file_results['skip_reason'] = f"No text found in PDF ({pages} pages)"
                    else:
                        file_results['skip_reason'] = "PDF extraction failed"
                
                results[file.name] = file_results
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Initial processing complete!")
            st.session_state.validation_results = results
            st.rerun()
    
    # Display results and manual review forms
    if st.session_state.validation_results:
        st.markdown("---")
        st.markdown("## 📊 Validation Results")
        
        # Count files needing manual review
        manual_review_files = [
            (filename, data) for filename, data in st.session_state.validation_results.items()
            if data.get('needs_manual_review', False)
        ]
        
        if manual_review_files:
            st.markdown("### 📝 Manual Review Required")
            st.warning(f"{len(manual_review_files)} file(s) require manual review (image-based PDFs)")
            
            for filename, file_data in manual_review_files:
                with st.expander(f"📄 {filename} - Manual Review", expanded=True):
                    # Show PDF info
                    st.markdown(f"""
                    <div class="pdf-status">
                        <strong>Status:</strong> {file_data.get('skip_reason', 'Unknown')}<br>
                        <strong>Pages:</strong> {file_data.get('pages', 0)}<br>
                        <strong>Type:</strong> {file_data.get('product_info', {}).get('type', 'unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Get file type for appropriate form
                    file_type = file_data.get('product_info', {}).get('type', 'packaging')
                    
                    # Create manual validation form
                    with st.form(key=f"manual_{filename}"):
                        form_data = create_manual_validation_form(file_type, filename)
                        
                        if st.form_submit_button("Submit Review", type="primary"):
                            # Validate and store results
                            manual_results = validate_manual_input(form_data, file_type)
                            
                            # Update results
                            if filename not in st.session_state.manual_reviews:
                                st.session_state.manual_reviews[filename] = {}
                            
                            st.session_state.manual_reviews[filename] = manual_results
                            st.session_state.validation_results[filename]['manual'] = manual_results
                            st.success("✅ Manual review saved!")
                            st.rerun()
        
        # Display all results
        st.markdown("### 📋 All Results")
        
        # Summary metrics
        total_files = len(st.session_state.validation_results)
        approved_count = 0
        
        for r in st.session_state.validation_results.values():
            if isinstance(r, dict):
                for p in ['claude', 'openai', 'manual']:
                    if p in r and isinstance(r[p], dict) and r[p].get('overall_assessment') == 'APPROVED':
                        approved_count += 1
                        break
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", total_files)
        with col2:
            st.metric("Approved", approved_count)
        with col3:
            st.metric("Issues", total_files - approved_count)
        
        # Individual file results
        for filename, file_results in st.session_state.validation_results.items():
            if not isinstance(file_results, dict):
                continue
                
            with st.expander(f"📄 {filename}", expanded=True):
                # Product info card
                product_info = file_results.get('product_info', {})
                
                st.markdown(f"""
                <div class="product-card">
                    <strong>Product:</strong> {product_info.get('product', 'unknown').replace('_', ' ').title()}<br>
                    <strong>Type:</strong> {product_info.get('type', 'unknown')}<br>
                    <strong>Color:</strong> {product_info.get('color', 'unknown')}<br>
                    <strong>PDF Info:</strong> {file_results.get('pages', 0)} pages
                </div>
                """, unsafe_allow_html=True)
                
                # Display results from different sources
                displayed_any = False
                
                # AI results
                if 'claude' in file_results and file_results['claude'] is not None:
                    st.markdown("#### 🤖 Claude Validation")
                    display_vive_validation_results(file_results['claude'], 'Claude', filename)
                    displayed_any = True
                
                if 'openai' in file_results and file_results['openai'] is not None:
                    if displayed_any:
                        st.markdown("---")
                    st.markdown("#### 🤖 OpenAI Validation")
                    display_vive_validation_results(file_results['openai'], 'OpenAI', filename)
                    displayed_any = True
                
                # Manual results
                if 'manual' in file_results:
                    if displayed_any:
                        st.markdown("---")
                    st.markdown("#### 📝 Manual Review")
                    display_vive_validation_results(file_results['manual'], 'Manual', filename)
                    displayed_any = True
                
                if not displayed_any:
                    st.info("No validation results available. Please complete manual review above.")
        
        # Export options
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'company': 'Vive Health',
                    'providers': st.session_state.ai_providers,
                    'results': {}
                }
                
                # Clean results for JSON serialization
                for filename, file_results in st.session_state.validation_results.items():
                    if isinstance(file_results, dict):
                        export_data['results'][filename] = file_results
                
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"vive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error generating JSON export: {str(e)}")
        
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
