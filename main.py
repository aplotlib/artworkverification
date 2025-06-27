"""
Advanced Packaging Validation System for Vive Health
Multi-product support with cross-file validation and AI chat interface
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

# Global product database - expandable for all products
VIVE_PRODUCT_DATABASE = {
    "wheelchair_bag": {
        "name": "Wheelchair Bag Advanced",
        "sku_pattern": r"LVA3100[A-Z]{3}",
        "variants": {
            "BLACK": {"sku": "LVA3100BLK", "color_codes": ["black", "blk"]},
            "PURPLE FLORAL": {"sku": "LVA3100PUR", "color_codes": ["purple", "floral", "pur"]}
        },
        "materials": "60% Polyester, 20% PVC, 20% LDPE",
        "temp_range": "65°F to 85°F",
        "warranty": "1 year"
    },
    "alternating_pressure_mattress": {
        "name": "Alternating Air Pressure Mattress Pad",
        "sku_pattern": r"LVA1004[A-Z\-]*",
        "variants": {
            "STANDARD": {"sku": "LVA1004-UPC", "asin": "B00TZ73MUY"}
        }
    },
    "knee_scooter": {
        "name": "Knee Scooter",
        "sku_pattern": r"LVA1000[A-Z]{3}",
        "variants": {
            "BLACK": {"sku": "LVA1000BLK"},
            "BLUE": {"sku": "LVA1000BLU"},
            "RED": {"sku": "LVA1000RED"}
        }
    },
    "rollator": {
        "name": "Rollator Walker",
        "sku_pattern": r"LVA2000[A-Z]{3}",
        "variants": {
            "BLACK": {"sku": "LVA2000BLK"},
            "BLUE": {"sku": "LVA2000BLU"}
        }
    }
}

# Universal validation rules
UNIVERSAL_REQUIREMENTS = {
    "mandatory": [
        "Made in China",
        "vive® OR Vive Health",
        "vivehealth.com OR support website",
        "Contact information (phone/email)"
    ],
    "product_specific": {
        "packaging": ["Barcode/UPC", "SKU visible", "Product name", "Color identifier"],
        "washtag": ["Material composition", "Care instructions", "Temperature range"],
        "quickstart": ["Setup instructions", "Website/QR code", "Warranty info"],
        "manual": ["Safety warnings", "Product specifications", "Support contact"]
    }
}

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
        
        .validation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .validation-card {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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
        
        .insights-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .file-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .file-card {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .file-card:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }
        
        .progress-tracker {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .inconsistency-alert {
            background: #fffaf0;
            border: 2px solid #feb2b2;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from various sources"""
    keys = {}
    
    # Check streamlit secrets
    if hasattr(st, 'secrets'):
        for key_name in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
            if key_name in st.secrets:
                keys['openai'] = st.secrets[key_name]
                break
        
        for key_name in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key']:
            if key_name in st.secrets:
                keys['claude'] = st.secrets[key_name]
                break
    
    # Check environment variables
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

class ProductDetector:
    """Advanced product detection from filenames and content"""
    
    @staticmethod
    def detect_product(filename: str, text_content: str = "") -> Dict[str, Any]:
        """Detect product type and variant from filename and content"""
        result = {
            'product_type': None,
            'product_name': None,
            'variant': None,
            'color': None,
            'sku': None,
            'file_type': None,
            'confidence': 0
        }
        
        filename_lower = filename.lower()
        text_lower = text_content.lower() if text_content else ""
        
        # Detect product type
        for product_key, product_info in VIVE_PRODUCT_DATABASE.items():
            product_name_parts = product_info['name'].lower().split()
            
            # Check filename
            if any(part in filename_lower for part in product_name_parts[:2]):
                result['product_type'] = product_key
                result['product_name'] = product_info['name']
                result['confidence'] += 50
            
            # Check SKU pattern in text
            if text_content and re.search(product_info['sku_pattern'], text_content):
                result['product_type'] = product_key
                result['product_name'] = product_info['name']
                result['confidence'] += 30
                
                # Extract actual SKU
                sku_match = re.search(product_info['sku_pattern'], text_content)
                if sku_match:
                    result['sku'] = sku_match.group(0)
        
        # Detect variant/color
        if result['product_type'] and 'variants' in VIVE_PRODUCT_DATABASE[result['product_type']]:
            variants = VIVE_PRODUCT_DATABASE[result['product_type']]['variants']
            
            for variant_name, variant_info in variants.items():
                # Check color codes
                if 'color_codes' in variant_info:
                    for code in variant_info['color_codes']:
                        if code in filename_lower or code in text_lower:
                            result['variant'] = variant_name
                            result['color'] = variant_name
                            result['confidence'] += 20
                            break
        
        # Detect file type
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
                break
        
        return result

class CrossFileValidator:
    """Validates consistency across multiple files"""
    
    def __init__(self):
        self.inconsistencies = []
        self.product_groups = defaultdict(list)
        self.variant_groups = defaultdict(list)
    
    def add_file(self, filename: str, product_info: Dict, extracted_text: str):
        """Add a file to cross-validation groups"""
        product_type = product_info.get('product_type')
        variant = product_info.get('variant')
        
        if product_type:
            self.product_groups[product_type].append({
                'filename': filename,
                'info': product_info,
                'text': extracted_text
            })
        
        if variant:
            key = f"{product_type}_{variant}"
            self.variant_groups[key].append({
                'filename': filename,
                'info': product_info,
                'text': extracted_text
            })
    
    def validate_consistency(self) -> List[Dict]:
        """Check for inconsistencies across files"""
        inconsistencies = []
        
        # Check within product groups
        for product_type, files in self.product_groups.items():
            if len(files) > 1:
                # Check for consistent product naming
                product_names = [f['text'].lower() for f in files if f['text']]
                
                # Extract key information
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        # Check SKU consistency
                        if file1['info'].get('sku') and file2['info'].get('sku'):
                            if file1['info']['sku'] != file2['info']['sku']:
                                # This might be different variants - check if expected
                                if file1['info'].get('variant') != file2['info'].get('variant'):
                                    inconsistencies.append({
                                        'type': 'variant_difference',
                                        'severity': 'info',
                                        'files': [file1['filename'], file2['filename']],
                                        'message': f"Different variants detected: {file1['info'].get('variant')} vs {file2['info'].get('variant')}",
                                        'expected': True
                                    })
                                else:
                                    inconsistencies.append({
                                        'type': 'sku_mismatch',
                                        'severity': 'warning',
                                        'files': [file1['filename'], file2['filename']],
                                        'message': f"SKU mismatch for same variant: {file1['info']['sku']} vs {file2['info']['sku']}",
                                        'expected': False
                                    })
        
        # Check variant groups for consistency
        for variant_key, files in self.variant_groups.items():
            if len(files) > 1:
                # All files for same variant should have consistent info
                for field in ['sku', 'color']:
                    values = [f['info'].get(field) for f in files if f['info'].get(field)]
                    if len(set(values)) > 1:
                        inconsistencies.append({
                            'type': f'{field}_inconsistency',
                            'severity': 'error',
                            'files': [f['filename'] for f in files],
                            'message': f"Inconsistent {field} for {variant_key}: {', '.join(set(values))}",
                            'expected': False
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
    
    def create_comprehensive_prompt(self, filename: str, text: str, product_info: Dict, 
                                  all_files_context: List[Dict]) -> str:
        """Create context-aware validation prompt"""
        
        # Get product-specific requirements
        product_type = product_info.get('product_type')
        product_data = VIVE_PRODUCT_DATABASE.get(product_type, {})
        
        prompt = f"""You are a quality control expert for Vive Health medical devices. 
You're reviewing packaging files with awareness of the entire batch.

CURRENT FILE:
- Filename: {filename}
- Product: {product_info.get('product_name', 'Unknown')}
- Type: {product_info.get('file_type', 'Unknown')}
- Variant: {product_info.get('variant', 'Unknown')}
- SKU: {product_info.get('sku', 'Not detected')}

PRODUCT SPECIFICATIONS:
{json.dumps(product_data, indent=2)}

CONTEXT - OTHER FILES IN BATCH:
"""
        
        # Add context from other files
        for ctx in all_files_context[:5]:  # Limit to 5 for token management
            if ctx['filename'] != filename:
                prompt += f"\n- {ctx['filename']}: {ctx['product_info'].get('product_name')} - {ctx['product_info'].get('variant')}"
        
        prompt += f"""

EXTRACTED TEXT (first 2000 chars):
{text[:2000]}

VALIDATION REQUIREMENTS:
1. Universal Requirements:
   - Must have "Made in China"
   - Must have Vive branding (vive® or Vive Health)
   - Must have website (vivehealth.com)
   - Must have contact info

2. Product-Specific Requirements:
   - Check against product specifications above
   - Verify SKU format matches pattern
   - Confirm variant-specific details (color, size, etc.)

3. Cross-File Consistency:
   - Note any expected differences due to variants
   - Flag unexpected inconsistencies with other files

RESPONSE FORMAT (JSON):
{{
    "overall_assessment": "APPROVED" or "NEEDS_REVISION" or "REVIEW_REQUIRED",
    "product_confirmed": true/false,
    "variant_confirmed": "{product_info.get('variant')}",
    "sku_validation": {{
        "expected_pattern": "",
        "found": "",
        "valid": true/false
    }},
    "requirements_check": {{
        "made_in_china": true/false,
        "vive_branding": true/false,
        "website_correct": true/false,
        "contact_info": true/false
    }},
    "critical_issues": [],
    "warnings": [],
    "cross_file_notes": [],
    "variant_specific_notes": [],
    "improvement_suggestions": []
}}"""
        
        return prompt
    
    def validate_with_ai(self, filename: str, text: str, product_info: Dict) -> Dict:
        """Run AI validation with full context"""
        prompt = self.create_comprehensive_prompt(filename, text, product_info, self.context)
        
        results = {}
        
        # Try Claude first
        if 'claude' in self.api_keys and CLAUDE_AVAILABLE:
            try:
                client = anthropic.Anthropic(api_key=self.api_keys['claude'])
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    temperature=0.1,
                    system="You are a quality control expert. Always respond with valid JSON.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                results['claude'] = self._parse_response(response.content[0].text)
            except Exception as e:
                logger.error(f"Claude error: {e}")
                results['claude'] = {"error": str(e)}
        
        # Try OpenAI
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
        """Parse AI response"""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return {"error": "Failed to parse response", "raw": response_text}

class ChatInterface:
    """Interactive chat interface for Q&A about validation results"""
    
    def __init__(self, api_keys: Dict):
        self.api_keys = api_keys
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
    
    def add_message(self, role: str, content: str):
        """Add message to chat history"""
        st.session_state.chat_messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
    
    def get_context_prompt(self, user_question: str, validation_results: Dict) -> str:
        """Create context-aware prompt for chat"""
        
        # Summarize validation results
        summary = {
            'total_files': len(validation_results),
            'products': {},
            'issues': [],
            'cross_file_inconsistencies': []
        }
        
        for filename, results in validation_results.items():
            product = results.get('product_info', {}).get('product_name', 'Unknown')
            if product not in summary['products']:
                summary['products'][product] = {
                    'files': [],
                    'variants': set(),
                    'issues': []
                }
            
            summary['products'][product]['files'].append(filename)
            
            variant = results.get('product_info', {}).get('variant')
            if variant:
                summary['products'][product]['variants'].add(variant)
            
            # Collect issues
            for provider in ['claude', 'openai']:
                if provider in results and 'ai_results' in results:
                    ai_result = results['ai_results'].get(provider, {})
                    if 'critical_issues' in ai_result:
                        summary['issues'].extend(ai_result['critical_issues'])
        
        prompt = f"""You are a quality control expert assistant for Vive Health. 
You have access to packaging validation results for multiple files.

VALIDATION SUMMARY:
{json.dumps(summary, indent=2, default=str)}

DETAILED RESULTS AVAILABLE:
- Product detection and variant identification
- Cross-file consistency checks
- AI validation results from multiple providers
- Specific issues and warnings per file

USER QUESTION: {user_question}

Please provide a helpful, specific answer based on the validation results. 
If the user asks about specific files or issues, reference them directly.
Be concise but thorough. Suggest actionable next steps when relevant."""

        return prompt
    
    def process_message(self, user_message: str, validation_results: Dict) -> str:
        """Process user message and return AI response"""
        
        # Add user message to history
        self.add_message('user', user_message)
        
        # Create context-aware prompt
        prompt = self.get_context_prompt(user_message, validation_results)
        
        # Get AI response
        response = "I'm having trouble connecting to the AI service."
        
        if 'claude' in self.api_keys and CLAUDE_AVAILABLE:
            try:
                client = anthropic.Anthropic(api_key=self.api_keys['claude'])
                
                # Include chat history for context
                messages = []
                for msg in st.session_state.chat_messages[-5:]:  # Last 5 messages
                    if msg['role'] == 'user':
                        messages.append({"role": "user", "content": msg['content']})
                    else:
                        messages.append({"role": "assistant", "content": msg['content']})
                
                # Add current prompt
                messages.append({"role": "user", "content": prompt})
                
                ai_response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=messages
                )
                
                response = ai_response.content[0].text
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                response = f"Error: {str(e)}"
        
        # Add AI response to history
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
            except:
                pass
        
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
                file_bytes.seek(0)
            except:
                pass
        
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
            except:
                pass
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
    
    return extracted_text.strip(), extraction_method, page_count

def display_validation_dashboard(results: Dict, cross_validator: CrossFileValidator):
    """Display comprehensive validation dashboard"""
    
    # Summary metrics
    st.markdown("### 📊 Validation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_files = len(results)
    approved = sum(1 for r in results.values() 
                  if any(ai.get('overall_assessment') == 'APPROVED' 
                        for ai in r.get('ai_results', {}).values()))
    needs_revision = sum(1 for r in results.values() 
                        if any(ai.get('overall_assessment') == 'NEEDS_REVISION' 
                              for ai in r.get('ai_results', {}).values()))
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_files}</div>
            <div>Total Files</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #48bb78;">{approved}</div>
            <div>Approved</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #f56565;">{needs_revision}</div>
            <div>Needs Revision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        inconsistencies = cross_validator.validate_consistency()
        critical_count = len([i for i in inconsistencies if i['severity'] == 'error'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #ed8936;">{critical_count}</div>
            <div>Critical Issues</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Product breakdown
    st.markdown("### 🏷️ Products Detected")
    
    product_summary = defaultdict(lambda: {'count': 0, 'variants': set(), 'files': []})
    
    for filename, result in results.items():
        product_type = result.get('product_info', {}).get('product_type')
        if product_type:
            product_summary[product_type]['count'] += 1
            variant = result.get('product_info', {}).get('variant')
            if variant:
                product_summary[product_type]['variants'].add(variant)
            product_summary[product_type]['files'].append(filename)
    
    product_cols = st.columns(len(product_summary))
    for idx, (product_type, info) in enumerate(product_summary.items()):
        with product_cols[idx]:
            product_name = VIVE_PRODUCT_DATABASE.get(product_type, {}).get('name', product_type)
            st.markdown(f"""
            <div class="product-card">
                <h4>{product_name}</h4>
                <p>Files: {info['count']}</p>
                <p>Variants: {', '.join(info['variants']) if info['variants'] else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Cross-file validation results
    st.markdown("### 🔄 Cross-File Validation")
    
    inconsistencies = cross_validator.validate_consistency()
    
    if inconsistencies:
        # Group by severity
        errors = [i for i in inconsistencies if i['severity'] == 'error']
        warnings = [i for i in inconsistencies if i['severity'] == 'warning']
        info = [i for i in inconsistencies if i['severity'] == 'info']
        
        if errors:
            st.markdown("#### ❌ Critical Issues")
            for issue in errors:
                st.markdown(f"""
                <div class="cross-check-alert">
                    <strong>{issue['type'].replace('_', ' ').title()}</strong><br>
                    {issue['message']}<br>
                    <small>Files: {', '.join(issue['files'])}</small>
                </div>
                """, unsafe_allow_html=True)
        
        if warnings:
            st.markdown("#### ⚠️ Warnings")
            for issue in warnings:
                st.warning(f"{issue['message']} (Files: {', '.join(issue['files'])})")
        
        if info:
            st.markdown("#### ℹ️ Expected Variations")
            for issue in info:
                if issue.get('expected', False):
                    st.info(f"{issue['message']} - This is expected for different variants")
    else:
        st.success("✅ No cross-file inconsistencies detected!")

def display_file_results(filename: str, results: Dict):
    """Display detailed results for a single file"""
    
    product_info = results.get('product_info', {})
    
    # File info header
    st.markdown(f"""
    <div class="product-card">
        <h4>{filename}</h4>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
            <div>
                <strong>Product:</strong><br>
                {product_info.get('product_name', 'Unknown')}
            </div>
            <div>
                <strong>Variant:</strong><br>
                {product_info.get('variant', 'Not specified')}
            </div>
            <div>
                <strong>File Type:</strong><br>
                {product_info.get('file_type', 'Unknown')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Validation Results
    if 'ai_results' in results:
        st.markdown("#### 🤖 AI Validation Results")
        
        tabs = st.tabs([provider.title() for provider in results['ai_results'].keys()])
        
        for idx, (provider, ai_result) in enumerate(results['ai_results'].items()):
            with tabs[idx]:
                if 'error' in ai_result:
                    st.error(f"Error: {ai_result['error']}")
                else:
                    # Overall assessment
                    assessment = ai_result.get('overall_assessment', 'UNKNOWN')
                    assessment_color = {
                        'APPROVED': 'status-pass',
                        'NEEDS_REVISION': 'status-fail',
                        'REVIEW_REQUIRED': 'status-warning'
                    }.get(assessment, 'status-warning')
                    
                    st.markdown(f'<span class="status-badge {assessment_color}">{assessment}</span>', 
                              unsafe_allow_html=True)
                    
                    # Requirements check
                    req_check = ai_result.get('requirements_check', {})
                    if req_check:
                        cols = st.columns(4)
                        for idx, (req, status) in enumerate(req_check.items()):
                            with cols[idx % 4]:
                                if status:
                                    st.success(f"✅ {req.replace('_', ' ').title()}")
                                else:
                                    st.error(f"❌ {req.replace('_', ' ').title()}")
                    
                    # Issues and warnings
                    critical = ai_result.get('critical_issues', [])
                    if critical:
                        st.markdown("**Critical Issues:**")
                        for issue in critical:
                            st.error(f"• {issue}")
                    
                    warnings = ai_result.get('warnings', [])
                    if warnings:
                        st.markdown("**Warnings:**")
                        for warning in warnings:
                            st.warning(f"• {warning}")
                    
                    # Cross-file notes
                    cross_notes = ai_result.get('cross_file_notes', [])
                    if cross_notes:
                        st.markdown("**Cross-File Observations:**")
                        for note in cross_notes:
                            st.info(f"• {note}")
    
    # Extraction details
    with st.expander("📄 Extraction Details"):
        st.markdown(f"""
        - **Extraction Method:** {results.get('extraction_method', 'Unknown')}
        - **Pages:** {results.get('page_count', 0)}
        - **Text Length:** {len(results.get('text', ''))} characters
        - **Confidence Score:** {product_info.get('confidence', 0)}%
        """)
        
        if results.get('text'):
            st.text_area("Text Preview", results.get('text', '')[:500] + "...", height=200)

def main():
    inject_advanced_css()
    
    # Initialize session state
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = {}
    if 'cross_validator' not in st.session_state:
        st.session_state.cross_validator = CrossFileValidator()
    if 'ai_validator' not in st.session_state:
        st.session_state.ai_validator = None
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Vive Health Advanced Packaging Validator</h1>
        <p>Multi-product validation with AI-powered cross-file analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # AI Status
        if api_keys:
            if 'claude' in api_keys:
                st.success("✅ Claude AI Ready")
            if 'openai' in api_keys:
                st.success("✅ OpenAI Ready")
            
            # Initialize AI components
            if not st.session_state.ai_validator:
                st.session_state.ai_validator = AIValidator(api_keys)
            if not st.session_state.chat_interface:
                st.session_state.chat_interface = ChatInterface(api_keys)
        else:
            st.error("❌ No AI providers configured")
        
        st.markdown("---")
        
        # Product Database
        st.markdown("### 📦 Product Database")
        st.info(f"**{len(VIVE_PRODUCT_DATABASE)}** products configured")
        
        with st.expander("View Products"):
            for product_key, product_info in VIVE_PRODUCT_DATABASE.items():
                st.markdown(f"**{product_info['name']}**")
                st.markdown(f"- SKU Pattern: `{product_info['sku_pattern']}`")
                st.markdown(f"- Variants: {len(product_info.get('variants', {}))}")
        
        st.markdown("---")
        
        # Help
        st.markdown("### ❓ Help")
        st.info("""
        **Features:**
        - Multi-product support
        - Cross-file validation
        - Variant awareness
        - AI chat interface
        - Batch insights
        
        **File Types:**
        - Packaging artwork
        - Wash tags
        - Quick start guides
        - Manuals
        - QC sheets
        """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Validate", "📊 Results Dashboard", 
                                      "💬 AI Assistant", "📥 Export"])
    
    with tab1:
        st.markdown("### Upload Packaging Files")
        st.info("Upload all packaging files for comprehensive cross-validation")
        
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload packaging, labels, guides, and documentation"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{len(uploaded_files)}** files selected")
            
            with col2:
                if st.button("🚀 Validate All", type="primary", use_container_width=True):
                    with st.spinner("Processing files..."):
                        progress_bar = st.progress(0)
                        
                        # Reset validators
                        st.session_state.cross_validator = CrossFileValidator()
                        st.session_state.validation_results = {}
                        
                        for idx, file in enumerate(uploaded_files):
                            # Update progress
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                            
                            # Extract text
                            file.seek(0)
                            text, method, pages = extract_text_from_pdf(file, file.name)
                            
                            # Detect product
                            product_info = ProductDetector.detect_product(file.name, text)
                            
                            # Store results
                            results = {
                                'product_info': product_info,
                                'text': text,
                                'extraction_method': method,
                                'page_count': pages,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Add to cross-validator
                            st.session_state.cross_validator.add_file(
                                file.name, product_info, text
                            )
                            
                            # AI validation if text extracted
                            if text and st.session_state.ai_validator:
                                # Add context
                                st.session_state.ai_validator.add_context(
                                    file.name, product_info, text
                                )
                                
                                # Run validation
                                ai_results = st.session_state.ai_validator.validate_with_ai(
                                    file.name, text, product_info
                                )
                                results['ai_results'] = ai_results
                            
                            st.session_state.validation_results[file.name] = results
                        
                        st.success("✅ Validation complete!")
                        st.balloons()
    
    with tab2:
        if st.session_state.validation_results:
            display_validation_dashboard(
                st.session_state.validation_results,
                st.session_state.cross_validator
            )
            
            st.markdown("### 📁 Individual File Results")
            
            # File grid
            st.markdown('<div class="file-grid">', unsafe_allow_html=True)
            
            for filename in st.session_state.validation_results:
                if st.button(filename, key=f"file_{filename}", use_container_width=True):
                    with st.expander(f"Details: {filename}", expanded=True):
                        display_file_results(filename, st.session_state.validation_results[filename])
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload files in the first tab to see results")
    
    with tab3:
        st.markdown("### 💬 AI Quality Assistant")
        st.info("Ask questions about your validation results")
        
        if st.session_state.validation_results and st.session_state.chat_interface:
            # Display chat history
            for message in st.session_state.chat_messages:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">👤 {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-message">🤖 {message["content"]}</div>', 
                              unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_input("Ask about your validation results...", 
                                     placeholder="e.g., What are the main issues with the wheelchair bag packaging?")
            
            if user_input:
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_interface.process_message(
                        user_input, 
                        st.session_state.validation_results
                    )
                st.rerun()
            
            # Suggested questions
            st.markdown("#### 💡 Suggested Questions")
            suggestions = [
                "What are the most critical issues across all files?",
                "Are there any inconsistencies between color variants?",
                "Which files need immediate attention?",
                "Summarize the validation results by product type",
                "What improvements would you recommend?"
            ]
            
            cols = st.columns(2)
            for idx, suggestion in enumerate(suggestions):
                with cols[idx % 2]:
                    if st.button(suggestion, key=f"suggest_{idx}"):
                        st.session_state.chat_interface.process_message(
                            suggestion,
                            st.session_state.validation_results
                        )
                        st.rerun()
        else:
            st.warning("Please validate files first to use the AI assistant")
    
    with tab4:
        st.markdown("### 📥 Export Options")
        
        if st.session_state.validation_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON export
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'company': 'Vive Health',
                    'total_files': len(st.session_state.validation_results),
                    'cross_validation': st.session_state.cross_validator.validate_consistency(),
                    'results': st.session_state.validation_results
                }
                
                st.download_button(
                    "📄 Download JSON Report",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"vive_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Excel export
                if st.button("📊 Generate Excel Report", use_container_width=True):
                    st.info("Excel export coming soon!")
            
            with col3:
                # PDF report
                if st.button("📑 Generate PDF Report", use_container_width=True):
                    st.info("PDF export coming soon!")
            
            # Summary report
            st.markdown("### 📝 Executive Summary")
            
            summary = []
            summary.append("VIVE HEALTH PACKAGING VALIDATION REPORT")
            summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary.append("=" * 60)
            
            # Add insights
            insights = st.session_state.cross_validator.validate_consistency()
            critical_issues = [i for i in insights if i['severity'] == 'error']
            
            summary.append(f"\nFILES ANALYZED: {len(st.session_state.validation_results)}")
            summary.append(f"CRITICAL ISSUES: {len(critical_issues)}")
            summary.append(f"PRODUCTS FOUND: {len(st.session_state.cross_validator.product_groups)}")
            
            summary_text = "\n".join(summary)
            
            st.text_area("Report Preview", summary_text, height=300)
            
            st.download_button(
                "📥 Download Summary",
                data=summary_text,
                file_name=f"vive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("Upload and validate files to enable export options")

if __name__ == "__main__":
    main()
