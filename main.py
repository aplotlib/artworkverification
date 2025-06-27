"""
packaging_validator_fixed.py - Fixed AI-powered packaging and label validator
Resolves duplicate key error and improves overall stability
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Packaging Validator",
    page_icon="🤖",
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

# Validation checklist from the document
VALIDATION_CHECKLIST = {
    "Packaging Artwork": [
        "Product name is consistent across all artwork",
        "No misspelled words in content",
        "Origin is 'Made in China' by default",
        "Color identifier and size match SKU suffix (e.g., Beige-Small = SUP1030BGES)",
        "UDI Giftbox/UPC barcode is present and matches UPC serial",
        "Color code information is present"
    ],
    "Manual Artwork": [
        "Product name is consistent across artwork",
        "No misspelled words in content"
    ],
    "Washtag/Logo tag": [
        "Logo is present",
        "Washtag has care icons"
    ],
    "Made in China sticker": [
        "All products have Made in China sticker (unless rating label or washtag present)"
    ],
    "Shipping Mark": [
        "Format is SKU - QTY",
        "QR Code matches the SKU - QTY information"
    ],
    "Product QR Code": [
        "QR Code matches the SKU - QTY info"
    ],
    "Thank You Card": [
        "Thank you card needed for all Vive brand products"
    ]
}

# Common issues to check
COMMON_ISSUES = {
    "origin_errors": [
        "Made in Taiwan",
        "Made in Vietnam", 
        "Product of Taiwan"
    ],
    "required_elements": [
        "Vive",
        "Vive Health",
        "vivehealth.com"
    ],
    "spelling_errors": [
        ("recieve", "receive"),
        ("occured", "occurred"),
        ("seperate", "separate"),
        ("definately", "definitely"),
        ("managment", "management"),
        ("accomodate", "accommodate"),
        ("occassion", "occasion"),
        ("neccessary", "necessary")
    ]
}

def inject_css():
    """Inject CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        
        .provider-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }
        
        .claude-badge {
            background: #f3e5ff;
            color: #6b46c1;
        }
        
        .openai-badge {
            background: #e5f6ff;
            color: #0066cc;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from secrets or environment"""
    keys = {}
    
    try:
        if hasattr(st, 'secrets'):
            # Check for various key names
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
    
    # Fallback to environment
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file_bytes)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text:
                # Clean up text
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', page_text)
                text += page_text + "\n"
        
        # If no text was extracted, return a message
        if not text.strip():
            return "[No text could be extracted from PDF - file may be scanned or image-based]"
            
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return f"[Error extracting PDF: {str(e)}]"

def extract_file_info(filename):
    """Extract product and variant info from filename"""
    info = {
        'product': '',
        'variant': '',
        'color': '',
        'type': ''
    }
    
    name_lower = filename.lower()
    name_parts = name_lower.replace('_', ' ').replace('-', ' ').split()
    
    # Identify file type
    file_types = ['packaging', 'label', 'tag', 'manual', 'quickstart', 'shipping', 'washtag', 'giftbox']
    for ft in file_types:
        if ft in name_lower:
            info['type'] = ft
            break
    
    # Extract color
    colors = ['black', 'white', 'blue', 'red', 'purple', 'grey', 'gray', 'beige', 'floral']
    for color in colors:
        if color in name_lower:
            info['color'] = color
            break
    
    # Extract product
    product_parts = []
    skip_words = colors + file_types + ['pdf', 'png', 'jpg', 'jpeg', 'advanced']
    for part in name_parts:
        if part not in skip_words and len(part) > 2:
            product_parts.append(part)
    
    info['product'] = ' '.join(product_parts[:3])
    
    return info

def create_ai_prompt(text, filename, file_info, checklist_items):
    """Create comprehensive prompt for AI validation"""
    prompt = f"""You are a quality control expert reviewing packaging and label files for Vive Health medical devices.

FILE INFORMATION:
- Filename: {filename}
- Product: {file_info['product']}
- Type: {file_info['type']}
- Color variant: {file_info['color']}

EXTRACTED TEXT (first 2000 chars):
{text[:2000]}

VALIDATION CHECKLIST:
{json.dumps(checklist_items, indent=2)}

CRITICAL REQUIREMENTS:
1. Origin MUST be "Made in China" (not Taiwan, Vietnam, etc.)
2. Vive Health branding must be present
3. Color in content must match filename color variant
4. SKU format should be XXX####-COLOR (e.g., SUP1030BGES for Beige-Small)
5. No spelling errors
6. Product name consistency

IMPORTANT: You must respond with a valid JSON object with this exact structure:
{{
    "overall_assessment": "APPROVED" or "NEEDS_REVISION" or "REVIEW_REQUIRED",
    "checklist_validation": {{
        "item_name": {{
            "status": "PASS" or "FAIL" or "UNSURE",
            "explanation": "brief explanation"
        }}
    }},
    "critical_issues": ["list of critical issues"],
    "warnings": ["list of warnings"],
    "spelling_errors": ["list of spelling errors found"],
    "consistency_issues": ["list of consistency issues"]
}}

Analyze the file and provide your assessment."""

    return prompt

def parse_ai_response(response_text, provider):
    """Parse AI response with multiple fallback methods"""
    logger.info(f"Parsing {provider} response length: {len(response_text)}")
    
    # Initialize default structure
    parsed = {
        'overall_assessment': 'UNKNOWN',
        'checklist_validation': {},
        'critical_issues': [],
        'warnings': [],
        'spelling_errors': [],
        'consistency_issues': []
    }
    
    # Method 1: Try direct JSON parsing
    try:
        # Look for JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_text = json_match.group(0)
            # Clean up common JSON issues
            json_text = json_text.replace('\n', ' ').replace('\r', ' ')
            result = json.loads(json_text)
            
            # Handle different key formats (lowercase, uppercase, with/without underscores)
            for key in ['overall_assessment', 'overallAssessment', 'OVERALL_ASSESSMENT', 'overall']:
                if key in result:
                    parsed['overall_assessment'] = str(result[key]).upper().replace(' ', '_')
                    break
            
            for key in ['checklist_validation', 'checklistValidation', 'CHECKLIST_VALIDATION', 'checklist']:
                if key in result:
                    parsed['checklist_validation'] = result[key]
                    break
            
            for key in ['critical_issues', 'criticalIssues', 'CRITICAL_ISSUES', 'critical']:
                if key in result:
                    parsed['critical_issues'] = result[key] if isinstance(result[key], list) else [result[key]]
                    break
            
            for key in ['warnings', 'WARNINGS', 'warning']:
                if key in result:
                    parsed['warnings'] = result[key] if isinstance(result[key], list) else [result[key]]
                    break
            
            for key in ['spelling_errors', 'spellingErrors', 'SPELLING_ERRORS', 'spelling']:
                if key in result:
                    parsed['spelling_errors'] = result[key] if isinstance(result[key], list) else [result[key]]
                    break
            
            for key in ['consistency_issues', 'consistencyIssues', 'CONSISTENCY_ISSUES', 'consistency']:
                if key in result:
                    parsed['consistency_issues'] = result[key] if isinstance(result[key], list) else [result[key]]
                    break
            
            logger.info(f"Successfully parsed JSON from {provider}")
            return parsed
    except Exception as e:
        logger.warning(f"JSON parsing failed for {provider}: {e}")
    
    # Method 2: Extract key information using patterns
    try:
        # Extract overall assessment
        assessment_patterns = [
            r'overall[_\s]assessment["\s:]+([A-Z_\s]+)',
            r'"overall_assessment"\s*:\s*"([^"]+)"',
            r'Assessment:\s*([A-Z_\s]+)',
            r'(APPROVED|NEEDS[_\s]REVISION|REVIEW[_\s]REQUIRED)'
        ]
        
        for pattern in assessment_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                assessment = match.group(1).strip().upper().replace(' ', '_')
                if assessment in ['APPROVED', 'NEEDS_REVISION', 'REVIEW_REQUIRED']:
                    parsed['overall_assessment'] = assessment
                    break
        
        # Extract critical issues
        critical_section = re.search(r'critical[_\s]issues["\s:]+\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
        if critical_section:
            issues_text = critical_section.group(1)
            issues = re.findall(r'"([^"]+)"', issues_text)
            parsed['critical_issues'] = issues
        else:
            # Try line-based extraction
            critical_match = re.search(r'critical.*?:(.*?)(?:warnings|spelling|\Z)', response_text, re.IGNORECASE | re.DOTALL)
            if critical_match:
                issues = re.findall(r'[-•]\s*(.+)', critical_match.group(1))
                parsed['critical_issues'] = [issue.strip() for issue in issues]
        
        # Extract warnings
        warning_section = re.search(r'warnings["\s:]+\[(.*?)\]', response_text, re.IGNORECASE | re.DOTALL)
        if warning_section:
            warnings_text = warning_section.group(1)
            warnings = re.findall(r'"([^"]+)"', warnings_text)
            parsed['warnings'] = warnings
        
        # Extract checklist items
        checklist_patterns = [
            r'([✅❌⚠️])\s*([^:]+):\s*([^\n]+)',
            r'"([^"]+)":\s*\{\s*"status":\s*"(PASS|FAIL|UNSURE)"[^}]*"explanation":\s*"([^"]+)"',
            r'(PASS|FAIL|UNSURE):\s*([^:]+):\s*([^\n]+)'
        ]
        
        for pattern in checklist_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                for match in matches:
                    if len(match) == 3:
                        if match[0] in ['✅', '❌', '⚠️']:
                            status = 'PASS' if '✅' in match[0] else 'FAIL' if '❌' in match[0] else 'UNSURE'
                            item = match[1].strip()
                            explanation = match[2].strip()
                        else:
                            item = match[1].strip() if match[0] in ['PASS', 'FAIL', 'UNSURE'] else match[0].strip()
                            status = match[0] if match[0] in ['PASS', 'FAIL', 'UNSURE'] else match[1]
                            explanation = match[2].strip()
                        
                        parsed['checklist_validation'][item] = {
                            'status': status,
                            'explanation': explanation
                        }
        
        logger.info(f"Extracted data from {provider} using patterns")
        return parsed
        
    except Exception as e:
        logger.error(f"Pattern extraction failed for {provider}: {e}")
    
    # Method 3: Return with error info
    parsed['error'] = f"Failed to parse {provider} response completely"
    parsed['raw_response'] = response_text[:1000]
    return parsed

def call_claude(prompt, api_key):
    """Call Claude API with better error handling"""
    if not CLAUDE_AVAILABLE or not api_key:
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=2000,
            temperature=0.1,
            system="You are a quality control expert. Always respond with valid JSON only, no additional text.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        logger.info(f"Claude raw response: {response_text[:200]}...")
        
        # Parse the response
        return parse_ai_response(response_text, 'Claude')
            
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def call_openai(prompt, api_key):
    """Call OpenAI API with better error handling"""
    if not OPENAI_AVAILABLE or not api_key:
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a quality control expert. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"OpenAI raw response: {response_text[:200]}...")
        
        # Parse the response
        return parse_ai_response(response_text, 'OpenAI')
            
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {"error": str(e), "overall_assessment": "ERROR"}

def display_ai_results(results, provider, file_key="", unique_id=""):
    """Display AI validation results with better error handling and unique keys"""
    
    # Check for errors first
    if "error" in results:
        st.error(f"**{provider} Error:** {results['error']}")
        if "raw_response" in results:
            with st.expander("View raw response"):
                st.text(results['raw_response'])
        return
    
    # Overall assessment
    assessment = results.get('overall_assessment', 'UNKNOWN')
    assessment_color = {
        'APPROVED': 'success',
        'NEEDS_REVISION': 'error',
        'REVIEW_REQUIRED': 'warning',
        'ERROR': 'error',
        'UNKNOWN': 'info'
    }.get(assessment, 'info')
    
    st.markdown(f'<div class="validation-result {assessment_color}"><strong>Overall Assessment:</strong> {assessment}</div>', unsafe_allow_html=True)
    
    # Debug info in development - use unique key with UUID
    debug_key = f"debug_{provider}_{file_key}_{unique_id}_{uuid.uuid4().hex[:8]}"
    if st.checkbox(f"Show {provider} debug info", key=debug_key):
        st.markdown(f'<div class="debug-box">{json.dumps(results, indent=2)}</div>', unsafe_allow_html=True)
    
    # Checklist validation
    checklist = results.get('checklist_validation', {})
    if checklist:
        st.markdown("#### 📋 Checklist Validation")
        for item, details in checklist.items():
            if isinstance(details, dict):
                status = details.get('status', 'UNKNOWN')
                explanation = details.get('explanation', 'No explanation provided')
            else:
                # Handle string or other non-dict values
                status = 'UNKNOWN'
                explanation = str(details) if details else 'No details available'
            
            # Map status to symbol and class
            status_symbol = {'PASS': '✅', 'FAIL': '❌', 'UNSURE': '⚠️', 'UNKNOWN': '❓'}.get(status, '❓')
            status_class = {
                'PASS': 'checklist-pass', 
                'FAIL': 'checklist-fail', 
                'UNSURE': 'checklist-warning',
                'UNKNOWN': 'checklist-warning'
            }.get(status, 'checklist-warning')
            
            # Clean up item name (remove extra quotes or formatting)
            item_clean = str(item).strip('"\'')
            
            st.markdown(f'<div class="checklist-item {status_class}">{status_symbol} {item_clean}: {explanation}</div>', unsafe_allow_html=True)
    else:
        st.info("No checklist validation data available")
    
    # Critical issues
    critical = results.get('critical_issues', [])
    if critical:
        st.markdown("#### 🚨 Critical Issues")
        for issue in critical:
            st.markdown(f'<div class="validation-result error">❌ {issue}</div>', unsafe_allow_html=True)
    
    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        st.markdown("#### ⚠️ Warnings")
        for warning in warnings:
            st.markdown(f'<div class="validation-result warning">⚠️ {warning}</div>', unsafe_allow_html=True)
    
    # Spelling errors
    spelling = results.get('spelling_errors', [])
    if spelling:
        st.markdown("#### 📝 Spelling Errors")
        for error in spelling:
            st.markdown(f"- {error}")
    
    # Consistency issues
    consistency = results.get('consistency_issues', [])
    if consistency:
        st.markdown("#### 🔄 Consistency Issues")
        for issue in consistency:
            st.markdown(f"- {issue}")

def generate_summary_report(results):
    """Generate summary report of all validation results"""
    report = []
    report.append(f"AI PACKAGING VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    
    # Count overall results
    total_files = len(results)
    approved = 0
    needs_revision = 0
    review_required = 0
    
    for filename, file_results in results.items():
        for provider in ['claude', 'openai']:
            if provider in file_results and isinstance(file_results[provider], dict):
                assessment = file_results[provider].get('overall_assessment', 'UNKNOWN')
                if assessment == 'APPROVED':
                    approved += 1
                elif assessment == 'NEEDS_REVISION':
                    needs_revision += 1
                elif assessment == 'REVIEW_REQUIRED':
                    review_required += 1
    
    report.append(f"SUMMARY:")
    report.append(f"Total Files: {total_files}")
    report.append(f"Approved: {approved}")
    report.append(f"Needs Revision: {needs_revision}")
    report.append(f"Review Required: {review_required}")
    report.append("")
    report.append("=" * 60)
    report.append("")
    
    # Detailed results per file
    for filename, file_results in results.items():
        report.append(f"FILE: {filename}")
        report.append("-" * 40)
        
        for provider in ['claude', 'openai']:
            if provider in file_results and isinstance(file_results[provider], dict):
                ai_results = file_results[provider]
                report.append(f"\n{provider.upper()} Assessment: {ai_results.get('overall_assessment', 'UNKNOWN')}")
                
                # Critical issues
                critical = ai_results.get('critical_issues', [])
                if critical:
                    report.append("\nCritical Issues:")
                    for issue in critical:
                        report.append(f"  - {issue}")
                
                # Warnings
                warnings = ai_results.get('warnings', [])
                if warnings:
                    report.append("\nWarnings:")
                    for warning in warnings:
                        report.append(f"  - {warning}")
                
                # Spelling errors
                spelling = ai_results.get('spelling_errors', [])
                if spelling:
                    report.append("\nSpelling Errors:")
                    for error in spelling:
                        report.append(f"  - {error}")
        
        report.append("")
        report.append("=" * 60)
        report.append("")
    
    return "\n".join(report)

def test_ai_connection(api_keys):
    """Test AI connections with a simple prompt"""
    st.markdown("### 🧪 Testing AI Connections...")
    
    test_prompt = """Respond with this exact JSON:
{
    "overall_assessment": "APPROVED",
    "test": "success"
}"""
    
    results = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'claude' in api_keys:
            with st.spinner("Testing Claude..."):
                try:
                    client = anthropic.Anthropic(api_key=api_keys['claude'])
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=100,
                        messages=[{"role": "user", "content": test_prompt}]
                    )
                    st.success("✅ Claude connected!")
                    results['claude'] = True
                except Exception as e:
                    st.error(f"❌ Claude failed: {str(e)[:100]}")
                    results['claude'] = False
    
    with col2:
        if 'openai' in api_keys:
            with st.spinner("Testing OpenAI..."):
                try:
                    client = openai.OpenAI(api_key=api_keys['openai'])
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": test_prompt}],
                        max_tokens=100
                    )
                    st.success("✅ OpenAI connected!")
                    results['openai'] = True
                except Exception as e:
                    st.error(f"❌ OpenAI failed: {str(e)[:100]}")
                    results['openai'] = False
    
    return results

def main():
    inject_css()
    
    # Initialize session state
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'ai_providers' not in st.session_state:
        st.session_state.ai_providers = []
    if 'validation_counter' not in st.session_state:
        st.session_state.validation_counter = 0
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Packaging Validator</h1>
        <p>Intelligent validation using Claude and OpenAI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API availability
    api_keys = get_api_keys()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🤖 AI Providers")
        
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
        
        # Test connections button
        if st.button("🧪 Test AI Connections"):
            test_ai_connection(api_keys)
        
        st.markdown("---")
        
        # Checklist reference
        st.markdown("### 📋 Validation Checklist")
        for category, items in VALIDATION_CHECKLIST.items():
            with st.expander(category, expanded=False):
                for item in items:
                    st.markdown(f"• {item}")
    
    # Main content
    if not available_providers:
        st.warning("⚠️ Please configure at least one AI provider to use this tool.")
        return
    
    st.markdown("### 🤖 Select AI Provider(s)")
    
    col1, col2, col3 = st.columns(3)
    
    providers = []
    
    with col1:
        if 'claude' in available_providers:
            use_claude = st.checkbox("**Claude** (Anthropic)", value=True, help="Fast and accurate")
            if use_claude:
                providers.append('claude')
            st.markdown('<span class="provider-badge claude-badge">Claude 3 Haiku</span>', unsafe_allow_html=True)
    
    with col2:
        if 'openai' in available_providers:
            use_openai = st.checkbox("**OpenAI** (GPT-4)", value=True, help="Comprehensive analysis")
            if use_openai:
                providers.append('openai')
            st.markdown('<span class="provider-badge openai-badge">GPT-4 Mini</span>', unsafe_allow_html=True)
    
    with col3:
        if len(available_providers) >= 2:
            compare_both = st.checkbox("**Compare Both**", value=False, help="Get results from both AIs")
            if compare_both:
                providers = available_providers
    
    st.session_state.ai_providers = providers
    
    # File upload
    st.markdown("### 📤 Upload Files for AI Review")
    
    uploaded_files = st.file_uploader(
        "Select packaging files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload all packaging, label, and manual files for AI validation"
    )
    
    if uploaded_files and providers:
        # Validate button
        if st.button("🚀 Start AI Validation", type="primary", use_container_width=True):
            # Increment counter to ensure unique IDs
            st.session_state.validation_counter += 1
            validation_id = st.session_state.validation_counter
            
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                # Extract text
                file.seek(0)
                if file.type == 'application/pdf':
                    text = extract_text_from_pdf(file)
                else:
                    text = file.read().decode('utf-8', errors='ignore')
                
                # Check if text extraction was successful
                if not text or len(text) < 10:
                    st.warning(f"⚠️ Limited text extracted from {file.name}. Results may be incomplete.")
                elif "[Error extracting PDF" in text or "[No text could be extracted" in text:
                    st.error(f"❌ Could not extract text from {file.name}. File may be image-based or corrupted.")
                
                # Get file info
                file_info = extract_file_info(file.name)
                
                # Determine checklist items based on file type
                checklist_items = []
                for category, items in VALIDATION_CHECKLIST.items():
                    if file_info['type'] in category.lower() or not file_info['type']:
                        checklist_items.extend(items)
                
                if not checklist_items:
                    checklist_items = VALIDATION_CHECKLIST.get("Packaging Artwork", [])
                
                # Create prompt
                prompt = create_ai_prompt(text, file.name, file_info, checklist_items)
                
                # Call selected AI providers
                file_results = {
                    'file_info': file_info,
                    'text_length': len(text),
                    'text_preview': text[:500] + '...' if len(text) > 500 else text,
                    'validation_id': validation_id
                }
                
                if 'claude' in providers and 'claude' in api_keys:
                    with st.spinner(f"🤖 Claude reviewing {file.name}..."):
                        claude_result = call_claude(prompt, api_keys['claude'])
                        file_results['claude'] = claude_result
                        time.sleep(0.5)  # Rate limiting
                
                if 'openai' in providers and 'openai' in api_keys:
                    with st.spinner(f"🤖 OpenAI reviewing {file.name}..."):
                        openai_result = call_openai(prompt, api_keys['openai'])
                        file_results['openai'] = openai_result
                        time.sleep(0.5)  # Rate limiting
                
                results[file.name] = file_results
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            progress_bar.empty()  # Clean up progress bar
            status_text.empty()   # Clean up status text
            st.success("✅ AI validation complete!")
            st.session_state.validation_results = results
            st.balloons()
    
    elif uploaded_files and not providers:
        st.warning("⚠️ Please select at least one AI provider to validate files.")
    elif not uploaded_files and providers:
        st.info("📤 Upload files above to start AI validation.")
    
    # Display results
    if st.session_state.validation_results:
        st.markdown("---")
        st.markdown("## 📊 AI Validation Results")
        
        for filename, file_results in st.session_state.validation_results.items():
            with st.expander(f"📄 {filename}", expanded=True):
                # File info
                file_info = file_results['file_info']
                validation_id = file_results.get('validation_id', 0)
                st.markdown(f"**Type:** {file_info['type']} | **Color:** {file_info['color']} | **Text extracted:** {file_results['text_length']} chars")
                
                # Show results based on providers used
                if len(st.session_state.ai_providers) > 1 and 'claude' in file_results and 'openai' in file_results:
                    # Compare both
                    st.markdown("### 🔄 AI Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🤖 Claude Results")
                        display_ai_results(file_results['claude'], 'Claude', filename, f"compare_{validation_id}")
                    
                    with col2:
                        st.markdown("#### 🤖 OpenAI Results")
                        display_ai_results(file_results['openai'], 'OpenAI', filename, f"compare_{validation_id}")
                else:
                    # Single provider
                    if 'claude' in file_results:
                        st.markdown("#### 🤖 Claude Validation")
                        display_ai_results(file_results['claude'], 'Claude', filename, f"single_{validation_id}")
                    
                    if 'openai' in file_results:
                        st.markdown("#### 🤖 OpenAI Validation")
                        display_ai_results(file_results['openai'], 'OpenAI', filename, f"single_{validation_id}")
        
        # Export results
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'providers': st.session_state.ai_providers,
                'results': st.session_state.validation_results
            }
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"ai_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Summary export
            summary = generate_summary_report(st.session_state.validation_results)
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary,
                file_name=f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    else:
        # Instructions
        st.info("""
        ### 🚀 How to Use AI Validation
        
        1. **Select AI Provider(s)**: Choose Claude, OpenAI, or both for comparison
        2. **Upload Files**: Add your packaging PDFs and text files
        3. **Review Results**: AI will check against the full validation checklist
        4. **Export Reports**: Download detailed validation results
        
        **What AI Checks:**
        - ✅ Complete checklist validation
        - ✅ Spelling and grammar
        - ✅ Origin marking (Made in China)
        - ✅ Color consistency
        - ✅ SKU format validation
        - ✅ Brand presence
        - ✅ Overall quality assessment
        
        **Benefits of AI Review:**
        - 🎯 More thorough than pattern matching
        - 🔍 Catches subtle issues
        - 💡 Provides specific recommendations
        - ⚡ Fast and consistent
        """)

if __name__ == "__main__":
    main()
