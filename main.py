"""
packaging_validator_fixed.py - Fixed AI-powered packaging and label validator
Resolves UNKNOWN results issue with better response parsing and error handling
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
        
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

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
    logger.info(f"Parsing {provider} response: {response_text[:200]}...")
    
    # Method 1: Try direct JSON parsing
    try:
        # Look for JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_text = json_match.group(0)
            result = json.loads(json_text)
            
            # Normalize keys to lowercase
            normalized = {}
            for key, value in result.items():
                normalized[key.lower().replace('_', '')] = value
            
            # Map to expected structure
            parsed = {
                'overall_assessment': normalized.get('overallassessment', normalized.get('overall', 'UNKNOWN')),
                'checklist_validation': normalized.get('checklistvalidation', {}),
                'critical_issues': normalized.get('criticalissues', []),
                'warnings': normalized.get('warnings', []),
                'spelling_errors': normalized.get('spellingerrors', []),
                'consistency_issues': normalized.get('consistencyissues', [])
            }
            
            logger.info(f"Successfully parsed JSON from {provider}")
            return parsed
    except Exception as e:
        logger.warning(f"JSON parsing failed for {provider}: {e}")
    
    # Method 2: Extract key information using patterns
    try:
        parsed = {
            'overall_assessment': 'UNKNOWN',
            'checklist_validation': {},
            'critical_issues': [],
            'warnings': [],
            'spelling_errors': [],
            'consistency_issues': []
        }
        
        # Extract overall assessment
        assessment_match = re.search(r'(APPROVED|NEEDS[_ ]REVISION|REVIEW[_ ]REQUIRED)', response_text, re.IGNORECASE)
        if assessment_match:
            parsed['overall_assessment'] = assessment_match.group(1).upper().replace(' ', '_')
        
        # Extract critical issues
        critical_match = re.search(r'critical.*?:(.*?)(?:warnings|spelling|$)', response_text, re.IGNORECASE | re.DOTALL)
        if critical_match:
            issues = re.findall(r'[-•]\s*(.+)', critical_match.group(1))
            parsed['critical_issues'] = [issue.strip() for issue in issues]
        
        # Extract warnings
        warning_match = re.search(r'warnings.*?:(.*?)(?:spelling|consistency|$)', response_text, re.IGNORECASE | re.DOTALL)
        if warning_match:
            warnings = re.findall(r'[-•]\s*(.+)', warning_match.group(1))
            parsed['warnings'] = [warning.strip() for warning in warnings]
        
        # Extract checklist items (✅ PASS, ❌ FAIL, ⚠️ UNSURE)
        checklist_items = re.findall(r'([✅❌⚠️])\s*(.+?):\s*(.+)', response_text)
        for status_emoji, item, explanation in checklist_items:
            status = 'PASS' if '✅' in status_emoji else 'FAIL' if '❌' in status_emoji else 'UNSURE'
            parsed['checklist_validation'][item.strip()] = {
                'status': status,
                'explanation': explanation.strip()
            }
        
        logger.info(f"Extracted partial data from {provider} using patterns")
        return parsed
        
    except Exception as e:
        logger.error(f"Pattern extraction failed for {provider}: {e}")
    
    # Method 3: Return minimal structure
    return {
        'overall_assessment': 'ERROR',
        'error': f"Failed to parse {provider} response",
        'raw_response': response_text[:500]
    }

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

def display_ai_results(results, provider):
    """Display AI validation results with better error handling"""
    
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
    
    # Debug info in development
    if st.checkbox(f"Show {provider} debug info", key=f"debug_{provider}"):
        st.markdown(f'<div class="debug-box">{json.dumps(results, indent=2)}</div>', unsafe_allow_html=True)
    
    # Checklist validation
    checklist = results.get('checklist_validation', {})
    if checklist:
        st.markdown("#### 📋 Checklist Validation")
        for item, details in checklist.items():
            if isinstance(details, dict):
                status = details.get('status', 'UNKNOWN')
                explanation = details.get('explanation', '')
            else:
                status = 'UNKNOWN'
                explanation = str(details)
            
            status_symbol = {'PASS': '✅', 'FAIL': '❌', 'UNSURE': '⚠️'}.get(status, '❓')
            status_class = {'PASS': 'checklist-pass', 'FAIL': 'checklist-fail', 'UNSURE': 'checklist-warning'}.get(status, 'checklist-warning')
            
            st.markdown(f'<div class="checklist-item {status_class}">{status_symbol} {item}: {explanation}</div>', unsafe_allow_html=True)
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
                    'text_preview': text[:500] + '...' if len(text) > 500 else text
                }
                
                if 'claude' in providers:
                    with st.spinner(f"🤖 Claude reviewing {file.name}..."):
                        claude_result = call_claude(prompt, api_keys['claude'])
                        file_results['claude'] = claude_result
                        time.sleep(0.5)  # Rate limiting
                
                if 'openai' in providers:
                    with st.spinner(f"🤖 OpenAI reviewing {file.name}..."):
                        openai_result = call_openai(prompt, api_keys['openai'])
                        file_results['openai'] = openai_result
                        time.sleep(0.5)  # Rate limiting
                
                results[file.name] = file_results
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ AI validation complete!")
            st.session_state.validation_results = results
            st.balloons()
    
    # Display results
    if st.session_state.validation_results:
        st.markdown("---")
        st.markdown("## 📊 AI Validation Results")
        
        for filename, file_results in st.session_state.validation_results.items():
            with st.expander(f"📄 {filename}", expanded=True):
                # File info
                file_info = file_results['file_info']
                st.markdown(f"**Type:** {file_info['type']} | **Color:** {file_info['color']} | **Text extracted:** {file_results['text_length']} chars")
                
                # Show results based on providers used
                if len(st.session_state.ai_providers) > 1 and 'claude' in file_results and 'openai' in file_results:
                    # Compare both
                    st.markdown("### 🔄 AI Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🤖 Claude Results")
                        display_ai_results(file_results['claude'], 'Claude')
                    
                    with col2:
                        st.markdown("#### 🤖 OpenAI Results")
                        display_ai_results(file_results['openai'], 'OpenAI')
                else:
                    # Single provider
                    if 'claude' in file_results:
                        st.markdown("#### 🤖 Claude Validation")
                        display_ai_results(file_results['claude'], 'Claude')
                    
                    if 'openai' in file_results:
                        st.markdown("#### 🤖 OpenAI Validation")
                        display_ai_results(file_results['openai'], 'OpenAI')
        
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

if __name__ == "__main__":
    main()
