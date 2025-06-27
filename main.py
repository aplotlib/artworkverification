"""
Packaging Validator for Vive Health - WORKING VERSION
Fixed text extraction and chat functionality
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
import logging
import os
from typing import Dict, List, Any, Optional
import json
import PyPDF2
from collections import defaultdict
import time
import io
try:
    import chardet
except:
    chardet = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Try to import PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

def get_api_keys():
    """Get API keys from secrets or environment"""
    keys = {}
    
    # Check streamlit secrets first
    if hasattr(st, 'secrets'):
        keys['openai'] = st.secrets.get('OPENAI_API_KEY') or st.secrets.get('openai_api_key')
        keys['claude'] = st.secrets.get('ANTHROPIC_API_KEY') or st.secrets.get('anthropic_api_key')
    
    # Check environment
    if not keys.get('openai'):
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if not keys.get('claude'):
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

def extract_text_from_pdf_enhanced(file_bytes, filename=""):
    """Enhanced PDF text extraction with multiple methods"""
    text = ""
    pages = 0
    method = "none"
    
    # Log what we're trying
    logger.info(f"Attempting to extract text from {filename}")
    
    try:
        # Try PyMuPDF first if available
        if PYMUPDF_AVAILABLE:
            try:
                import fitz
                file_bytes.seek(0)
                doc = fitz.open(stream=file_bytes.read(), filetype="pdf")
                pages = len(doc)
                
                for page_num in range(pages):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                doc.close()
                file_bytes.seek(0)
                
                if text.strip():
                    method = "PyMuPDF"
                    logger.info(f"Extracted {len(text)} chars using PyMuPDF from {filename}")
                    
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")
        
        # Try pdfplumber
        if not text.strip() and PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                file_bytes.seek(0)
                with pdfplumber.open(file_bytes) as pdf:
                    pages = len(pdf.pages)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    method = "pdfplumber"
                    logger.info(f"Extracted {len(text)} chars using pdfplumber from {filename}")
                    
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}")
        
        # Try PyPDF2 as last resort
        if not text.strip():
            try:
                file_bytes.seek(0)
                reader = PyPDF2.PdfReader(file_bytes)
                pages = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except:
                        continue
                
                if text.strip():
                    method = "PyPDF2"
                    logger.info(f"Extracted {len(text)} chars using PyPDF2 from {filename}")
                    
            except Exception as e:
                logger.warning(f"PyPDF2 failed: {e}")
    
    except Exception as e:
        logger.error(f"All PDF extraction methods failed for {filename}: {e}")
    
    # Clean the text
    if text:
        # Remove null bytes and weird characters
        text = text.replace('\x00', '').replace('\u200b', '')
        # Normalize whitespace
        text = ' '.join(text.split())
    
    logger.info(f"Final result for {filename}: {len(text)} chars extracted using {method}")
    
    return text.strip(), pages, method

def manual_text_input_form(filename):
    """Provide manual text input when extraction fails"""
    st.warning(f"⚠️ Could not extract text from {filename}")
    st.info("This might be an image-based PDF. Please enter key information manually:")
    
    manual_text = st.text_area(
        "Paste or type the text content from the PDF:",
        placeholder="Enter product name, SKU, Made in China text, etc.",
        key=f"manual_{filename}",
        height=200
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        has_made_in_china = st.checkbox("Contains 'Made in China'", key=f"china_{filename}")
        has_vive_branding = st.checkbox("Contains Vive branding", key=f"vive_{filename}")
        has_website = st.checkbox("Contains website URL", key=f"web_{filename}")
    
    with col2:
        sku = st.text_input("SKU/Model (if visible):", key=f"sku_{filename}")
        product_name = st.text_input("Product name:", key=f"product_{filename}")
    
    if manual_text or sku or product_name:
        # Create synthetic text for validation
        synthetic_text = f"{manual_text}\n"
        if product_name:
            synthetic_text += f"Product: {product_name}\n"
        if sku:
            synthetic_text += f"SKU: {sku}\n"
        if has_made_in_china:
            synthetic_text += "Made in China\n"
        if has_vive_branding:
            synthetic_text += "vive®\n"
        if has_website:
            synthetic_text += "vivehealth.com\n"
        
        return synthetic_text.strip()
    
    return ""

def create_validation_prompt(filename, text, other_files=[]):
    """Create validation prompt for AI"""
    prompt = f"""You are a quality control expert validating packaging for Vive Health medical devices.

FILE: {filename}
TEXT CONTENT ({len(text)} characters):
{text[:2000] if text else 'No text content'}

OTHER FILES IN BATCH: {len(other_files)}

CHECK THESE REQUIREMENTS:
1. Made in China text
2. Vive or vive® branding  
3. Website (vivehealth.com)
4. Contact information
5. Product identifier (SKU/Model)
6. Safety/regulatory marks

RESPOND WITH JSON ONLY:
{{
    "product_name": "detected product name or Unknown",
    "sku_found": "any SKU/model found or None",
    "status": "PASS or FAIL or NEEDS_REVIEW",
    "requirements": {{
        "made_in_china": true/false,
        "vive_branding": true/false,
        "website": true/false,
        "contact_info": true/false,
        "product_id": true/false
    }},
    "issues": ["list of problems found"],
    "suggestions": ["list of fixes needed"]
}}"""
    
    return prompt

def validate_with_ai(filename, text, api_keys):
    """Run AI validation with both providers"""
    results = {}
    
    if not text:
        return {"error": "No text to validate"}
    
    prompt = create_validation_prompt(filename, text)
    
    # Try OpenAI first (more reliable)
    if 'openai' in api_keys and OPENAI_AVAILABLE:
        try:
            client = openai.OpenAI(api_key=api_keys['openai'])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a packaging validation expert. Respond only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                results['openai'] = json.loads(json_match.group(0))
            else:
                results['openai'] = {"error": "Failed to parse response"}
                
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            results['openai'] = {"error": str(e)}
    
    # Try Claude
    if 'claude' in api_keys and CLAUDE_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=api_keys['claude'])
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # CORRECT MODEL
                max_tokens=1000,
                temperature=0.1,
                system="You are a packaging validation expert. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                results['claude'] = json.loads(json_match.group(0))
            else:
                results['claude'] = {"error": "Failed to parse response"}
                
        except Exception as e:
            logger.error(f"Claude error: {e}")
            results['claude'] = {"error": str(e)}
    
    return results

def ai_chat(user_message, context, api_keys):
    """Simple chat function that WORKS"""
    
    prompt = f"""You are a helpful quality assistant for Vive Health packaging validation.

CONTEXT:
{context}

USER QUESTION: {user_message}

Provide a helpful, specific answer based on the validation results."""
    
    # Try OpenAI first (more reliable)
    if 'openai' in api_keys and OPENAI_AVAILABLE:
        try:
            client = openai.OpenAI(api_key=api_keys['openai'])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful quality control assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
    
    # Try Claude as backup
    if 'claude' in api_keys and CLAUDE_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=api_keys['claude'])
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # CORRECT MODEL NAME
                max_tokens=500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude chat error: {e}")
    
    return "Sorry, I couldn't connect to the AI service. Please check your API keys."

def main():
    st.markdown("""
    <style>
        .main-header {
            background: #2c3e50;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        .user-msg {
            background: #3498db;
            color: white;
            margin-left: 20%;
        }
        .ai-msg {
            background: #ecf0f1;
            margin-right: 20%;
        }
        .success { color: #27ae60; }
        .error { color: #e74c3c; }
        .warning { color: #f39c12; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Vive Health Packaging Validator</h1>
        <p>Universal validation for all products</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        if api_keys:
            if 'openai' in api_keys:
                st.success("✅ OpenAI Connected")
            if 'claude' in api_keys:
                st.success("✅ Claude Connected")
        else:
            st.error("❌ No AI providers configured")
            st.code("""
# Add to .streamlit/secrets.toml:
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
            """)
        
        st.markdown("---")
        st.markdown("### 📋 Requirements Checked")
        st.markdown("""
        - Made in China
        - Vive branding
        - Website URL
        - Contact info
        - Product ID/SKU
        - Safety marks
        """)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Validate", "💬 AI Assistant", "📊 Results"])
    
    with tab1:
        st.markdown("### Upload Packaging Files")
        
        uploaded_files = st.file_uploader(
            "Select PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Start Validation", type="primary"):
                st.session_state.results = {}
                
                # Process each file
                for file in uploaded_files:
                    with st.expander(f"Processing {file.name}...", expanded=True):
                        # Extract text
                        file.seek(0)
                        text, pages, method = extract_text_from_pdf_enhanced(file, file.name)
                        
                        st.info(f"📄 Pages: {pages} | Method: {method} | Text length: {len(text)}")
                        
                        # If no text extracted, provide manual input
                        if not text:
                            st.warning("No text could be extracted automatically.")
                            text = manual_text_input_form(file.name)
                            
                            if st.button(f"Validate {file.name} with manual input", key=f"val_{file.name}"):
                                if text:
                                    method = "manual"
                                else:
                                    st.error("Please provide some text to validate")
                                    continue
                        
                        # Store basic info
                        file_data = {
                            'filename': file.name,
                            'pages': pages,
                            'method': method,
                            'text_length': len(text),
                            'text': text[:1000],  # Store preview
                            'full_text': text
                        }
                        
                        # Run AI validation if we have text and API keys
                        if text and api_keys:
                            with st.spinner("Running AI validation..."):
                                ai_results = validate_with_ai(file.name, text, api_keys)
                                file_data['ai_results'] = ai_results
                                
                                # Show quick results
                                for provider, result in ai_results.items():
                                    if 'error' not in result:
                                        status = result.get('status', 'UNKNOWN')
                                        if status == 'PASS':
                                            st.success(f"✅ {provider}: PASS")
                                        elif status == 'FAIL':
                                            st.error(f"❌ {provider}: FAIL")
                                        else:
                                            st.warning(f"⚠️ {provider}: NEEDS REVIEW")
                        
                        st.session_state.results[file.name] = file_data
                
                st.success(f"✅ Processed {len(uploaded_files)} files!")
    
    with tab2:
        st.markdown("### 💬 AI Quality Assistant")
        
        if not api_keys:
            st.error("Please configure API keys to use the chat")
        elif not st.session_state.results:
            st.info("Please validate some files first")
        else:
            # Create context from results
            context = f"Validated {len(st.session_state.results)} files:\n"
            for filename, data in st.session_state.results.items():
                context += f"\n{filename}:"
                if 'ai_results' in data:
                    for provider, result in data['ai_results'].items():
                        if 'error' not in result:
                            context += f"\n  - {provider}: {result.get('status', 'Unknown')}"
                            if result.get('issues'):
                                context += f" (Issues: {', '.join(result['issues'][:2])})"
            
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-msg">👤 {msg["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message ai-msg">🤖 {msg["content"]}</div>', 
                              unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_input("Ask about your validation results...")
            
            if st.button("Send", type="primary"):
                if user_input:
                    # Add to history
                    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                    
                    # Get AI response
                    with st.spinner("Thinking..."):
                        response = ai_chat(user_input, context, api_keys)
                    
                    # Add response to history
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                    st.rerun()
    
    with tab3:
        st.markdown("### 📊 Validation Results")
        
        if st.session_state.results:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Files Processed", len(st.session_state.results))
            
            with col2:
                passed = sum(1 for r in st.session_state.results.values() 
                           if any(ai.get('status') == 'PASS' 
                                 for ai in r.get('ai_results', {}).values()
                                 if 'error' not in ai))
                st.metric("Passed", passed)
            
            with col3:
                failed = sum(1 for r in st.session_state.results.values() 
                           if any(ai.get('status') == 'FAIL' 
                                 for ai in r.get('ai_results', {}).values()
                                 if 'error' not in ai))
                st.metric("Failed", failed)
            
            # Detailed results
            for filename, data in st.session_state.results.items():
                with st.expander(f"📄 {filename}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Extraction Method:** {data.get('method', 'Unknown')}")
                        st.markdown(f"**Pages:** {data.get('pages', 0)}")
                        st.markdown(f"**Text Length:** {data.get('text_length', 0)} characters")
                    
                    with col2:
                        if 'ai_results' in data:
                            for provider, result in data['ai_results'].items():
                                if 'error' not in result:
                                    status = result.get('status', 'UNKNOWN')
                                    if status == 'PASS':
                                        st.success(f"✅ {provider}: PASS")
                                    elif status == 'FAIL':
                                        st.error(f"❌ {provider}: FAIL")
                                    else:
                                        st.warning(f"⚠️ {provider}: {status}")
                    
                    # Show validation details
                    if 'ai_results' in data:
                        for provider, result in data['ai_results'].items():
                            if 'error' not in result:
                                st.markdown(f"**{provider.title()} Analysis:**")
                                
                                # Product info
                                if result.get('product_name'):
                                    st.info(f"Product: {result['product_name']}")
                                if result.get('sku_found'):
                                    st.info(f"SKU: {result['sku_found']}")
                                
                                # Requirements
                                reqs = result.get('requirements', {})
                                if reqs:
                                    cols = st.columns(5)
                                    for idx, (req, met) in enumerate(reqs.items()):
                                        with cols[idx % 5]:
                                            if met:
                                                st.success(f"✅ {req.replace('_', ' ').title()}")
                                            else:
                                                st.error(f"❌ {req.replace('_', ' ').title()}")
                                
                                # Issues
                                if result.get('issues'):
                                    st.markdown("**Issues Found:**")
                                    for issue in result['issues']:
                                        st.markdown(f"• {issue}")
                                
                                # Suggestions
                                if result.get('suggestions'):
                                    st.markdown("**Suggestions:**")
                                    for suggestion in result['suggestions']:
                                        st.markdown(f"• {suggestion}")
                    
                    # Show text preview
                    if data.get('text'):
                        with st.expander("Text Preview"):
                            st.text(data['text'][:500] + "...")
            
            # Export button
            st.markdown("---")
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'results': st.session_state.results
            }
            
            st.download_button(
                "📥 Download Results (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("No results yet. Upload and validate files to see results.")

if __name__ == "__main__":
    main()
