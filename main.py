"""
Universal Packaging Validator for Vive Health
Works with ANY product - no database required
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
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Vive Health Universal Packaging Validator",
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

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Universal validation requirements
UNIVERSAL_REQUIREMENTS = [
    "Made in China",
    "Vive or vive® branding",
    "Website URL (vivehealth.com)",
    "Contact information",
    "Product identifier (SKU/Model)",
    "Regulatory compliance marks"
]

def inject_css():
    """Clean, professional CSS"""
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .file-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-box {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .chat-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            height: 500px;
            overflow-y: auto;
        }
        
        .user-msg {
            background: #3498db;
            color: white;
            padding: 0.75rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            margin-left: 20%;
        }
        
        .ai-msg {
            background: #ecf0f1;
            padding: 0.75rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            margin-right: 20%;
        }
        
        .issue-card {
            background: #fee;
            border-left: 4px solid #c0392b;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        
        .warning-card {
            background: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        
        .success-card {
            background: #d4edda;
            border-left: 4px solid #27ae60;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

def get_api_keys():
    """Get API keys from secrets or environment"""
    keys = {}
    
    # Check streamlit secrets
    if hasattr(st, 'secrets'):
        if 'OPENAI_API_KEY' in st.secrets:
            keys['openai'] = st.secrets['OPENAI_API_KEY']
        elif 'openai_api_key' in st.secrets:
            keys['openai'] = st.secrets['openai_api_key']
            
        if 'ANTHROPIC_API_KEY' in st.secrets:
            keys['claude'] = st.secrets['ANTHROPIC_API_KEY']
        elif 'anthropic_api_key' in st.secrets:
            keys['claude'] = st.secrets['anthropic_api_key']
    
    # Check environment
    if 'openai' not in keys:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    if 'claude' not in keys:
        keys['claude'] = os.getenv('ANTHROPIC_API_KEY')
    
    return {k: v for k, v in keys.items() if v}

def extract_text_from_pdf(file_bytes, filename=""):
    """Extract text from PDF"""
    text = ""
    pages = 0
    
    try:
        # Try pdfplumber first
        if PDFPLUMBER_AVAILABLE:
            try:
                import pdfplumber
                file_bytes.seek(0)
                with pdfplumber.open(file_bytes) as pdf:
                    pages = len(pdf.pages)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except:
                pass
        
        # Try PyPDF2
        if not text:
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            pages = len(reader.pages)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
    
    return text.strip(), pages

def detect_file_info(filename, text):
    """Detect basic file information"""
    info = {
        'file_type': 'unknown',
        'has_text': bool(text),
        'detected_elements': []
    }
    
    filename_lower = filename.lower()
    
    # Detect file type
    if any(x in filename_lower for x in ['packaging', 'package', 'box']):
        info['file_type'] = 'packaging'
    elif any(x in filename_lower for x in ['wash', 'tag', 'care', 'label']):
        info['file_type'] = 'washtag'
    elif any(x in filename_lower for x in ['quick', 'start', 'guide', 'qsg']):
        info['file_type'] = 'quickstart'
    elif any(x in filename_lower for x in ['manual', 'instruction']):
        info['file_type'] = 'manual'
    elif any(x in filename_lower for x in ['shipping', 'ship']):
        info['file_type'] = 'shipping'
    
    # Detect elements in text
    if text:
        text_lower = text.lower()
        
        # Check for key elements
        if 'made in china' in text_lower:
            info['detected_elements'].append('Made in China')
        if 'vive' in text_lower:
            info['detected_elements'].append('Vive branding')
        if 'vivehealth.com' in text_lower:
            info['detected_elements'].append('Website')
        
        # Try to find SKU patterns
        sku_patterns = [
            r'LVA\d{4}[A-Z]{0,3}',
            r'SUP\d{4}[A-Z]{0,3}',
            r'MOB\d{4}[A-Z]{0,3}',
            r'[A-Z]{3}\d{4}[A-Z]{0,3}'
        ]
        
        for pattern in sku_patterns:
            matches = re.findall(pattern, text)
            if matches:
                info['detected_elements'].append(f'SKU: {matches[0]}')
                break
    
    return info

def create_validation_prompt(filename, text, file_info, other_files):
    """Create AI validation prompt"""
    prompt = f"""You are a quality control expert for Vive Health medical devices.

CURRENT FILE: {filename}
File Type: {file_info.get('file_type', 'unknown')}
Has Text: {file_info.get('has_text', False)}

OTHER FILES IN BATCH ({len(other_files)} total):
"""
    
    # Add context from other files
    for other in other_files[:5]:  # Limit to 5 files
        if other['filename'] != filename:
            prompt += f"- {other['filename']} ({other['file_type']})\n"
    
    prompt += f"""

EXTRACTED TEXT (first 2000 chars):
{text[:2000] if text else 'No text extracted'}

UNIVERSAL REQUIREMENTS TO CHECK:
1. Must have "Made in China" text
2. Must have Vive or vive® branding
3. Must have website (vivehealth.com)
4. Must have contact information
5. Must have product identifier (SKU/Model)
6. Must have appropriate regulatory marks

TASK:
1. Identify what product this is for (based on content)
2. Check all universal requirements
3. Note any inconsistencies with other files if applicable
4. Identify critical issues vs minor warnings

RESPONSE FORMAT (JSON):
{{
    "product_detected": "description of product",
    "overall_status": "PASS" or "FAIL" or "REVIEW",
    "requirements_met": {{
        "made_in_china": true/false,
        "vive_branding": true/false,
        "website": true/false,
        "contact_info": true/false,
        "product_id": true/false,
        "regulatory_marks": true/false
    }},
    "critical_issues": ["list of critical problems"],
    "warnings": ["list of minor issues"],
    "cross_file_observations": ["observations about consistency with other files"],
    "recommendations": ["specific fixes needed"]
}}"""
    
    return prompt

def validate_with_ai(filename, text, file_info, other_files, api_keys):
    """Run AI validation"""
    results = {}
    prompt = create_validation_prompt(filename, text, file_info, other_files)
    
    # Try Claude
    if 'claude' in api_keys and CLAUDE_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=api_keys['claude'])
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # FIXED MODEL NAME
                max_tokens=1500,
                temperature=0.1,
                system="You are a packaging validation expert. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            try:
                json_text = response.content[0].text
                json_match = re.search(r'\{[\s\S]*\}', json_text)
                if json_match:
                    results['claude'] = json.loads(json_match.group(0))
                else:
                    results['claude'] = {"error": "Failed to parse response"}
            except Exception as e:
                results['claude'] = {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Claude error: {e}")
            results['claude'] = {"error": str(e)}
    
    # Try OpenAI
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
                max_tokens=1500
            )
            
            # Parse response
            try:
                json_text = response.choices[0].message.content
                json_match = re.search(r'\{[\s\S]*\}', json_text)
                if json_match:
                    results['openai'] = json.loads(json_match.group(0))
                else:
                    results['openai'] = {"error": "Failed to parse response"}
            except Exception as e:
                results['openai'] = {"error": str(e)}
                
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            results['openai'] = {"error": str(e)}
    
    return results

class SimpleChatBot:
    """Simple chat interface for Q&A"""
    
    def __init__(self, api_keys):
        self.api_keys = api_keys
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def chat(self, user_message, validation_results):
        """Process chat message"""
        # Add to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Create context
        context = self._create_context(validation_results)
        
        # Build prompt
        prompt = f"""You are a helpful quality control assistant for Vive Health.

VALIDATION RESULTS SUMMARY:
{context}

USER QUESTION: {user_message}

Please provide a helpful, specific answer based on the validation results.
Be concise and focus on actionable insights."""
        
        # Get response
        response = None
        
        # Try Claude first (FIXED MODEL)
        if 'claude' in self.api_keys and CLAUDE_AVAILABLE:
            try:
                client = anthropic.Anthropic(api_key=self.api_keys['claude'])
                
                # Convert chat history to Claude format
                messages = []
                for msg in st.session_state.chat_history[-6:]:  # Last 6 messages
                    messages.append({
                        "role": "user" if msg['role'] == 'user' else "assistant",
                        "content": msg['content']
                    })
                
                # Add current question with context
                messages.append({"role": "user", "content": prompt})
                
                ai_response = client.messages.create(
                    model="claude-3-haiku-20240307",  # FIXED MODEL NAME
                    max_tokens=1000,
                    temperature=0.7,
                    messages=messages
                )
                
                response = ai_response.content[0].text
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                
                # Fallback to OpenAI
                if 'openai' in self.api_keys and OPENAI_AVAILABLE:
                    try:
                        client = openai.OpenAI(api_key=self.api_keys['openai'])
                        
                        messages = [{"role": "system", "content": "You are a helpful quality control assistant."}]
                        for msg in st.session_state.chat_history[-6:]:
                            messages.append({
                                "role": msg['role'],
                                "content": msg['content']
                            })
                        messages.append({"role": "user", "content": prompt})
                        
                        ai_response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        response = ai_response.choices[0].message.content
                        
                    except Exception as e2:
                        response = f"Error connecting to AI services: {str(e)}"
        
        # If no Claude, try OpenAI
        elif 'openai' in self.api_keys and OPENAI_AVAILABLE:
            try:
                client = openai.OpenAI(api_key=self.api_keys['openai'])
                
                messages = [{"role": "system", "content": "You are a helpful quality control assistant."}]
                for msg in st.session_state.chat_history[-6:]:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
                messages.append({"role": "user", "content": prompt})
                
                ai_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                response = ai_response.choices[0].message.content
                
            except Exception as e:
                response = f"Error: {str(e)}"
        else:
            response = "No AI service available. Please configure API keys."
        
        # Add response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def _create_context(self, results):
        """Create summary context from results"""
        total_files = len(results)
        passed = sum(1 for r in results.values() 
                    if any(ai.get('overall_status') == 'PASS' 
                          for ai in r.get('ai_results', {}).values()))
        failed = sum(1 for r in results.values() 
                    if any(ai.get('overall_status') == 'FAIL' 
                          for ai in r.get('ai_results', {}).values()))
        
        # Collect all issues
        all_issues = []
        products_found = set()
        
        for filename, data in results.items():
            for provider, ai_result in data.get('ai_results', {}).items():
                if 'error' not in ai_result:
                    if 'product_detected' in ai_result:
                        products_found.add(ai_result['product_detected'])
                    if 'critical_issues' in ai_result:
                        all_issues.extend(ai_result['critical_issues'])
        
        context = f"""
Total Files: {total_files}
Passed: {passed}
Failed: {failed}
Products Found: {', '.join(products_found) if products_found else 'Various'}
Common Issues: {', '.join(set(all_issues[:5])) if all_issues else 'None'}
"""
        return context

def analyze_cross_file_patterns(results):
    """Analyze patterns across files"""
    patterns = {
        'products': defaultdict(list),
        'file_types': defaultdict(list),
        'common_issues': defaultdict(int),
        'missing_requirements': defaultdict(int)
    }
    
    for filename, data in results.items():
        file_info = data.get('file_info', {})
        
        # Group by file type
        file_type = file_info.get('file_type', 'unknown')
        patterns['file_types'][file_type].append(filename)
        
        # Analyze AI results
        for provider, ai_result in data.get('ai_results', {}).items():
            if 'error' not in ai_result:
                # Track products
                product = ai_result.get('product_detected', 'Unknown')
                if product and product != 'Unknown':
                    patterns['products'][product].append(filename)
                
                # Track issues
                for issue in ai_result.get('critical_issues', []):
                    patterns['common_issues'][issue] += 1
                
                # Track missing requirements
                reqs = ai_result.get('requirements_met', {})
                for req, met in reqs.items():
                    if not met:
                        patterns['missing_requirements'][req] += 1
    
    return patterns

def main():
    inject_css()
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'chat_bot' not in st.session_state:
        st.session_state.chat_bot = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Vive Health Universal Packaging Validator</h1>
        <p>Validates packaging for ANY product - no database required</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Initialize chat bot
    if api_keys and not st.session_state.chat_bot:
        st.session_state.chat_bot = SimpleChatBot(api_keys)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        if 'claude' in api_keys:
            st.success("✅ Claude AI Ready")
        else:
            st.warning("❌ Claude Not Configured")
            
        if 'openai' in api_keys:
            st.success("✅ OpenAI Ready")
        else:
            st.warning("❌ OpenAI Not Configured")
        
        if not api_keys:
            st.error("⚠️ No AI providers configured")
            st.markdown("""
            Add to `.streamlit/secrets.toml`:
            ```
            OPENAI_API_KEY = "sk-..."
            ANTHROPIC_API_KEY = "sk-ant-..."
            ```
            """)
        
        st.markdown("---")
        st.markdown("### 📋 Universal Requirements")
        for req in UNIVERSAL_REQUIREMENTS:
            st.markdown(f"• {req}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Validate", "💬 AI Assistant", "📊 Analysis"])
    
    with tab1:
        st.markdown("### Upload Packaging Files")
        
        uploaded_files = st.file_uploader(
            "Select PDF files for any Vive Health products",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Validate All", type="primary"):
                progress = st.progress(0)
                st.session_state.results = {}
                
                # First pass - extract all text
                file_data = []
                for idx, file in enumerate(uploaded_files):
                    progress.progress((idx + 1) / len(uploaded_files) * 0.5)
                    
                    file.seek(0)
                    text, pages = extract_text_from_pdf(file, file.name)
                    file_info = detect_file_info(file.name, text)
                    
                    file_data.append({
                        'filename': file.name,
                        'text': text,
                        'pages': pages,
                        'file_info': file_info,
                        'file_type': file_info.get('file_type', 'unknown')
                    })
                
                # Second pass - validate with context
                for idx, data in enumerate(file_data):
                    progress.progress(0.5 + (idx + 1) / len(file_data) * 0.5)
                    
                    # Run AI validation if we have providers
                    if api_keys and data['text']:
                        ai_results = validate_with_ai(
                            data['filename'],
                            data['text'],
                            data['file_info'],
                            file_data,
                            api_keys
                        )
                        data['ai_results'] = ai_results
                    
                    st.session_state.results[data['filename']] = data
                
                progress.empty()
                st.success("✅ Validation complete!")
                
                # Show summary
                patterns = analyze_cross_file_patterns(st.session_state.results)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{len(st.session_state.results)}</div>
                        <div>Files Processed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    products_count = len(patterns['products'])
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{products_count}</div>
                        <div>Products Found</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    issues_count = sum(patterns['common_issues'].values())
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value">{issues_count}</div>
                        <div>Total Issues</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show results
                st.markdown("### 📋 Validation Results")
                
                for filename, data in st.session_state.results.items():
                    with st.expander(f"📄 {filename}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**File Type:** {data['file_info'].get('file_type', 'unknown')}")
                            st.markdown(f"**Pages:** {data['pages']}")
                            st.markdown(f"**Text Extracted:** {'Yes' if data['text'] else 'No'}")
                            
                            if data['file_info'].get('detected_elements'):
                                st.markdown("**Detected Elements:**")
                                for elem in data['file_info']['detected_elements']:
                                    st.markdown(f"• {elem}")
                        
                        with col2:
                            if 'ai_results' in data:
                                for provider, result in data['ai_results'].items():
                                    if 'error' not in result:
                                        status = result.get('overall_status', 'UNKNOWN')
                                        if status == 'PASS':
                                            st.markdown('<div class="success-card">✅ PASS</div>', 
                                                      unsafe_allow_html=True)
                                        elif status == 'FAIL':
                                            st.markdown('<div class="issue-card">❌ FAIL</div>', 
                                                      unsafe_allow_html=True)
                                        else:
                                            st.markdown('<div class="warning-card">⚠️ REVIEW</div>', 
                                                      unsafe_allow_html=True)
                                        break
                        
                        # Show AI results
                        if 'ai_results' in data:
                            for provider, result in data['ai_results'].items():
                                if 'error' not in result:
                                    st.markdown(f"**{provider.title()} Analysis:**")
                                    
                                    product = result.get('product_detected', 'Unknown')
                                    st.info(f"Product: {product}")
                                    
                                    # Requirements check
                                    reqs = result.get('requirements_met', {})
                                    if reqs:
                                        cols = st.columns(3)
                                        for idx, (req, met) in enumerate(reqs.items()):
                                            with cols[idx % 3]:
                                                if met:
                                                    st.success(f"✅ {req.replace('_', ' ').title()}")
                                                else:
                                                    st.error(f"❌ {req.replace('_', ' ').title()}")
                                    
                                    # Issues
                                    if result.get('critical_issues'):
                                        st.markdown("**Critical Issues:**")
                                        for issue in result['critical_issues']:
                                            st.markdown(f"• {issue}")
                                    
                                    if result.get('recommendations'):
                                        st.markdown("**Recommendations:**")
                                        for rec in result['recommendations']:
                                            st.markdown(f"• {rec}")
    
    with tab2:
        st.markdown("### 💬 AI Quality Assistant")
        
        if not api_keys:
            st.warning("Please configure AI providers to use the chat assistant")
        elif not st.session_state.results:
            st.info("Please validate files first to use the assistant")
        else:
            # Display chat history
            chat_container = st.container()
            
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f'<div class="user-msg">👤 {msg["content"]}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ai-msg">🤖 {msg["content"]}</div>', 
                                  unsafe_allow_html=True)
            
            # Chat input
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_input("Ask about your validation results...", 
                                         placeholder="e.g., What are the main issues found?")
            
            with col2:
                if st.button("Send", type="primary"):
                    if user_input and st.session_state.chat_bot:
                        with st.spinner("Thinking..."):
                            st.session_state.chat_bot.chat(user_input, st.session_state.results)
                        st.rerun()
            
            # Quick questions
            st.markdown("#### 💡 Quick Questions")
            
            questions = [
                "What are the most common issues across all files?",
                "Which files have critical problems?",
                "Are all products properly identified?",
                "What's missing from the packaging files?"
            ]
            
            cols = st.columns(2)
            for idx, question in enumerate(questions):
                with cols[idx % 2]:
                    if st.button(question, key=f"q_{idx}"):
                        if st.session_state.chat_bot:
                            with st.spinner("Thinking..."):
                                st.session_state.chat_bot.chat(question, st.session_state.results)
                            st.rerun()
    
    with tab3:
        st.markdown("### 📊 Cross-File Analysis")
        
        if st.session_state.results:
            patterns = analyze_cross_file_patterns(st.session_state.results)
            
            # File type breakdown
            st.markdown("#### 📁 Files by Type")
            
            file_type_cols = st.columns(len(patterns['file_types']))
            for idx, (file_type, files) in enumerate(patterns['file_types'].items()):
                with file_type_cols[idx]:
                    st.markdown(f"""
                    <div class="file-card">
                        <strong>{file_type.title()}</strong><br>
                        {len(files)} files
                    </div>
                    """, unsafe_allow_html=True)
            
            # Products found
            if patterns['products']:
                st.markdown("#### 🏷️ Products Identified")
                
                for product, files in patterns['products'].items():
                    st.markdown(f"""
                    <div class="file-card">
                        <strong>{product}</strong><br>
                        Found in {len(files)} file(s): {', '.join(files[:3])}{'...' if len(files) > 3 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Common issues
            if patterns['common_issues']:
                st.markdown("#### ⚠️ Most Common Issues")
                
                sorted_issues = sorted(patterns['common_issues'].items(), 
                                     key=lambda x: x[1], reverse=True)
                
                for issue, count in sorted_issues[:5]:
                    st.markdown(f"""
                    <div class="warning-card">
                        {issue} - Found in {count} validation(s)
                    </div>
                    """, unsafe_allow_html=True)
            
            # Missing requirements
            if patterns['missing_requirements']:
                st.markdown("#### ❌ Missing Requirements")
                
                cols = st.columns(3)
                for idx, (req, count) in enumerate(patterns['missing_requirements'].items()):
                    with cols[idx % 3]:
                        st.error(f"{req.replace('_', ' ').title()}: {count} files")
            
            # Export
            st.markdown("### 💾 Export Results")
            
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'total_files': len(st.session_state.results),
                'patterns': {
                    'products': dict(patterns['products']),
                    'common_issues': dict(patterns['common_issues']),
                    'missing_requirements': dict(patterns['missing_requirements'])
                },
                'detailed_results': st.session_state.results
            }
            
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"packaging_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("Validate files to see analysis")

if __name__ == "__main__":
    main()
