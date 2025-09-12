import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from ai_analyzer import AIReviewer, check_api_keys

# --- Security: Rate Limiting Constants ---
CHAT_COOLDOWN_SECONDS = 3

def display_header():
    st.markdown("""
        <style>
            .header-title {
                font-size: 3rem;
                font-weight: 700;
                color: #2A9D8F;
                padding: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">🎨 Artwork Verification Co-pilot</h1>', unsafe_allow_html=True)

def display_instructions():
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        **1. Upload Files or Folders**: Drag and drop all related artwork files or a folder for a single product to run a verification report.
        **2. Review Report**: Key metrics appear at the top. Use the tabs below for detailed findings.
        **3. Chat with AI**: Use the chat interface at the bottom to ask about the report or general compliance questions.
        """)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str, bool]:
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        run_test_validation = st.button("🧪 Run Test Validation")
        st.header("📝 AI Compliance Check")
        reference_text = st.text_area(
            "Enter text for the AI to verify in uploaded files, one phrase per line.",
            placeholder="Made in China\n1-year warranty",
            max_chars=1000
        )
        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area("Ask the AI a specific question for the final summary.",
                                           placeholder="Does the manual mention a return policy?",
                                           max_chars=1000
                                          )
        if not api_keys.get('openai'):
            st.warning("An OpenAI API key is required for AI features.")
        if st.button("Clear & Reset"): st.session_state.clear(); st.rerun()
    return run_validation, custom_instructions, reference_text, run_test_validation

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX) or a folder",
                             type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)

def display_results_page(global_results: List, per_doc_results: Dict, processed_docs: List, skus: List, ai_summary: str, ai_facts: Dict, compliance_results: List, quality_results: Dict, **kwargs):
    st.header("📊 Verification Report")

    if not processed_docs:
        st.warning("No files were uploaded. The verification report is empty.")
        return

    # --- UI/UX Improvement: Summary Dashboard ---
    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        total_failures = len([r for r in global_results if r[0] == 'failed']) + \
                       sum(1 for res in compliance_results if res.get('status') == 'Fail')
        quality_issues = len(quality_results.get('issues', []))
        total_issues = total_failures + quality_issues
        
        sparkline_data = np.random.randn(20)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
            st.line_chart(sparkline_data, height=100)
        col2.metric("Rule-Based Failures", total_failures)
        col3.metric("Quality/Typo Issues", quality_issues)
        col4.metric("SKUs Found", len(skus), ", ".join(skus))
        
        if total_issues > 0:
            st.error(f"**🚨 {total_issues} Potential Issue(s) Detected**")
        else:
            st.success("**✅ No Potential Issues Detected**")

    # --- Executive Summary ---
    with st.expander("🤖 Read Executive Summary", expanded=True):
        if 'ai_processing_complete' not in st.session_state:
            st.info("🤖 AI analysis is running...")
        else:
            st.markdown(ai_summary)

    # --- Detailed Tabs ---
    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        tab_titles = ["📋 Reports & Checks"] + sorted(defaultdict(list, {d['doc_type'].replace('_', ' ').title(): [] for d in processed_docs}).keys())
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            if quality_results and 'issues' in quality_results:
                with st.expander("AI-Powered Proofreading", expanded=True):
                    if not quality_results['issues']:
                        st.markdown("✅ No spelling or grammar issues found.")
                    else:
                        for issue in quality_results['issues']:
                            st.markdown(f"**- Error:** `{issue['error']}` -> **Correction:** `{issue['correction']}`")
                            st.caption(f"Context: \"...{issue['context']}...\"")
                st.divider()

            if compliance_results:
                with st.expander("AI Compliance Check", expanded=True):
                    for res in compliance_results:
                        icon = "✅" if res.get('status') == 'Pass' else "❌"
                        st.markdown(f"**{icon} {res.get('phrase')}**")
                        st.caption(f"Reasoning: {res.get('reasoning')}")
                st.divider()
            
            with st.expander("Rule-Based Checks", expanded=True):
                for status, msg, _ in global_results:
                    st.markdown(f"{'✅' if status == 'passed' else '❌'} {msg}")

        docs_by_type = defaultdict(list)
        for doc in processed_docs: docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
        for i, title in enumerate(sorted(docs_by_type.keys())):
            with tabs[i + 1]:
                for doc in docs_by_type[title]:
                    nature_tag = f":blue[Shared File]" if doc['file_nature'] == 'Shared File' else f":orange[Unique Artwork]"
                    with st.expander(f"**{doc['filename']}** ({nature_tag})"):
                        if doc['doc_type'] == 'packaging_artwork' and ai_facts:
                            st.markdown("**AI Fact Extraction:**"); st.json(ai_facts)
                        
                        text_preview = doc['text']
                        if len(text_preview) > 2000: text_preview = text_preview[:2000] + "\n\n... (text truncated)"
                        st.text_area("Extracted Text Preview", text_preview, height=250, key=f"text_{doc['filename']}", label_visibility="collapsed")

def display_pdf_previews(files: List[Dict[str, Any]]):
    """Renders PDFs using the new st.pdf function."""
    if not files: return
    st.header("📄 PDF Previews")
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    st.pdf(pdf_file['bytes'])
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")

def display_chat_interface():
    """Displays the standalone AI chat interface with rate limiting."""
    st.header("💬 Chat with AI Assistant")
    st.caption("Ask me about the report or anything related to medical device packaging.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_chat_time" not in st.session_state:
        st.session_state.last_chat_time = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the verification report..."):
        current_time = time.time()
        # --- Security: Chat Rate Limiting ---
        if current_time - st.session_state.last_chat_time < CHAT_COOLDOWN_SECONDS:
            st.toast(f"Please wait {CHAT_COOLDOWN_SECONDS} seconds between messages.", icon="⏳", duration=3000)
        else:
            st.session_state.last_chat_time = current_time
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    api_keys = check_api_keys()
                    reviewer = AIReviewer(api_keys)
                    
                    analysis_context = {}
                    if st.session_state.get('validation_complete'):
                        analysis_context = {
                            "Executive Summary": st.session_state.ai_summary,
                            "Rule-Based Results": st.session_state.global_results,
                            "AI Compliance Results": st.session_state.compliance_results,
                            "AI Quality Results": st.session_state.quality_results
                        }

                    response = reviewer.run_chatbot_interaction(st.session_state.messages, analysis_context)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
