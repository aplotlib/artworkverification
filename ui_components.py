import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

def display_header():
    st.markdown("<h1>🎨 Artwork Verification Co-pilot</h1>", unsafe_allow_html=True)

def display_instructions():
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        **1. Upload Files**: Drag and drop all related artwork files for a single product to run a verification report.
        **2. Chat with AI**: Use the chat interface at the bottom to ask general questions about medical device packaging at any time.
        **3. Configure (Optional)**: In the sidebar, add any specific text for the AI to check for during file verification.
        **4. Run Verification**: Click the button to start. Results will appear below as they are generated.
        """)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str]:
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        st.header("📝 AI Compliance Check")
        reference_text = st.text_area(
            "Enter text for the AI to verify in uploaded files, one phrase per line.",
            placeholder="Made in China\n1-year warranty"
        )
        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area("Ask the AI a specific question for the final summary.",
                                           placeholder="Does the manual mention a return policy?")
        if not api_keys.get('openai'):
            st.warning("An OpenAI API key is required for AI features.")
        if st.button("Clear & Reset"): st.session_state.clear(); st.rerun()
    return run_validation, custom_instructions, reference_text

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX)",
                             type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)

def display_results_page(global_results: List, per_doc_results: Dict, processed_docs: List, skus: List, ai_summary: str, ai_facts: Dict, compliance_results: List, **kwargs):
    st.header("📊 Verification Report")

    # Display a message if verification was run without files
    if not processed_docs:
        st.warning("No files were uploaded. The verification report is empty.")
        return

    with st.container(border=True):
        total_failures = len([r for r in global_results if r[0] == 'failed']) + \
                       sum(1 for res in compliance_results if res.get('status') == 'Fail')
        if total_failures > 0:
            st.error(f"**🚨 {total_failures} Potential Issue(s) Detected**")
        else:
            st.success("**✅ No Potential Issues Detected**")

    with st.container(border=True):
        if 'ai_processing_complete' not in st.session_state:
            st.info("🤖 AI analysis is running...")
        else:
            st.subheader("🤖 Executive Summary")
            st.markdown(ai_summary)

    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        tab_titles = ["📋 Reports & Checks"] + sorted(defaultdict(list, {d['doc_type'].replace('_', ' ').title(): [] for d in processed_docs}).keys())
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            if compliance_results:
                st.subheader("AI Compliance Check")
                for res in compliance_results:
                    icon = "✅" if res.get('status') == 'Pass' else "❌"
                    st.markdown(f"**{icon} {res.get('phrase')}**")
                    st.caption(f"Reasoning: {res.get('reasoning')}")
                st.divider()
            
            st.subheader("Rule-Based Checks")
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
    """Renders PDF pages as images for reliable in-browser previewing."""
    if not files: return
    st.header("📄 PDF Previews")
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    doc = fitz.open(stream=pdf_file['bytes'], filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        st.image(pix.tobytes("png"), caption=f"Page {page_num + 1}", use_container_width=True)
                    doc.close()
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")

def display_chat_interface():
    """Displays the standalone AI chat interface."""
    st.header("💬 Chat with AI Assistant")
    st.caption("Ask me anything about medical device packaging, labeling, or compliance.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is a UDI?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                api_keys = check_api_keys()
                reviewer = AIReviewer(api_keys)
                response = reviewer.run_chatbot_interaction(st.session_state.messages)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
