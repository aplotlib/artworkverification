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
        **1. Upload Files**: Drag and drop all related artwork files for a single product.
        **2. Configure (Optional)**: In the sidebar, add any specific text for the AI to check for.
        **3. Run Verification**: Click the button to start. Results will appear below as they are generated.
        """)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str]:
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        st.header("📝 AI Compliance Check")
        reference_text = st.text_area(
            "Enter text for the AI to verify, one phrase per line.",
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

def display_results_page(global_results: List, per_doc_results: Dict, docs: List, skus: List, ai_summary: str, ai_facts: Dict, compliance_results: List):
    st.header("📊 Verification Report")

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
        tab_titles = ["📋 Reports & Checks"] + sorted(defaultdict(list, {d['doc_type'].replace('_', ' ').title(): [] for d in docs}).keys())
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
        for doc in docs: docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
        for i, title in enumerate(sorted(docs_by_type.keys())):
            with tabs[i + 1]:
                for doc in docs_by_type[title]:
                    nature_tag = f":blue[Shared File]" if doc['file_nature'] == 'Shared File' else f":orange[Unique Artwork]"
                    with st.expander(f"**{doc['filename']}** ({nature_tag})"):
                        if doc['doc_type'] == 'packaging_artwork' and ai_facts:
                            st.markdown("**AI Fact Extraction:**"); st.json(ai_facts)
