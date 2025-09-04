import streamlit as st
import base64
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def display_header():
    """Displays the main application header."""
    st.markdown("<h1>Artwork Verification Dashboard</h1>", unsafe_allow_html=True)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str]:
    """Displays the sidebar controls and returns their current values."""
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        
        st.header("🤖 AI Review")
        custom_instructions = st.text_area("Custom Instructions for AI (Optional)", 
                                           help="Guide the AI's focus, e.g., 'Check for a 1-year warranty statement.'")
        
        if not api_keys.get('openai') or not api_keys.get('anthropic'):
            st.warning("Both OpenAI and Anthropic API keys are required for the AI review feature.")

        if st.button("Clear & Reset"):
            st.session_state.clear()
            st.rerun()
            
    return run_validation, custom_instructions

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Displays the file uploader and returns the list of uploaded files."""
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX)", 
                             type=['pdf', 'csv', 'xlsx'], 
                             accept_multiple_files=True)

def display_dashboard(results: List[Tuple[str, str, str]], docs: List[Dict[str, Any]], skus: List[str]):
    """Displays the main validation report as a dashboard."""
    st.header("📊 Verification Results")

    # --- Overall Status ---
    failures = [r for r in results if r[0] == 'failed']
    if not results:
        st.info("Run verification to see results.")
        return
        
    if failures:
        st.error(f"**Overall Status: 🚨 {len(failures)} Issue(s) Found**")
    else:
        st.success("**Overall Status: ✅ All Automated Checks Passed**")

    # --- Key Information Expander ---
    with st.expander("Key Information & Rule-Based Checks", expanded=True):
        if len(skus) > 1:
            st.warning(f"**Multiple Variants Detected**: SKUs `{', '.join(skus)}` were found. Please manually ensure differences are correctly reflected.")
        elif skus:
            st.info(f"**Product SKU Detected**: `{skus[0]}`")

        st.markdown("---")
        st.subheader("UDI & Serial Number Analysis")
        for status, msg, _ in results:
            icon = '✅' if status == 'passed' else '❌'
            st.markdown(f"{icon} {msg}")

    # --- Document-Specific Details ---
    st.header("📂 Document Inspector")
    
    docs_by_type = defaultdict(list)
    for doc in docs:
        docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
    
    if not docs_by_type:
        return

    tab_titles = sorted(docs_by_type.keys())
    tabs = st.tabs(tab_titles)
    
    for i, title in enumerate(tab_titles):
        with tabs[i]:
            for doc in docs_by_type[title]:
                with st.expander(f"📄 **{doc['filename']}**"):
                    st.text_area(
                        "Extracted Text", 
                        doc['text'], 
                        height=250, 
                        key=f"text_{doc['filename']}",
                        label_visibility="collapsed"
                    )


def display_ai_review(summary: str):
    """Displays the AI-generated review."""
    if summary:
        st.markdown("---")
        st.header("🤖 AI-Powered Final Summary")
        st.markdown(summary, unsafe_allow_html=True)

def display_pdf_previews(files: List[Dict[str, Any]]):
    """Displays uploaded PDFs using an iframe."""
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        with st.expander("📄 PDF Previews"):
            for pdf_file in pdf_files:
                base64_pdf = base64.b64encode(pdf_file['bytes']).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
                st.markdown(f"**{pdf_file['name']}**")
                st.markdown(pdf_display, unsafe_allow_html=True)
