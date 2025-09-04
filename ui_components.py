import streamlit as st
import base64
import fitz # PyMuPDF
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def display_header():
    """Displays the main application header."""
    st.markdown("<h1>Artwork Verification Dashboard</h1>", unsafe_allow_html=True)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str]:
    """Displays the sidebar controls and returns their current values."""
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        
        st.subheader("Required Text (Optional)")
        reference_text = st.text_area(
            "Enter text that MUST appear in the files, one phrase per line.",
            help="E.g., 'Made in China' or '90% cotton'. The tool will check each file for each phrase.",
            placeholder="Made in China\n90% cotton, 10% polyester\nLVA3100PUR"
        )
        
        st.header("🤖 AI Review")
        custom_instructions = st.text_area("Custom Instructions for AI (Optional)", 
                                           help="Guide the AI's focus, e.g., 'Check for a 1-year warranty statement.'")
        
        if not api_keys.get('openai') or not api_keys.get('anthropic'):
            st.warning("Both OpenAI and Anthropic API keys are required for the AI review feature.")

        if st.button("Clear & Reset"):
            st.session_state.clear()
            st.rerun()
            
    return run_validation, custom_instructions, reference_text

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Displays the file uploader and returns the list of uploaded files."""
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX)", 
                             type=['pdf', 'csv', 'xlsx'], 
                             accept_multiple_files=True)

def display_dashboard(global_results: List, per_doc_results: Dict, docs: List, skus: List):
    """Displays the main validation report as a dashboard."""
    st.header("📊 Verification Results")

    # --- Overall Status ---
    global_failures = [r for r in global_results if r[0] == 'failed']
    doc_failures = sum(1 for-res-list in per_doc_results.values() for res in res_list if res[0] == 'failed')
    total_failures = len(global_failures) + doc_failures
        
    if total_failures > 0:
        st.error(f"**Overall Status: 🚨 {total_failures} Issue(s) Found**")
    else:
        st.success("**Overall Status: ✅ All Automated Checks Passed**")

    # --- Key Information Expander ---
    with st.expander("Global Checks & Key Information", expanded=True):
        if len(skus) > 1:
            st.warning(f"**Multiple Variants Detected**: SKUs `{', '.join(skus)}` were found.")
        elif skus:
            st.info(f"**Product SKU Detected**: `{skus[0]}`")

        st.markdown("---")
        st.subheader("UDI & Serial Number Analysis (Across All Files)")
        if not any('UPC' in r[1] for r in global_results):
             st.markdown("- No UPCs or UDIs found to analyze.")
        for status, msg, _ in global_results:
            icon = '✅' if status == 'passed' else '❌'
            st.markdown(f"{icon} {msg}")

    # --- Document-Specific Details ---
    st.header("📂 Document Inspector")
    docs_by_type = defaultdict(list)
    for doc in docs:
        docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
    
    tab_titles = sorted(docs_by_type.keys())
    if not tab_titles: return
    
    tabs = st.tabs(tab_titles)
    
    for i, title in enumerate(tab_titles):
        with tabs[i]:
            for doc in docs_by_type[title]:
                doc_results = per_doc_results.get(doc['filename'], [])
                has_failure = any(r[0] == 'failed' for r in doc_results)
                icon = '❌' if has_failure else '✅'
                
                with st.expander(f"{icon} **{doc['filename']}**", expanded=True):
                    if doc_results:
                        st.markdown("**File-Specific Checks:**")
                        for status, msg in doc_results:
                            res_icon = '✅' if status == 'passed' else '❌'
                            st.markdown(f"- {res_icon} {msg}")
                        st.markdown("---")
                    
                    st.text_area(
                        "Extracted Text", doc['text'], height=200, key=f"text_{doc['filename']}",
                        label_visibility="collapsed"
                    )

def display_ai_review(summary: str):
    """Displays the AI-generated review."""
    if summary:
        st.markdown("---")
        st.header("🤖 AI-Powered Final Summary")
        st.markdown(summary, unsafe_allow_html=True)

def display_pdf_previews(files: List[Dict[str, Any]]):
    """Renders PDF pages as images for reliable in-browser previewing."""
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        st.header("📄 PDF Previews")
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    doc = fitz.open(stream=pdf_file['bytes'], filetype="pdf")
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap()
                        img_bytes = pix.tobytes("png")
                        st.image(
                            img_bytes, caption=f"Page {page_num + 1}",
                            use_container_width=True
                        )
                    doc.close()
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")
