import streamlit as st
import pandas as pd
import fitz # PyMuPDF
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

def display_header():
    st.markdown("<h1>🎨 Artwork Verification Dashboard</h1>", unsafe_allow_html=True)

def display_instructions():
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        **1. Upload Files**: Drag and drop all related artwork files for a single product.
        **2. Configure (Optional)**: In the sidebar, add any specific text that must be present in the files.
        **3. Run Verification**: Click the button to start. Rule-based checks will appear instantly, followed by the AI summary.
        """)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str]:
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Verification", type="primary")
        st.header("📝 Required Text (Optional)")
        reference_text = st.text_area("Enter text that MUST appear in the files, one phrase per line.",
                                      placeholder="Made in China\n90% cotton, 10% polyester")
        st.header("🤖 AI Review")
        custom_instructions = st.text_area("Custom Instructions for AI (Optional)",
                                           help="Guide the AI's focus, e.g., 'Check for a 1-year warranty statement.'")
        if not api_keys.get('openai') or not api_keys.get('anthropic'):
            st.warning("Both OpenAI and Anthropic API keys are required for full AI features.")
        if st.button("Clear & Reset"): st.session_state.clear(); st.rerun()
    return run_validation, custom_instructions, reference_text

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX)",
                             type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)

def create_summary_tables(docs: List, skus: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # ... (implementation remains the same as previous version)
    serial_data = []
    all_text = " ".join(d['text'] for d in docs)
    upcs = set(re.findall(r'\b(\d{12})\b', all_text.replace(" ", "")))
    udis = set(re.findall(r'\(01\)(\d{14})', all_text.replace(" ", "")))
    for sku in skus:
        related_upc = next((upc for upc in upcs if sku[:7] in upc), "N/A")
        related_udi = next((udi for udi in udis if related_upc != "N/A" and related_upc in udi), "N/A")
        serial_data.append({"SKU": sku, "UPC": related_upc, "UDI": related_udi})
    serials_df = pd.DataFrame(serial_data)
    dims_data = [{"File Name": doc['filename'], "Detected Dimensions": ", ".join(doc['dimensions'])}
                 for doc in docs if doc['dimensions']]
    dims_df = pd.DataFrame(dims_data)
    return serials_df, dims_df


def display_results_page(global_results: List, per_doc_results: Dict, docs: List, skus: List, ai_summary: str, ai_facts: Dict):
    """Displays the main results page with a streamlined, enterprise-grade UI."""
    st.header("📊 Verification Summary")
    summary_container = st.container(border=True)
    with summary_container:
        total_failures = len([r for r in global_results if r[0] == 'failed']) + \
                       sum(1 for res_list in per_doc_results.values() for res in res_list if res[0] == 'failed')
        if total_failures > 0:
            st.error(f"**🚨 {total_failures} Issue(s) Found**")
        else:
            st.success("**✅ All Automated Checks Passed**")
        
        if 'ai_processing_complete' not in st.session_state:
            st.info("🤖 AI analysis is running... the summary will appear here shortly.")
        else:
            st.subheader("🤖 AI-Powered Analysis")
            st.markdown(ai_summary, unsafe_allow_html=True)

    with st.expander("View Detailed Reports & Rule-Based Checks"):
        serials_df, dims_df = create_summary_tables(docs, skus)
        if not serials_df.empty: st.subheader("Serial Number Report"); st.dataframe(serials_df, use_container_width=True)
        if not dims_df.empty: st.subheader("Artwork Dimensions Report"); st.dataframe(dims_df, use_container_width=True)
        st.subheader("Rule-Based Analysis")
        for status, msg, _ in global_results:
            st.markdown(f"{'✅' if status == 'passed' else '❌'} {msg}")

    st.header("🔍 Document Inspector")
    docs_by_type = defaultdict(list)
    for doc in docs: docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
    
    tabs = st.tabs(sorted(docs_by_type.keys()))
    for i, title in enumerate(sorted(docs_by_type.keys())):
        with tabs[i]:
            for doc in docs_by_type[title]:
                icon = '❌' if any(r[0] == 'failed' for r in per_doc_results.get(doc['filename'], [])) else '✅'
                with st.expander(f"{icon} **{doc['filename']}** ({doc['file_nature']})", expanded=False):
                    if doc['doc_type'] == 'packaging_artwork' and ai_facts:
                        st.markdown("**AI Fact Check Results:**")
                        st.json(ai_facts)
                        st.markdown("---")
                    
                    if per_doc_results.get(doc['filename']):
                        st.markdown("**Rule-Based Checks:**")
                        for status, msg in per_doc_results[doc['filename']]:
                            st.markdown(f"- {'✅' if status == 'passed' else '❌'} {msg}")
                    
                    text_preview = doc['text']
                    if len(text_preview) > 2000: text_preview = text_preview[:2000] + "\n\n... (text truncated)"
                    st.text_area("Extracted Text Preview", text_preview, height=250, key=f"text_{doc['filename']}", label_visibility="collapsed")

def display_pdf_previews(files: List[Dict[str, Any]]):
    st.header("📄 PDF Previews")
    for pdf_file in [f for f in files if f['name'].lower().endswith('.pdf')]:
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
