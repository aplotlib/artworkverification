import streamlit as st
import pandas as pd
import time
import csv
from io import StringIO
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from ai_analyzer import AIReviewer, check_api_keys
from config import AppConfig
from datetime import datetime

def on_batch_selection_change():
    """Callback to update the current batch SKU when a user selects from the dropdown."""
    st.session_state.current_batch_sku = st.session_state.get('selected_batch_for_review')
    st.session_state.messages = [] # Reset chat messages on batch switch

def display_header():
    """Displays the branded application header."""
    st.markdown("""
        <style>
            .header-title { font-size: 2.5rem; font-weight: 700; color: #2A9D8F; padding-bottom: 0rem; }
            .st-emotion-cache-183lzff {
                color: #FFFFFF;
                background-color: #D33682;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">Vive Health Artwork Verification Co-pilot</h1>', unsafe_allow_html=True)

def display_instructions():
    """Displays the simplified 'How to Use' guide."""
    with st.expander("📖 How to Use the Co-pilot"):
        st.markdown("""
        **1. 📋 Select Brand & AI Provider**
        - Use the buttons in the sidebar to choose **Vive** or **Coretech**. This loads the correct checklist.
        - Select your preferred AI provider (OpenAI or Anthropic).

        **2. ✍️ Provide Primary Validation Source (Recommended)**
        - You can either upload a file named **Primary_Validation_SKU.pdf** OR provide the validation text in the sidebar. The file will take precedence.
        - Providing a primary validation source is highly recommended to ensure the highest accuracy.

        **3. 📁 Upload Files & Run Verification**
        - Drag and drop all artwork files for one product into the uploader.
        - Click **"🔍 Run Verification"** to start the analysis. The app will process the files and display a detailed report.
        """)

def display_sidebar(api_keys: Dict[str, str]):
    """Renders the sidebar with brand buttons, keyword inputs, and batch management."""
    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🚀 Start New Batch", type="primary", help="Clear all data and start a fresh verification session."):
            st.session_state.clear()
            st.rerun()

        if 'batches' in st.session_state and st.session_state.batches:
            batch_options = list(st.session_state.batches.keys())
            
            # Find the index of the current batch to set as the default for the selectbox
            try:
                current_index = batch_options.index(st.session_state.get('current_batch_sku'))
            except (ValueError, TypeError):
                current_index = 0 # Default to the first item if not found

            st.selectbox(
                "𗂬️ Review Previous Batch:",
                options=batch_options,
                key='selected_batch_for_review', # Use the new, dedicated key
                index=current_index,
                on_change=on_batch_selection_change # Use the callback to sync state
            )
        
        st.divider()
        st.header("📋 Verification Setup")
        
        st.file_uploader("Upload Primary Validation File (Optional)", key='primary_validation_file', help="Upload the primary validation file. This will be used as the source of truth for the verification.")
        
        st.text_area("Primary Validation Text (Optional)", key='primary_validation_text', height=150, help="Enter the source text for validation (UPC, SKU, etc.). This will be ignored if a primary validation file is uploaded.")
        
        col1, col2 = st.columns(2)
        if col1.button("Vive Product", use_container_width=True):
            st.session_state.brand_selection = "Vive"
        if col2.button("Coretech Product", use_container_width=True):
            st.session_state.brand_selection = "Coretech"
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        st.info(f"Selected Brand: **{brand_selection}**")

        st.selectbox("Select AI Provider:", options=["openai", "anthropic"], help="Choose which AI service to use for analysis.", key='ai_provider')
        
    # The button that triggers the run is now the only thing that sets 'run_validation'
    if st.sidebar.button("🔍 Run Verification", use_container_width=True):
        st.session_state.run_validation = True
        st.rerun()


def display_main_interface():
    """Displays the main UI with tabs for interaction and results."""
    
    st.text_area("Optional: Provide special instructions for the AI Co-pilot:", key='custom_instructions', height=100)
    display_chat_interface(st.session_state.get('batches', {}).get(st.session_state.get('current_batch_sku')))

    tab1, tab2 = st.tabs(["📋 Manual Checklist & File Upload", "📊 AI Analysis Report"])

    with tab1:
        st.header("✅ Manual Verification")
        st.session_state['uploaded_files'] = display_file_uploader()
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        batch_key = st.session_state.get('current_batch_sku', 'standalone_checklist')
        display_dynamic_checklist(brand_selection, batch_key)

    with tab2:
        st.header("🤖 AI Co-pilot Report")
        current_batch_data = st.session_state.get('batches', {}).get(st.session_state.get('current_batch_sku'))
        if current_batch_data:
            display_results_page(current_batch_data)
        else:
            st.info("Run a verification to see the AI report here.")


def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Renders the file uploader with guidance on file size."""
    return st.file_uploader(
        "📁 Upload Artwork Files (or Drag & Drop a Folder)",
        type=['pdf', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help=f"Max individual file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
    )
    
def display_dynamic_checklist(brand: str, batch_key: str):
    """Renders an interactive checklist based on the selected brand."""
    with st.container(border=True):
        checklist_data = AppConfig.CHECKLISTS.get(brand, {})
        if not checklist_data:
            st.warning("No checklist found for the selected brand.")
            return

        if 'checklist_state' not in st.session_state: st.session_state.checklist_state = {}
        if batch_key not in st.session_state.checklist_state: st.session_state.checklist_state[batch_key] = {}
        current_state = st.session_state.checklist_state[batch_key]

        total_items = sum(len(items) for items in checklist_data.values())
        
        num_checked = sum(1 for item_key in current_state if current_state.get(item_key, False))
        
        for category, items in checklist_data.items():
            st.subheader(category)
            for item in items:
                unique_key = f"check_{batch_key}_{category}_{item}"
                item_state_key = f"{brand}_{category}_{item}"
                
                is_checked = st.checkbox(item, value=current_state.get(item_state_key, False), key=unique_key)
                current_state[item_state_key] = is_checked
        
        st.divider()
        if total_items > 0:
            num_checked = sum(1 for item_key in current_state if current_state.get(item_key, False))
            progress = num_checked / total_items if total_items > 0 else 0
            st.progress(progress, text=f"{num_checked} / {total_items} items completed")
            if progress == 1.0:
                st.success("🎉 Checklist complete! Great work.")
                st.balloons()

def generate_report_text(batch_data: Dict) -> str:
    """Generates a plain text report from the analysis results."""
    skus = ", ".join(batch_data.get('skus', ['N/A']))
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"Artwork Verification Report\n"
    report += f"SKU(s): {skus}\n"
    report += f"Date: {report_date}\n"
    report += "="*40 + "\n\n"
    
    report += "Executive Summary\n" + "-"*20 + "\n"
    report += batch_data.get('ai_summary', "No summary available.") + "\n\n"
    
    report += "Rule-Based Checks\n" + "-"*20 + "\n"
    for result in batch_data.get('global_results', []):
        report += f"[{result.get('status', 'N/A').upper()}] {result.get('message', '')}\n"
    report += "\n"

    report += "AI Compliance Check\n" + "-"*20 + "\n"
    compliance_results = batch_data.get('compliance_results', {}).get('data', [])
    if compliance_results:
        for res in compliance_results:
            report += f"[{res.get('status', 'N/A').upper()}] {res.get('phrase', '')}: {res.get('reasoning', '')}\n"
    else:
        report += "No compliance checks were run.\n"
    report += "\n"

    report += "AI-Powered Proofreading\n" + "-"*20 + "\n"
    quality_issues = batch_data.get('quality_results', {}).get('data', {}).get('issues', [])
    if quality_issues:
        for issue in quality_issues:
            report += f"- Error: '{issue['error']}' -> Suggested: '{issue['correction']}'\n"
    else:
        report += "No spelling or grammar issues found.\n"
        
    return report

def display_results_page(batch_data: Dict):
    """Displays the main results page with all tabs and details."""
    skus = batch_data.get('skus', [])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")

    report_text = generate_report_text(batch_data)
    file_name_sku = "_".join(skus) if skus else "report"
    if len(file_name_sku) > 20: file_name_sku = "multiple"
    
    download_filename = f"{datetime.now().strftime('%Y-%m-%d')}_{file_name_sku}.txt"
    st.download_button(
        label="📥 Download Report",
        data=report_text,
        file_name=download_filename,
        mime="text/plain"
    )

    global_results = batch_data.get('global_results', [])
    compliance_results_data = batch_data.get('compliance_results', {})
    compliance_results = compliance_results_data.get('data', []) if compliance_results_data.get('success') else []
    quality_issues_data = batch_data.get('quality_results', {})
    quality_issues = len(quality_issues_data.get('data', {}).get('issues', [])) if quality_issues_data.get('success') else 0
    ai_summary = batch_data.get('ai_summary', "AI summary could not be generated.")

    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in global_results if r.get('status') == 'failed'])
        compliance_failures = sum(1 for res in compliance_results if res.get('status') == 'Fail')
        total_issues = rule_failures + compliance_failures + quality_issues

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
        col2.metric("Rule-Based Failures", rule_failures)
        col3.metric("AI Compliance Failures", compliance_failures)
        col4.metric("Quality/Typo Issues", quality_issues)

    with st.expander("🤖 AI Executive Summary", expanded=True):
        st.markdown(ai_summary)

    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        processed_docs = batch_data.get('processed_docs', [])
        tab_titles = ["📋 Automated Checks", "🎨 Brand Compliance"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            display_automated_checks(global_results)

        with tabs[1]:
            display_brand_compliance(processed_docs)

        for i, doc_type in enumerate(sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))):
            with tabs[i+2]:
                docs_to_display = [d for d in processed_docs if d['doc_type'].replace('_', ' ').title() == doc_type]
                display_document_details(docs_to_display)


def display_automated_checks(global_results: List[Dict]):
    """Displays the results of the automated checks with color-coding."""
    st.subheader("Automated Checks")
    for result in global_results:
        status = result.get('status', 'info')
        message = result.get('message', '')
        details = result.get('details', '')
        
        if status == 'passed':
            st.success(message)
        elif status == 'failed':
            st.error(message)
        elif status == 'warning':
            st.warning(message)
        elif status == 'orange':
            st.warning(f"🟠 {message}")
        else:
            st.info(message)
            
        if details:
            st.markdown(f"> {details}")

def display_brand_compliance(processed_docs: List[Dict]):
    """Displays the brand compliance results."""
    st.subheader("Brand Compliance")
    for doc in processed_docs:
        if 'brand_compliance' in doc and doc['brand_compliance'].get('success'):
            with st.expander(f"**{doc['filename']}**"):
                st.write("#### Fonts")
                st.write(", ".join(doc['brand_compliance']['fonts']))
                
                st.write("#### Colors")
                for color in doc['brand_compliance']['colors']:
                    rgb = color['rgb']
                    closest_brand_color = color['closest_brand_color']
                    delta_e = color['delta_e']
                    compliant = color['compliant']
                    
                    color_block = f'<div style="width: 20px; height: 20px; background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); display: inline-block; margin-right: 10px;"></div>'
                    
                    if compliant:
                        st.markdown(f"{color_block} **rgb({rgb[0]}, {rgb[1]}, {rgb[2]})** is compliant. Closest brand color: **{closest_brand_color}** (Delta E: {delta_e:.2f})", unsafe_allow_html=True)
                    else:
                        st.markdown(f"{color_block} **rgb({rgb[0]}, {rgb[1]}, {rgb[2]})** is not compliant. Closest brand color: **{closest_brand_color}** (Delta E: {delta_e:.2f})", unsafe_allow_html=True)


def display_document_details(docs: List[Dict]):
    """Displays the details of each processed document."""
    for doc in docs:
        with st.expander(f"**{doc['filename']}**"):
            st.write(f"**Document Type:** {doc['doc_type']}")
            st.write(f"**File Nature:** {doc['file_nature']}")
            
            if doc['qr_codes']:
                st.write("**QR Codes:**")
                for qr in doc['qr_codes']:
                    st.code(qr)
                    
            st.write("**Extracted Text:**")
            st.text(doc['text'])

def display_chat_interface(batch_data: Dict[str, Any]):
    # ... (chat logic is the same)
    pass
