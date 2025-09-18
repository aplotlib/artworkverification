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
        **1. 📋 Select Brand & Analysis Mode**
        - Use the buttons in the sidebar to choose **Vive** or **Coretech**. This loads the correct checklist.
        - Select your desired analysis mode. **OCR + AI Analysis** is recommended for the most comprehensive review.

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
            
            try:
                current_index = batch_options.index(st.session_state.get('current_batch_sku'))
            except (ValueError, TypeError):
                current_index = 0

            st.selectbox(
                "𗂬️ Review Previous Batch:",
                options=batch_options,
                key='selected_batch_for_review',
                index=current_index,
                on_change=on_batch_selection_change
            )
        
        st.divider()
        st.header("📋 Verification Setup")
        
        st.radio(
            "Select Analysis Mode:",
            options=["OCR + AI Analysis", "AI Analysis Only", "OCR Correction Only"],
            key='analysis_mode',
            help="Choose the type of analysis to perform. 'OCR + AI' is the default and most comprehensive."
        )
        
        st.text_area("Primary Validation Text (Optional)", key='primary_validation_text', height=150, help="Enter the source text for validation (UPC, SKU, etc.). This will be ignored if a primary validation file is uploaded.")
        
        col1, col2 = st.columns(2)
        if col1.button("Vive Product", use_container_width=True):
            st.session_state.brand_selection = "Vive"
        if col2.button("Coretech Product", use_container_width=True):
            st.session_state.brand_selection = "Coretech"
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        st.info(f"Selected Brand: **{brand_selection}**")
        st.info("AI Provider: **OpenAI**")

    if st.sidebar.button("🔍 Run Verification", use_container_width=True):
        st.session_state.run_validation = True
        st.rerun()

# ... (the rest of the UI components file remains the same)
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
    # This function would need to be updated to reflect the new validation results
    return "Report generation logic needs to be updated for new validation results."

def display_results_page(batch_data: Dict):
    """Displays the main results page with all tabs and details."""
    skus = batch_data.get('skus', [])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")
    
    # This section would need updating to show the new validation and AI results
    st.info("Report display logic needs to be updated to show new validation results.")

def display_automated_checks(global_results: List[Dict]):
    """Displays the results of the automated checks with color-coding."""
    st.subheader("Automated Checks")
    for result in global_results:
        # This function would need updating
        st.write(result)

def display_brand_compliance(processed_docs: List[Dict]):
    """Displays the brand compliance results."""
    st.subheader("Brand Compliance")
    for doc in processed_docs:
        # This function would need updating
        st.write(doc.get('filename'), doc.get('brand_compliance'))
        
def display_document_details(docs: List[Dict]):
    """Displays the details of each processed document."""
    for doc in docs:
        with st.expander(f"**{doc['filename']}**"):
            st.write(f"**Document Type:** {doc['doc_type']}")
            st.write(f"**File Nature:** {doc['file_nature']}")
            # ... and so on

def display_chat_interface(batch_data: Dict[str, Any]):
    # ... (chat logic is the same)
    pass
