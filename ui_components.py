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

        **2. ✍️ Upload a QC Sheet (Required for New Validator)**
        - For the automated validation to work, you **must** upload a QC sheet (CSV or XLSX) that the app can identify as the 'source of truth'.
        - The app will use this file to create a "Golden Record" to validate all other files against.

        **3. 📁 Upload Files & Run Verification**
        - Drag and drop all artwork files for one product into the uploader.
        - Click **"🔍 Run Verification"** to start the analysis. The app will process the files and display a detailed report with automatically checked-off items.
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
        
        # This is now less important as the QC sheet is the primary source
        st.text_area("Primary Validation Text (Legacy)", key='primary_validation_text', height=100, help="Legacy field. The new validator prioritizes the uploaded QC Sheet.")
        
        col1, col2 = st.columns(2)
        if col1.button("Vive Product", use_container_width=True):
            st.session_state.brand_selection = "Vive"
        if col2.button("Coretech Product", use_container_width=True):
            st.session_state.brand_selection = "Coretech"
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        st.info(f"Selected Brand: **{brand_selection}**")

        st.selectbox("Select AI Provider:", options=["openai", "anthropic"], help="Choose which AI service to use for analysis.", key='ai_provider')
        
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
        
        # Pass validation results to the checklist display
        current_batch_data = st.session_state.get('batches', {}).get(st.session_state.get('current_batch_sku'))
        validation_results = current_batch_data.get('validation_results', []) if current_batch_data else []
        
        display_dynamic_checklist(brand_selection, batch_key, validation_results)

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
        "📁 Upload Artwork Files (including a QC Sheet)",
        type=['pdf', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help=f"Max individual file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
    )
    
def display_dynamic_checklist(brand: str, batch_key: str, validation_results: List[Dict[str, Any]]):
    """Renders an interactive checklist that reflects automated validation results."""
    with st.container(border=True):
        checklist_data = AppConfig.CHECKLISTS.get(brand, {})
        if not checklist_data:
            st.warning("No checklist found for the selected brand.")
            return
            
        # --- New Logic to pre-populate checklist from validation ---
        validation_map = {
            "Product Name Consistency": "Product Name",
            "SKU ID": "SKU Consistency",
        }
        passed_checks = {
            result['check_name'] for result in validation_results if result.get('status') == 'passed'
        }
        # --- End New Logic ---

        if 'checklist_state' not in st.session_state: st.session_state.checklist_state = {}
        if batch_key not in st.session_state.checklist_state: st.session_state.checklist_state[batch_key] = {}
        current_state = st.session_state.checklist_state[batch_key]

        total_items = sum(len(items) for items in checklist_data.values())
        
        for category, items in checklist_data.items():
            st.subheader(category)
            for item in items:
                unique_key = f"check_{batch_key}_{category}_{item}"
                item_state_key = f"{brand}_{category}_{item}"
                
                # --- Conditionally disable and check the box ---
                is_validated = validation_map.get(item) in passed_checks
                
                # If it's validated, the value is True. Otherwise, use the stored manual state.
                checkbox_value = True if is_validated else current_state.get(item_state_key, False)
                
                is_checked = st.checkbox(
                    item, 
                    value=checkbox_value, 
                    key=unique_key,
                    disabled=is_validated # Disable the checkbox if it was auto-validated
                )
                
                # Only update the state if it's not disabled (i.e., it's a manual check)
                if not is_validated:
                    current_state[item_state_key] = is_checked
                # --- End Conditional Logic ---
        
        st.divider()
        if total_items > 0:
            # Recalculate checked items including validated ones
            num_auto_checked = sum(1 for item in validation_map if validation_map[item] in passed_checks)
            num_manual_checked = sum(1 for item_key, checked in current_state.items() if checked and not any(item_key.endswith(k) for k in validation_map))

            num_checked = num_auto_checked + num_manual_checked
            progress = num_checked / total_items if total_items > 0 else 0
            st.progress(progress, text=f"{num_checked} / {total_items} items completed")
            if progress == 1.0:
                st.success("🎉 Checklist complete! Great work.")
                st.balloons()

def generate_report_text(batch_data: Dict) -> str:
    # (No changes needed for this function)
    pass

def display_results_page(batch_data: Dict):
    """Displays the main results page with all tabs and details."""
    skus = batch_data.get('skus', [])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")
    
    # ... (download button logic is the same)

    validation_results = batch_data.get('validation_results', [])
    ai_summary = batch_data.get('ai_summary', "AI summary could not be generated.")

    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in validation_results if r.get('status') == 'failed'])
        # ... (rest of the dashboard logic can be updated to use new results)
        total_issues = rule_failures # Placeholder for more detailed issue counting
        col1, col2 = st.columns(2)
        col1.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
        col2.metric("Rule-Based Failures", rule_failures)

    with st.expander("🤖 AI Executive Summary", expanded=True):
        st.markdown(ai_summary)

    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        processed_docs = batch_data.get('processed_docs', [])
        # --- Add New Tab for AI Consistency Audit ---
        tab_titles = ["📋 Automated Checks", "🕵️ AI Consistency Audit", "🎨 Brand Compliance"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            display_automated_checks(validation_results) # Use new validation_results

        with tabs[1]: # New Tab Display
            display_ai_consistency_audit(batch_data.get('ai_consistency_results', {}))

        with tabs[2]:
            display_brand_compliance(processed_docs)

        for i, doc_type in enumerate(sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))):
            with tabs[i+3]:
                docs_to_display = [d for d in processed_docs if d['doc_type'].replace('_', ' ').title() == doc_type]
                display_document_details(docs_to_display)

def display_automated_checks(validation_results: List[Dict]):
    """Displays the results of the new rule-based validation engine."""
    st.subheader("Rule-Based Validation Checks")
    for result in validation_results:
        status = result.get('status', 'info')
        message = result.get('message', '')
        details = result.get('details', '')
        
        if status == 'passed':
            st.success(message)
        elif status == 'failed':
            st.error(message)
        elif status == 'warning':
            st.warning(message)
        else:
            st.info(message)
            
        if details:
            st.markdown(f"> {details}")

def display_ai_consistency_audit(audit_results: Dict):
    """Displays the results from the AI cross-document consistency audit."""
    st.subheader("AI Cross-Document Consistency Audit")
    if not audit_results.get('success'):
        st.error(f"AI Consistency Audit failed: {audit_results.get('error')}")
        return

    inconsistencies = audit_results.get('data', {}).get('inconsistencies', [])
    if not inconsistencies:
        st.success("✅ No logical inconsistencies were found between documents.")
        return

    for issue in inconsistencies:
        confidence = issue.get('confidence', 'low')
        description = issue.get('description', 'No description provided.')
        sources = ", ".join(issue.get('sources', []))
        
        if confidence == 'high':
            st.error(f"🔴 **High Confidence:** {description} (Sources: {sources})")
        elif confidence == 'medium':
            st.warning(f"🟠 **Medium Confidence:** {description} (Sources: {sources})")
        else:
            st.info(f"🟡 **Low Confidence:** {description} (Sources: {sources})")


def display_brand_compliance(processed_docs: List[Dict]):
    # (No changes needed for this function)
    pass

def display_document_details(docs: List[Dict]):
    # (No changes needed for this function)
    pass

def display_chat_interface(batch_data: Dict[str, Any]):
    # (No changes needed for this function)
    pass
