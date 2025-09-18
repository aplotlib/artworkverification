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

def display_header():
    """Displays the branded application header."""
    st.markdown("""
        <style>
            .header-title { font-size: 2.5rem; font-weight: 700; color: #2A9D8F; padding-bottom: 0rem; }
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

        **2. ✍️ Add Specific Checks (Optional)**
        - Enter any keywords that **must** or **must not** appear in the artwork files for the AI to verify.

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
            st.selectbox("𗂬️ Review Previous Batch:", options=batch_options, key='current_batch_sku', on_change=lambda: st.session_state.update(messages=[]))
        
        st.divider()
        st.header("📋 Verification Setup")
        
        col1, col2 = st.columns(2)
        if col1.button("Vive Product", use_container_width=True):
            st.session_state.brand_selection = "Vive"
        if col2.button("Coretech Product", use_container_width=True):
            st.session_state.brand_selection = "Coretech"
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        st.info(f"Selected Brand: **{brand_selection}**")

        st.selectbox("Select AI Provider:", options=["openai", "anthropic"], help="Choose which AI service to use for analysis.", key='ai_provider')
        
    st.session_state.run_validation = st.sidebar.button("🔍 Run Verification", use_container_width=True)


def display_main_interface():
    """Displays the main UI with tabs for interaction and results."""
    
    # AI Chat and Instructions at the top
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
        
        # Calculate checked items based on the current state
        num_checked = sum(1 for item_key in current_state if current_state.get(item_key, False))
        
        for category, items in checklist_data.items():
            st.subheader(category)
            for item in items:
                # CRITICAL FIX: Create a unique key for each checkbox
                unique_key = f"check_{batch_key}_{category}_{item}"
                item_state_key = f"{brand}_{category}_{item}"
                
                is_checked = st.checkbox(item, value=current_state.get(item_state_key, False), key=unique_key)
                current_state[item_state_key] = is_checked
        
        st.divider()
        if total_items > 0:
            # Recalculate num_checked after rendering checkboxes
            num_checked = sum(1 for item_key in current_state if current_state.get(item_key, False))
            progress = num_checked / total_items if total_items > 0 else 0
            st.progress(progress, text=f"{num_checked} / {total_items} items completed")
            if progress == 1.0:
                st.success("🎉 Checklist complete! Great work.")
                st.balloons()

def display_results_page(batch_data: Dict):
    """Displays the main results page with all tabs and details."""
    skus = batch_data.get('skus', ['N/A'])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")

    global_results = batch_data.get('global_results', [])
    compliance_results_data = batch_data.get('compliance_results', {})
    compliance_results = compliance_results_data.get('data', []) if compliance_results_data.get('success') else []
    quality_issues_data = batch_data.get('quality_results', {})
    quality_issues = len(quality_issues_data.get('data', {}).get('issues', [])) if quality_issues_data.get('success') else 0
    ai_summary = batch_data.get('ai_summary', "AI summary could not be generated.")

    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in global_results if r[0] == 'failed'])
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
            st.subheader("AI-Powered Proofreading")
            if not quality_issues_data.get('success'):
                st.error(f"Proofreading failed: {quality_issues_data.get('error')}")
            elif not quality_issues:
                st.markdown("✅ No spelling or grammar issues found.")
            else:
                for issue in quality_issues_data['data']['issues']:
                    st.markdown(f"**- Error:** `{issue['error']}` ➡️ **Correction:** `{issue['correction']}`")
            st.divider()

            st.subheader("AI Compliance Check")
            if not compliance_results_data.get('success'):
                st.error(f"Compliance check failed: {compliance_results_data.get('error')}")
            elif compliance_results:
                for res in compliance_results:
                    icon = "✅" if res.get('status') == 'Pass' else "❌"
                    st.markdown(f"**{icon} {res.get('phrase')}** — *{res.get('reasoning')}*")
            st.divider()

            st.subheader("Rule-Based Checks")
            for status, msg, _ in global_results:
                st.markdown(f"{'✅' if status == 'passed' else '❌'} {msg}")
        
        with tabs[1]:
            st.write("Brand compliance tab not yet implemented.")

        docs_by_type = defaultdict(list)
        for doc in processed_docs:
            docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
        
        for i, title in enumerate(sorted(docs_by_type.keys())):
            with tabs[i + 2]: # Offset by 2 for the first two tabs
                for doc in docs_by_type[title]:
                    with st.expander(f"**{doc['filename']}**"):
                        text_preview = doc['text']
                        if len(text_preview) > 500:
                            st.text(text_preview[:500] + "...")
                            if st.button("Show More", key=f"show_more_{doc['filename']}"):
                                st.text(text_preview)
                        else:
                            st.text(text_preview)


def display_chat_interface(batch_data: Dict[str, Any]):
    """Renders the chat interface for interacting with the AI about the current batch."""
    st.subheader("💬 Chat with AI Co-pilot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            api_keys = check_api_keys()
            provider = st.session_state.get('ai_provider', 'openai')
            
            if not api_keys.get(provider):
                full_response = f"Error: {provider.capitalize()} API key not configured."
            else:
                reviewer = AIReviewer(api_keys, provider=provider)
                analysis_context = {
                    "summary": batch_data.get('ai_summary'),
                    "files": [doc['filename'] for doc in batch_data.get('processed_docs', [])]
                } if batch_data else {}

                full_response = reviewer.run_chatbot_interaction(
                    st.session_state.messages,
                    analysis_context
                )

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
