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
    """Displays the new, branded application header."""
    st.markdown("""
        <style>
            .header-title { font-size: 2.5rem; font-weight: 700; color: #2A9D8F; padding-bottom: 0rem; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">Vive Health Artwork Verification Co-pilot</h1>', unsafe_allow_html=True)

def display_instructions():
    """Displays the simplified 'How to Use' guide."""
    with st.expander("📖 How to Use the Co-pilot in 3 Steps"):
        st.markdown("""
        **1. 📋 Set Up Your Review**
        - Paste your project's requirements into the **Verification Checklist** box in the sidebar. This creates your manual checklist and also tells the AI what to look for.
        - Optionally, ask the AI a specific question in the **Custom Summary** box.

        **2. 📁 Upload Artwork**
        - Drag and drop all artwork files for **one product** (or an entire folder) into the upload area.

        **3. ✅ Verify & Export**
        - Click **"🔍 Run Verification"**. Review the AI report, complete the interactive checklist, and download the results.
        - To start a new product, click **"🚀 Start New Batch"**.
        """)

def display_sidebar(api_keys: Dict[str, str], current_batch_data: Dict = None) -> Tuple[bool, str, str, bool]:
    """Renders the sidebar with improved state management for the checklist."""
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🚀 Start New Batch", type="primary", help="Clear all data and start a fresh verification session."):
            st.session_state.clear()
            st.rerun()

        if st.session_state.get('batches'):
            batch_options = list(st.session_state.batches.keys())
            current_index = batch_options.index(st.session_state.current_batch_sku) if st.session_state.current_batch_sku in batch_options else 0
            selected_sku = st.selectbox("🗂️ Reviewing Batch:", options=batch_options, index=current_index)
            if selected_sku != st.session_state.current_batch_sku:
                st.session_state.current_batch_sku = selected_sku
                st.session_state.checklist_text_input = st.session_state.batches[selected_sku].get('checklist_text', '')
                st.rerun()
        st.divider()

        st.header("📋 Verification Checklist")
        checklist_text = st.text_area(
            "Paste your checklist here, one item per line.",
            key='checklist_text_input',
            height=200,
            placeholder="e.g.,\nMade in China\n1-year warranty\nProduct dimensions are 10cm x 5cm"
        )

        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area(
            "Ask the AI a specific question for the summary.",
            value=current_batch_data.get('custom_instructions', '') if current_batch_data else "",
            max_chars=1000
        )
        if not api_keys.get('openai'):
            st.warning("An OpenAI API key is required for AI features.")

        st.divider()
        run_validation = st.button("🔍 Run Verification")
        run_test_validation = st.button("🧪 Run Test Validation")

    return run_validation, checklist_text, custom_instructions, run_test_validation

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Renders the file uploader with guidance on file size."""
    # UPGRADE: Added help text to inform users about the file size limit.
    return st.file_uploader(
        "📁 Upload Artwork Files (or Drag & Drop a Folder)",
        type=['pdf', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help=f"Maximum individual file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
    )

def display_dynamic_checklist(checklist_text: str, batch_key: str, batch_data: Dict = None):
    """Renders an interactive checklist with progress bar and enhanced export."""
    st.header("✅ Manual Verification Checklist")
    with st.container(border=True):
        if not checklist_text.strip():
            st.info("Paste a checklist into the sidebar to generate an interactive list here.")
            return

        checklist_items = [line.strip() for line in checklist_text.split('\n') if line.strip()]
        total_items = len(checklist_items)

        if 'checklist_state' not in st.session_state: st.session_state.checklist_state = {}
        if batch_key not in st.session_state.checklist_state: st.session_state.checklist_state[batch_key] = {}
        current_state = st.session_state.checklist_state[batch_key]
        
        num_checked = sum(1 for item in checklist_items if current_state.get(item, False))

        for i, item in enumerate(checklist_items):
            is_checked = st.checkbox(item, value=current_state.get(item, False), key=f"check_{batch_key}_{i}")
            current_state[item] = is_checked

        st.divider()
        if total_items > 0:
            progress = num_checked / total_items
            st.progress(progress, text=f"{num_checked} / {total_items} items completed")
            if progress == 1.0:
                st.success("🎉 Checklist complete! Great work.")
                st.balloons()

        if st.button("Download Checklist Report"):
            output = StringIO()
            if batch_data:
                total_issues = len([r for r in batch_data.get('global_results', []) if r[0] == 'failed']) + len(batch_data.get('compliance_results', {}).get('data', []))
                output.write(f"Verification Report for SKU(s):,{','.join(batch_data.get('skus', []))}\n")
                output.write(f"Report Generated On (UTC):,{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                output.write(f"Automated Issues Found:,{total_issues}\n\n")

            writer = csv.writer(output)
            writer.writerow(['Checklist Task', 'Status', 'Batch/SKU'])
            for item in checklist_items:
                writer.writerow([item, "DONE" if current_state.get(item) else "PENDING", batch_key])

            st.download_button(
                label="Click to Download CSV", data=output.getvalue(),
                file_name=f"{batch_key}_checklist_report_{time.strftime('%Y%m%d')}.csv", mime="text/csv"
            )

def display_results_page(batch_data: Dict):
    """Displays the main results page with improved error visibility."""
    skus = batch_data.get('skus', ['N/A'])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")

    global_results = batch_data.get('global_results', [])
    compliance_results = batch_data.get('compliance_results', {}).get('data', [])
    quality_issues_data = batch_data.get('quality_results', {})
    ai_summary = batch_data.get('ai_summary', "")

    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in global_results if r[0] == 'failed'])
        compliance_failures = sum(1 for res in compliance_results if res.get('status') == 'Fail')
        quality_issues = len(quality_issues_data.get('data', {}).get('issues', [])) if quality_issues_data.get('success') else 0
        total_issues = rule_failures + compliance_failures + quality_issues

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
        col2.metric("Rule-Based Failures", rule_failures)
        col3.metric("AI Compliance Failures", compliance_failures)
        col4.metric("Quality/Typo Issues", quality_issues)

    with st.expander("🤖 AI Executive Summary", expanded=True):
        if not batch_data.get('ai_processing_complete'):
            st.info("🤖 AI analysis is running...")
        else:
            st.markdown(ai_summary)

    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        tab_titles = ["📋 Automated Checks"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in batch_data.get('processed_docs', []))))
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
            if not batch_data.get('compliance_results', {}).get('success'):
                st.error(f"Compliance check failed: {batch_data['compliance_results'].get('error')}")
            elif compliance_results:
                for res in compliance_results:
                    icon = "✅" if res.get('status') == 'Pass' else "❌"
                    st.markdown(f"**{icon} {res.get('phrase')}** — *{res.get('reasoning')}*")
            st.divider()

            st.subheader("Rule-Based Checks")
            for status, msg, _ in global_results:
                st.markdown(f"{'✅' if status == 'passed' else '❌'} {msg}")

def display_pdf_previews(files: List[Dict[str, Any]]):
    """Renders PDFs using st.pdf."""
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        st.header("📄 PDF Previews")
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    st.pdf(pdf_file['bytes'])
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")

def display_chat_interface(batch_data: Dict = None):
    st.header("💬 Chat with AI Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about the verification report..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                api_keys = check_api_keys()
                reviewer = AIReviewer(api_keys)
                context = {"Executive Summary": batch_data.get('ai_summary')} if batch_data else {}
                response = reviewer.run_chatbot_interaction(st.session_state.messages, context)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
