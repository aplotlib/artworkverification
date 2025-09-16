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
    with st.expander("📖 How to Use the Co-pilot in 3 Steps"):
        st.markdown("""
        **1. 📋 Review the Checklist**
        - A default checklist is provided below. You can edit or replace it with your own project-specific requirements before running the verification.
        **2. 📁 Upload Artwork**
        - Drag and drop all artwork files for **one product** (or an entire folder) into the upload area.
        **3. ✅ Verify & Export**
        - Click **"🔍 Run Verification"**. Review the AI report, complete the interactive manual checklist, and check the new **Brand Compliance** tab for color and font analysis.
        """)

def display_sidebar(api_keys: Dict[str, str], current_batch_data: Dict = None) -> Tuple[bool, str, str, bool]:
    """Renders the sidebar with the new default checklist and improved state management."""
    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🚀 Start New Batch", type="primary", help="Clear all data and start a fresh verification session."):
            st.session_state.clear()
            st.rerun()

        if st.session_state.get('batches'):
            batch_options = list(st.session_state.batches.keys())
            selected_sku = st.selectbox("🗂️ Reviewing Batch:", options=batch_options, index=0)
            if selected_sku != st.session_state.get('current_batch_sku'):
                st.session_state.current_batch_sku = selected_sku
                # When batch changes, update the text area to show the correct checklist.
                st.session_state.checklist_text_input = st.session_state.batches[selected_sku].get('checklist_text', AppConfig.DEFAULT_CHECKLIST)
                st.rerun()
        st.divider()
        st.header("📋 Verification Checklist")
        st.info("A default checklist is loaded below. You can edit it for your specific needs.")
        checklist_text = st.text_area(
            "Paste checklist here:",
            key='checklist_text_input',
            height=250
        )
        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area(
            "Ask AI a specific question:",
            value=current_batch_data.get('custom_instructions', '') if current_batch_data else ""
        )
        st.divider()
        run_validation = st.button("🔍 Run Verification")
        run_test_validation = st.button("🧪 Run Test Validation")
    return run_validation, checklist_text, custom_instructions, run_test_validation

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Renders the file uploader with guidance on file size."""
    return st.file_uploader(
        "📁 Upload Artwork Files (or Drag & Drop a Folder)",
        type=['pdf', 'csv', 'xlsx'],
        accept_multiple_files=True,
        help=f"Max individual file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
    )

def display_dynamic_checklist(checklist_text: str, batch_key: str, batch_data: Dict = None):
    """Renders an interactive checklist with progress bar and enhanced export."""
    st.header("✅ Manual Verification Checklist")
    with st.container(border=True):
        if not checklist_text.strip():
            st.info("Paste a checklist into the sidebar to generate an interactive list here.")
            return

        checklist_items = [line.strip() for line in checklist_text.split('\n') if line.strip() and not line.strip().startswith('#')]
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
                label="Click to Download CSV",
                data=output.getvalue(),
                file_name=f"{batch_key}_checklist_report_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def display_brand_compliance_tab(processed_docs: List[Dict[str, Any]]):
    """Displays the new tab for brand color and font compliance."""
    st.subheader("🎨 Brand Compliance Analysis")
    for doc in processed_docs:
        # Only process docs where brand compliance analysis was successful
        if not doc.get('brand_compliance', {}).get('success'):
            continue
        
        with st.expander(f"**{doc['filename']}**"):
            # FONT ANALYSIS
            st.markdown("#### Font Usage")
            fonts_found = doc['brand_compliance'].get('fonts', [])
            if not fonts_found:
                st.markdown("No font data extracted from this file.")
            else:
                approved_fonts = AppConfig.BRAND_GUIDE['fonts']['main'] + AppConfig.BRAND_GUIDE['fonts']['secondary']
                for font in fonts_found:
                    is_compliant = any(approved.lower() in font.lower() for approved in approved_fonts)
                    icon = "✅" if is_compliant else "❌"
                    st.markdown(f"{icon} {font}")
            
            # COLOR ANALYSIS
            st.markdown("#### Color Palette Analysis")
            colors_found = doc['brand_compliance'].get('colors', [])
            if not colors_found:
                st.markdown("No color data extracted from vector graphics in this file.")
            else:
                for color_data in colors_found:
                    rgb = color_data['rgb']
                    color_hex = '#%02x%02x%02x' % rgb
                    icon = "✅" if color_data['compliant'] else "❌"
                    
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="width: 20px; height: 20px; background-color: {color_hex}; border: 1px solid #ccc; margin-right: 10px;"></div>
                        <div>
                            {icon} <b>{color_hex.upper()}</b> | Closest Match: <b>{color_data['closest_brand_color']}</b> 
                            (ΔE: {color_data['delta_e']:.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

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
            display_brand_compliance_tab(processed_docs)

        docs_by_type = defaultdict(list)
        for doc in processed_docs:
            docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
        
        for i, title in enumerate(sorted(docs_by_type.keys())):
            with tabs[i + 2]: # Offset by 2 for the first two tabs
                for doc in docs_by_type[title]:
                    with st.expander(f"**{doc['filename']}**"):
                        text_preview = doc['text']
                        if len(text_preview) > 2000: text_preview = text_preview[:2000] + "\n\n... (text truncated)"
                        st.text_area("Extracted Text Preview", text_preview, height=250, key=f"text_{doc['filename']}", label_visibility="collapsed")


def display_pdf_previews(files: List[Dict[str, Any]]):
    """Renders PDFs using st.pdf."""
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        st.header("📄 PDF Previews")
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    # Convert tuple back to bytes for st.pdf
                    st.pdf(bytes(pdf_file['bytes']))
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")

def display_chat_interface(batch_data: Dict = None):
    """Displays the AI chat interface."""
    st.header("💬 Chat with AI Assistant")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
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
