import streamlit as st
import pandas as pd
import time
import csv
from io import StringIO
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
from ai_analyzer import AIReviewer, check_api_keys

# --- Security: Rate Limiting Constants ---
CHAT_COOLDOWN_SECONDS = 3

def display_header():
    """Displays the new, branded application header."""
    st.markdown("""
        <style>
            .header-title {
                font-size: 3rem; font-weight: 700; color: #2A9D8F; padding-bottom: 0rem;
            }
            .header-subtitle {
                font-size: 1.5rem; font-weight: 300; color: #264653; padding-top: 0rem;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">VAVE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">Vive Health Artwork Verification Engine</p>', unsafe_allow_html=True)


def display_instructions():
    """Displays the new, simplified 'How to Use' guide."""
    with st.expander("📖 How to Use VAVE in 3 Steps"):
        st.markdown("""
        **1. 📋 Paste Checklist & Set AI Query**
        - Paste your project's requirements into the **Verification Checklist** box in the sidebar. This creates a manual checklist and also tells the AI what to look for.
        - Optionally, ask the AI a specific question in the **Custom Summary** box.

        **2. 📁 Upload Artwork**
        - Drag and drop all artwork files for **one product** (or a whole folder) into the upload area. VAVE will start processing automatically.

        **3. ✅ Verify & Export**
        - Review the AI-generated report and complete the interactive manual checklist.
        - Download the checklist results for your records.
        - To start a new product, click the **"🚀 Start New Batch"** button.
        """)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str, str, bool]:
    """Renders the sidebar with a more intuitive layout and the new checklist feature."""
    with st.sidebar:
        st.header("⚙️ Controls")

        if st.button("🚀 Start New Batch", type="primary", help="Clear all data and start a fresh verification session."):
            # Keep messages and API keys, clear everything else
            messages = st.session_state.get('messages', [])
            st.session_state.clear()
            st.session_state.messages = messages
            st.rerun()

        if st.session_state.get('batches'):
            batch_options = list(st.session_state.batches.keys())
            current_index = batch_options.index(st.session_state.current_batch_sku) if st.session_state.current_batch_sku in batch_options else 0
            selected_sku = st.selectbox("🗂️ Reviewing Batch:", options=batch_options, index=current_index)
            if selected_sku != st.session_state.current_batch_sku:
                st.session_state.current_batch_sku = selected_sku
                st.rerun()
        st.divider()

        # --- NEW: Dynamic Checklist Input ---
        st.header("📋 Verification Checklist")
        checklist_text = st.text_area(
            "Paste your checklist here, one item per line. This will be used by the AI and will also generate a manual checklist.",
            key='checklist_text_input',
            height=200,
            placeholder="e.g.,\nMade in China\n1-year warranty\nProduct dimensions are 10cm x 5cm"
        )

        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area(
            "Ask the AI a specific question for the final summary.",
            placeholder="Does the manual mention a return policy?",
            max_chars=1000
        )
        if not api_keys.get('openai'):
            st.warning("An OpenAI API key is required for AI features.")

        st.divider()
        run_validation = st.button("🔍 Run Verification on Uploaded Files")
        run_test_validation = st.button("🧪 Run Test Validation")

    return run_validation, checklist_text, custom_instructions, run_test_validation


def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    return st.file_uploader(
        "📁 Upload Artwork Files (or Drag & Drop a Folder)",
        type=['pdf', 'csv', 'xlsx'],
        accept_multiple_files=True
    )

def display_dynamic_checklist(checklist_text: str, batch_key: str):
    """Renders an interactive checklist from user-pasted text."""
    st.header("✅ Manual Verification Checklist")
    with st.container(border=True):
        if not checklist_text.strip():
            st.info("Paste a checklist into the sidebar to generate an interactive list here.")
            return

        checklist_items = [line.strip() for line in checklist_text.split('\n') if line.strip()]

        # Ensure checklist state is initialized for the current context (batch or standalone)
        if 'checklist_state' not in st.session_state:
            st.session_state.checklist_state = {}
        if batch_key not in st.session_state.checklist_state:
            st.session_state.checklist_state[batch_key] = {}

        current_state = st.session_state.checklist_state[batch_key]

        for i, item in enumerate(checklist_items):
            is_checked = st.checkbox(item, value=current_state.get(item, False), key=f"check_{batch_key}_{i}")
            current_state[item] = is_checked

        st.divider()

        # Export functionality
        if st.button("Download Checklist Results"):
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['Task', 'Status', 'Batch/SKU'])
            for item, checked in current_state.items():
                writer.writerow([item, "DONE" if checked else "PENDING", batch_key])

            st.download_button(
                label="Click to Download CSV",
                data=output.getvalue(),
                file_name=f"{batch_key}_checklist_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def display_results_page(batch_data: Dict):
    """Displays the main results page for a processed batch."""
    skus = batch_data.get('skus', ['N/A'])
    st.header(f"📊 Verification Report for SKU(s): {', '.join(skus)}")

    global_results = batch_data.get('global_results', [])
    compliance_results = batch_data.get('compliance_results', [])
    quality_results = batch_data.get('quality_results', {})
    ai_summary = batch_data.get('ai_summary', "")

    # --- Dashboard ---
    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in global_results if r[0] == 'failed'])
        compliance_failures = sum(1 for res in compliance_results if res.get('status') == 'Fail')
        quality_issues = len(quality_results.get('issues', []))
        total_issues = rule_failures + compliance_failures + quality_issues

        sparkline_data = np.random.randn(20)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
        col2.metric("Rule-Based Failures", rule_failures)
        col3.metric("AI Compliance Failures", compliance_failures)
        col4.metric("Quality/Typo Issues", quality_issues)
    # --- End Dashboard ---

    # --- AI Summary ---
    with st.expander("🤖 AI Executive Summary", expanded=True):
        if not batch_data.get('ai_processing_complete'):
            st.info("🤖 AI analysis is running or has not been run...")
        else:
            st.markdown(ai_summary)

    # --- Detailed Tabs ---
    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        processed_docs = batch_data.get('processed_docs', [])
        tab_titles = ["📋 Automated Checks"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            st.subheader("AI-Powered Proofreading")
            if quality_results and 'issues' in quality_results and quality_results['issues']:
                for issue in quality_results['issues']:
                    st.markdown(f"**- Error:** `{issue['error']}` ➡️ **Correction:** `{issue['correction']}`")
                    st.caption(f"Context: \"...{issue['context']}...\"")
            else:
                st.markdown("✅ No spelling or grammar issues found.")
            st.divider()

            st.subheader("AI Compliance Check")
            if compliance_results:
                for res in compliance_results:
                    icon = "✅" if res.get('status') == 'Pass' else "❌"
                    st.markdown(f"**{icon} {res.get('phrase')}**")
                    st.caption(f"Reasoning: {res.get('reasoning')}")
            else:
                st.markdown("ℹ️ No AI compliance checks were run. Paste a checklist in the sidebar.")
            st.divider()

            st.subheader("Rule-Based Checks")
            for status, msg, _ in global_results:
                st.markdown(f"{'✅' if status == 'passed' else '❌'} {msg}")

        docs_by_type = defaultdict(list)
        for doc in processed_docs: docs_by_type[doc['doc_type'].replace('_', ' ').title()].append(doc)
        for i, title in enumerate(sorted(docs_by_type.keys())):
            with tabs[i + 1]:
                for doc in docs_by_type[title]:
                    nature_tag = f":blue[Shared File]" if doc['file_nature'] == 'Shared File' else f":orange[Unique Artwork]"
                    with st.expander(f"**{doc['filename']}** ({nature_tag})"):
                        ai_facts = batch_data.get('ai_facts', {})
                        if doc['doc_type'] == 'packaging_artwork' and ai_facts:
                            st.markdown("**AI Fact Extraction:**"); st.json(ai_facts)

                        text_preview = doc['text']
                        if len(text_preview) > 2000: text_preview = text_preview[:2000] + "\n\n... (text truncated)"
                        st.text_area("Extracted Text Preview", text_preview, height=250, key=f"text_{doc['filename']}", label_visibility="collapsed")


def display_pdf_previews(files: List[Dict[str, Any]]):
    """Renders PDFs using st.pdf."""
    if not files: return
    st.header("📄 PDF Previews")
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    st.pdf(pdf_file['bytes'])
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")


def display_chat_interface(batch_data: Dict = None):
    """Displays the AI chat interface."""
    st.header("💬 Chat with AI Assistant")
    st.caption("Ask about the report or general packaging compliance questions.")

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
                analysis_context = {}
                if batch_data:
                    analysis_context = {
                        "Executive Summary": batch_data.get('ai_summary'),
                        "Rule-Based Results": batch_data.get('global_results'),
                        "AI Compliance Results": batch_data.get('compliance_results'),
                        "SKUs in this Batch": batch_data.get('skus')
                    }
                response = reviewer.run_chatbot_interaction(st.session_state.messages, analysis_context)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
