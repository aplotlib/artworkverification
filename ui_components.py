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
    st.markdown("""
        <style>
            .header-title {
                font-size: 3rem;
                font-weight: 700;
                color: #2A9D8F;
                padding: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="header-title">🎨 Artwork Verification Co-pilot</h1>', unsafe_allow_html=True)

def display_instructions():
    # Updated instructions for new features
    with st.expander("📖 How to Use This Tool"):
        st.markdown("""
        **1. Upload Files or a Folder**: To start a new product review, click **"Start New Batch"** in the sidebar. Then, drag and drop all related artwork files **or an entire folder** for a single product.
        **2. Switch Between Batches**: If you upload multiple product batches, a dropdown will appear in the sidebar to switch between reports.
        **3. Review Report**: Key metrics appear at the top. Use the tabs and the new **Manual Checklist** to complete your review.
        **4. Export & Chat**: Download the checklist results for your records and use the chat interface to ask questions.
        """)

def manage_batch_selection():
    """Manages the UI for selecting the current batch in the sidebar."""
    if st.session_state.batches:
        batch_options = list(st.session_state.batches.keys())
        # Use the index to manage the selectbox state
        current_index = batch_options.index(st.session_state.current_batch_sku) if st.session_state.current_batch_sku in batch_options else 0

        selected_sku = st.selectbox(
            "Reviewing Batch:",
            options=batch_options,
            index=current_index,
            key='batch_selector'
        )
        # If selection changes, update the current batch SKU
        if selected_sku != st.session_state.current_batch_sku:
            st.session_state.current_batch_sku = selected_sku
            st.rerun()

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str, bool]:
    with st.sidebar:
        st.header("⚙️ Controls")

        # --- NEW: Batch Management ---
        if st.button("🚀 Start New Batch", type="primary"):
            st.session_state.clear() # Clears everything to start fresh
            st.rerun()

        manage_batch_selection()
        st.divider()
        # --- End Batch Management ---

        run_validation = st.button("🔍 Run Verification on Uploaded Files")
        run_test_validation = st.button("🧪 Run Test Validation")

        st.header("📝 AI Compliance Check")
        reference_text = st.text_area(
            "Enter text for the AI to verify in uploaded files, one phrase per line.",
            placeholder="Made in China\n1-year warranty", max_chars=1000
        )
        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area("Ask the AI a specific question for the final summary.",
                                           placeholder="Does the manual mention a return policy?", max_chars=1000)

        if not api_keys.get('openai'):
            st.warning("An OpenAI API key is required for AI features.")
    return run_validation, custom_instructions, reference_text, run_test_validation

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    # Updated text to reflect folder upload capability
    return st.file_uploader("Upload all artwork files for one product (PDF, CSV, XLSX). You can also drag and drop a folder.",
                             type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True)

def display_manual_checklist(batch_data: Dict):
    """Renders the new interactive manual verification checklist."""
    st.header("✅ Manual Verification Checklist")

    checklist_items = {
        "Country of Origin": "Verify 'Made in China' (or appropriate country) is present on packaging and product.",
        "Dimensions Match": "Confirm product dimensions on packaging, manual, and spec sheet are accurate and consistent.",
        "Shipping Mark File": "Confirm the shipping mark file is present and correct for the SKU.",
        "UDI File Present": "Confirm the UDI file is present and the format is correct.",
        "UPC Present": "Confirm the UPC is present on all required artwork (e.g., packaging, tags).",
        "Contact Information": "Verify the distributor's address and phone number are correct on all relevant documents.",
        "Proposition 65 Warning": "If applicable, confirm the Prop 65 warning is present and correctly worded.",
    }

    checklist_state = batch_data.get('checklist_state', {})

    for key, description in checklist_items.items():
        # Use the key to store the state (True/False)
        is_checked = st.checkbox(f"**{key}**", value=checklist_state.get(key, False), help=description)
        checklist_state[key] = is_checked # Update state on interaction

    # Store the updated checklist state back into the batch data
    batch_data['checklist_state'] = checklist_state

    # --- NEW: Export Checklist Button ---
    if st.button("Download Checklist Results"):
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Task', 'Status', 'SKU'])
        status = "Completed" if all(checklist_state.values()) else "Incomplete"
        for key, checked in checklist_state.items():
            writer.writerow([key, "DONE" if checked else "PENDING", ", ".join(batch_data['skus'])])
        
        st.download_button(
            label="Click to Download",
            data=output.getvalue(),
            file_name=f"{batch_data['skus'][0]}_checklist_{time.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def display_results_page(batch_data: Dict):
    st.header(f"📊 Verification Report for SKU(s): {', '.join(batch_data['skus'])}")

    if not batch_data.get('processed_docs'):
        st.warning("No files were processed for this batch.")
        return

    global_results = batch_data.get('global_results', [])
    compliance_results = batch_data.get('compliance_results', [])
    quality_results = batch_data.get('quality_results', {})
    ai_summary = batch_data.get('ai_summary', "")

    # --- UI/UX Improvement: Summary Dashboard with Sparkline ---
    with st.container(border=True):
        st.subheader("📈 Report Dashboard")
        rule_failures = len([r for r in global_results if r[0] == 'failed'])
        compliance_failures = sum(1 for res in compliance_results if res.get('status') == 'Fail')
        quality_issues = len(quality_results.get('issues', []))
        total_issues = rule_failures + compliance_failures + quality_issues

        # Placeholder for sparkline data. In a real app, this would come from a database of historical results.
        sparkline_data = np.random.randn(20)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # NEW: Using st.metric with a sparkline
            st.metric("Total Issues", total_issues, "High" if total_issues > 0 else "None", delta_color="inverse")
            st.line_chart(sparkline_data, height=100) # Sparkline visualization
        col2.metric("Rule-Based Failures", rule_failures)
        col3.metric("AI Compliance Failures", compliance_failures)
        col4.metric("Quality/Typo Issues", quality_issues)

        if total_issues > 0:
            st.error(f"**🚨 {total_issues} Potential Issue(s) Detected**")
        else:
            st.success("**✅ No Potential Issues Detected**")

    # --- Executive Summary ---
    with st.expander("🤖 Read Executive Summary", expanded=True):
        if not batch_data.get('ai_processing_complete'):
            st.info("🤖 AI analysis is running or has not been run...")
        else:
            st.markdown(ai_summary)

    # --- NEW: Manual Checklist ---
    display_manual_checklist(batch_data)

    # --- Detailed Tabs ---
    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        processed_docs = batch_data.get('processed_docs', [])
        tab_titles = ["📋 Reports & Checks"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            if quality_results and 'issues' in quality_results:
                with st.expander("AI-Powered Proofreading", expanded=True):
                    if not quality_results['issues']:
                        st.markdown("✅ No spelling or grammar issues found.")
                    else:
                        for issue in quality_results['issues']:
                            st.markdown(f"**- Error:** `{issue['error']}` -> **Correction:** `{issue['correction']}`")
                            st.caption(f"Context: \"...{issue['context']}...\"")
                st.divider()

            if compliance_results:
                with st.expander("AI Compliance Check", expanded=True):
                    for res in compliance_results:
                        icon = "✅" if res.get('status') == 'Pass' else "❌"
                        st.markdown(f"**{icon} {res.get('phrase')}**")
                        st.caption(f"Reasoning: {res.get('reasoning')}")
                st.divider()

            with st.expander("Rule-Based Checks", expanded=True):
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
    """Renders PDFs using the new st.pdf function for better inline viewing."""
    if not files: return
    st.header("📄 PDF Previews")
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        for pdf_file in pdf_files:
            with st.expander(f"View Preview: {pdf_file['name']}"):
                try:
                    # NEW: Using st.pdf for native PDF rendering
                    st.pdf(pdf_file['bytes'])
                except Exception as e:
                    st.error(f"Could not render preview for {pdf_file['name']}. Error: {e}")

def display_chat_interface(batch_data: Dict = None):
    """Displays the standalone AI chat interface with rate limiting and context."""
    st.header("💬 Chat with AI Assistant")
    st.caption("Ask me about the report or anything related to medical device packaging.")

    if "messages" not in st.session_state: st.session_state.messages = []
    if "last_chat_time" not in st.session_state: st.session_state.last_chat_time = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the verification report..."):
        current_time = time.time()
        if current_time - st.session_state.last_chat_time < CHAT_COOLDOWN_SECONDS:
            st.toast(f"Please wait {CHAT_COOLDOWN_SECONDS} seconds between messages.", icon="⏳")
        else:
            st.session_state.last_chat_time = current_time
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    api_keys = check_api_keys()
                    reviewer = AIReviewer(api_keys)

                    analysis_context = {}
                    if batch_data: # Pass current batch data to the chatbot
                        analysis_context = {
                            "Executive Summary": batch_data.get('ai_summary'),
                            "Rule-Based Results": batch_data.get('global_results'),
                            "AI Compliance Results": batch_data.get('compliance_results'),
                            "AI Quality Results": batch_data.get('quality_results'),
                            "SKUs in this Batch": batch_data.get('skus')
                        }

                    response = reviewer.run_chatbot_interaction(st.session_state.messages, analysis_context)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
