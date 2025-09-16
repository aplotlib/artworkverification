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
        **1. 📋 Set Up Your Review**
        - Paste your project's requirements into the **Verification Checklist** box in the sidebar.
        **2. 📁 Upload Artwork**
        - Drag and drop all artwork files for **one product** (or an entire folder) into the upload area.
        **3. ✅ Verify & Export**
        - Click **"🔍 Run Verification"**. Review the AI report, complete the manual checklist, and check the new **Brand Compliance** tab.
        """)

def display_sidebar(api_keys: Dict[str, str], current_batch_data: Dict = None) -> Tuple[bool, str, str, bool]:
    """Renders the sidebar with state management for the checklist."""
    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🚀 Start New Batch", type="primary"):
            st.session_state.clear()
            st.rerun()
        if st.session_state.get('batches'):
            batch_options = list(st.session_state.batches.keys())
            selected_sku = st.selectbox("🗂️ Reviewing Batch:", options=batch_options, index=0)
            if selected_sku != st.session_state.get('current_batch_sku'):
                st.session_state.current_batch_sku = selected_sku
                st.session_state.checklist_text_input = st.session_state.batches[selected_sku].get('checklist_text', '')
                st.rerun()
        st.divider()
        st.header("📋 Verification Checklist")
        checklist_text = st.text_area("Paste checklist here:", key='checklist_text_input', height=200)
        st.header("🤖 AI Custom Summary")
        custom_instructions = st.text_area("Ask AI a specific question:", value=current_batch_data.get('custom_instructions', '') if current_batch_data else "")
        st.divider()
        run_validation = st.button("🔍 Run Verification")
        run_test_validation = st.button("🧪 Run Test Validation")
    return run_validation, checklist_text, custom_instructions, run_test_validation

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Renders the file uploader with guidance on file size."""
    return st.file_uploader(
        "📁 Upload Artwork Files (or Drag & Drop a Folder)",
        type=['pdf', 'csv', 'xlsx'], accept_multiple_files=True,
        help=f"Max individual file size: {AppConfig.MAX_FILE_SIZE_MB}MB"
    )

def display_brand_compliance_tab(processed_docs: List[Dict[str, Any]]):
    """Displays the new tab for brand color and font compliance."""
    st.subheader("🎨 Brand Compliance Analysis")
    for doc in processed_docs:
        if not doc.get('brand_compliance', {}).get('success'):
            continue
        
        with st.expander(f"**{doc['filename']}**"):
            # FONT ANALYSIS
            st.markdown("#### Font Usage")
            fonts_found = doc['brand_compliance'].get('fonts', [])
            if not fonts_found:
                st.markdown("No font data extracted.")
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
                st.markdown("No color data extracted from vector graphics.")
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
    """Displays the main results page with the new brand compliance tab."""
    st.header(f"📊 Verification Report for SKU(s): {', '.join(batch_data.get('skus', ['N/A']))}")
    # ... (Dashboard and AI Summary code remains the same) ...

    st.header("🔍 Details & Inspector")
    with st.container(border=True):
        processed_docs = batch_data.get('processed_docs', [])
        # UPGRADE: Added new Brand Compliance tab.
        tab_titles = ["📋 Automated Checks", "🎨 Brand Compliance"] + sorted(list(set(d['doc_type'].replace('_', ' ').title() for d in processed_docs)))
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            # ... (Automated Checks code remains the same) ...
            pass
        
        with tabs[1]:
            display_brand_compliance_tab(processed_docs)

        # ... (Loop for individual document type tabs remains the same) ...


# All other UI functions (display_dynamic_checklist, display_pdf_previews, etc.) remain the same
# as the previously provided versions. Their code is omitted here for brevity.
