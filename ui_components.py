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
        **1. 📋 Select a Brand and AI Provider**
        - Choose a brand from the dropdown to load the appropriate checklist. You can also select your preferred AI provider.
        **2. ✍️ Enter Keywords (Optional)**
        - Add any specific keywords that must or must not be present in the artwork.
        **3. 📁 Upload Artwork & Verify**
        - Drag and drop all artwork files for one product and click **"🔍 Run Verification"**.
        """)

def display_sidebar(api_keys: Dict[str, str], current_batch_data: Dict = None) -> Tuple[bool, str, str, str, str, bool]:
    """Renders the sidebar with brand selection, keyword inputs, and batch management."""
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
                st.rerun()

        st.divider()
        st.header("📋 Verification Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Vive"):
                st.session_state.brand_selection = "Vive"
        with col2:
            if st.button("Coretech"):
                st.session_state.brand_selection = "Coretech"
        
        brand_selection = st.session_state.get("brand_selection", "Vive")
        st.info(f"Selected Brand: **{brand_selection}**")

        ai_provider = st.selectbox("Select AI Provider:", options=["openai", "anthropic"])

        st.info("Enter keywords for the AI to verify. One per line.")
        must_contain_text = st.text_area("Must Contain Keywords:", height=100)
        must_not_contain_text = st.text_area("Must Not Contain Keywords:", height=100)
        
        st.divider()
        run_validation = st.button("🔍 Run Verification")
        run_test_validation = st.button("🧪 Run Test Validation")

    return run_validation, brand_selection, must_contain_text, must_not_contain_text, ai_provider, run_test_validation

def display_dynamic_checklist(brand: str, batch_key: str, batch_data: Dict = None):
    """Renders an interactive checklist based on the selected brand."""
    st.header("✅ Manual Verification Checklist")
    with st.container(border=True):
        checklist_data = AppConfig.CHECKLISTS.get(brand, {})
        if not checklist_data:
            st.info("Select a brand to see the corresponding checklist.")
            return

        if 'checklist_state' not in st.session_state: st.session_state.checklist_state = {}
        if batch_key not in st.session_state.checklist_state: st.session_state.checklist_state[batch_key] = {}
        current_state = st.session_state.checklist_state[batch_key]

        total_items = sum(len(items) for items in checklist_data.values())
        num_checked = 0

        for category, items in checklist_data.items():
            st.subheader(category)
            for i, item in enumerate(items):
                is_checked = st.checkbox(item, value=current_state.get(item, False), key=f"check_{batch_key}_{category}_{i}")
                current_state[item] = is_checked
                if is_checked:
                    num_checked += 1
        
        st.divider()
        if total_items > 0:
            progress = num_checked / total_items
            st.progress(progress, text=f"{num_checked} / {total_items} items completed")
            if progress == 1.0:
                st.success("🎉 Checklist complete! Great work.")
                st.balloons()
