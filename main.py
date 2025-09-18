import streamlit as st
import time
import os
from config import AppConfig
from file_processor import process_files_cached
from validator import ArtworkValidator # Updated import
from ai_analyzer import AIReviewer, check_api_keys
from ui_components import (
    display_header,
    display_sidebar,
    display_main_interface
)

@st.cache_data(show_spinner=False)
def run_analysis_cached(_file_tuples, must_contain_text, must_not_contain_text, custom_instructions, ai_provider, primary_validation_text):
    """
    Performs all validation and AI analysis. Caching this function prevents
    re-running AI calls for the same data.
    """
    with st.spinner("Step 1/3: Processing and extracting text from files..."):
        processed_docs, skus = process_files_cached(_file_tuples)
    
    with st.spinner("Step 2/3: Running new rule-based validation engine..."):
        # --- New Validation Logic ---
        validator = ArtworkValidator()
        # Note: We pass a copy of processed_docs if the validator modifies them in place.
        # Based on the new design, it doesn't, but this is good practice.
        validation_results = validator.validate(processed_docs.copy()) 
        # --- End New Logic ---

    with st.spinner("Step 3/3: Running AI analysis and validation... This may take a moment."):
        all_text = "\n\n".join(doc['text'] for doc in processed_docs)
        
        api_keys = check_api_keys()
        if not api_keys.get(ai_provider):
            return {
                "error": f"{ai_provider.capitalize()} API key not found.", 
                "validation_results": validation_results, # Pass the new results
                "skus": skus, 
                "processed_docs": processed_docs
            }

        reviewer = AIReviewer(api_keys, provider=ai_provider)
        
        packaging_doc_text = next((d['text'] for d in processed_docs if d['doc_type'] == 'packaging_artwork'), all_text)
        
        # --- New AI Consistency Audit Call ---
        ai_consistency_results = reviewer.run_ai_consistency_audit(processed_docs)
        # --- End New Call ---
        
        ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text)
        compliance_results = reviewer.run_ai_compliance_check(ai_facts.get('data', {}), must_contain_text, must_not_contain_text)
        quality_results = reviewer.run_ai_quality_check(all_text)
        ai_summary = reviewer.generate_executive_summary(processed_docs, validation_results, compliance_results.get('data', []), quality_results, custom_instructions)
    
    return {
        "validation_results": validation_results, # Changed from global_results
        "skus": skus,
        "ai_summary": ai_summary, 
        "ai_facts": ai_facts, 
        "compliance_results": compliance_results,
        "quality_results": quality_results, 
        "processed_docs": processed_docs,
        "ai_consistency_results": ai_consistency_results # Add new results to batch
    }

def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    
    # --- Session State Initialization ---
    if 'batches' not in st.session_state: 
        st.session_state.batches = {}
    if 'current_batch_sku' not in st.session_state: 
        st.session_state.current_batch_sku = None
    if 'messages' not in st.session_state: 
        st.session_state.messages = []
    if 'brand_selection' not in st.session_state: 
        st.session_state.brand_selection = "Vive"
    if 'run_validation' not in st.session_state: 
        st.session_state.run_validation = False
    if 'selected_batch_for_review' not in st.session_state:
        st.session_state.selected_batch_for_review = None


    api_keys = check_api_keys()
    display_sidebar(api_keys)
    
    display_header()
    
    if st.session_state.run_validation:
        uploaded_files = st.session_state.get('uploaded_files', [])
        if not uploaded_files:
            st.toast("📂 Please upload artwork files first!", icon="⚠️")
        else:
            files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

            if files_to_process:
                file_tuples = tuple(sorted(({"name": f["name"], "bytes": tuple(f["bytes"])} for f in files_to_process), key=lambda x: x['name']))
                
                analysis_results = run_analysis_cached(
                    file_tuples, 
                    st.session_state.get('must_contain', ''), 
                    st.session_state.get('must_not_contain', ''), 
                    st.session_state.get('custom_instructions', ''), 
                    st.session_state.get('ai_provider', 'openai'),
                    st.session_state.get('primary_validation_text', '')
                )

                batch_key = analysis_results.get("skus", [])[0] if analysis_results.get("skus") else f"Batch-{int(time.time())}"
                
                st.session_state.current_batch_sku = batch_key
                st.session_state.selected_batch_for_review = batch_key

                st.session_state.batches[batch_key] = {
                    "uploaded_files_data": [dict(f, bytes=tuple(f['bytes'])) for f in files_to_process],
                    "brand": st.session_state.brand_selection,
                    **analysis_results
                }
                st.session_state.messages = []
                st.session_state.run_validation = False
                st.toast('✅ Analysis complete! Click the "AI Analysis Report" tab to view the results.', icon="🎉")
                st.rerun()

    display_main_interface()

if __name__ == "__main__":
    main()
