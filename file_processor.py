import streamlit as st
import time
import os
from config import AppConfig
from file_processor import process_files_cached
from validator import ArtworkValidator
from ai_analyzer import AIReviewer, check_api_keys
from ui_components import (
    display_header,
    display_instructions,
    display_sidebar,
    display_file_uploader,
    display_results_page,
    display_pdf_previews,
    display_dynamic_checklist,
    display_chat_interface
)

@st.cache_data(show_spinner=False) # Spinner is handled manually
def run_analysis_cached(_file_tuples, must_contain_text, must_not_contain_text, custom_instructions, ai_provider):
    """
    Performs all validation and AI analysis. Caching this function prevents
    re-running AI calls for the same data.
    """
    with st.spinner("Step 1/2: Processing and extracting text from files..."):
        processed_docs, skus = process_files_cached(_file_tuples)
    
    with st.spinner("Step 2/2: Running AI analysis... This may take a moment."):
        all_text = "\n\n".join(doc['text'] for doc in processed_docs)
        
        validator = ArtworkValidator(all_text)
        global_results, per_doc_results = validator.validate(processed_docs)
        
        api_keys = check_api_keys()
        if not api_keys.get(ai_provider):
            return {
                "error": f"{ai_provider.capitalize()} API key not found.", "global_results": global_results,
                "per_doc_results": per_doc_results, "skus": skus, "processed_docs": processed_docs
            }

        reviewer = AIReviewer(api_keys, provider=ai_provider)
        # AI OCR Correction is now part of the cached file processing
        
        packaging_doc_text = next((d['text'] for d in processed_docs if d['doc_type'] == 'packaging_artwork'), all_text)
        
        ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text)
        compliance_results = reviewer.run_ai_compliance_check(ai_facts.get('data', {}), must_contain_text, must_not_contain_text)
        quality_results = reviewer.run_ai_quality_check(all_text)
        ai_summary = reviewer.generate_executive_summary(processed_docs, global_results, compliance_results.get('data', []), quality_results, custom_instructions)
    
    return {
        "global_results": global_results, "per_doc_results": per_doc_results, "skus": skus,
        "ai_summary": ai_summary, "ai_facts": ai_facts, "compliance_results": compliance_results,
        "quality_results": quality_results, "processed_docs": processed_docs
    }

def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    
    # Initialize session state for robust multi-batch management
    if 'batches' not in st.session_state: st.session_state.batches = {}
    if 'current_batch_sku' not in st.session_state: st.session_state.current_batch_sku = None
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'brand_selection' not in st.session_state: st.session_state.brand_selection = "Vive"

    api_keys = check_api_keys()
    run_validation, brand_selection, must_contain_text, must_not_contain_text, ai_provider, run_test_validation = display_sidebar(api_keys)
    
    display_header()
    
    # --- Main Logic for Processing and Analysis ---
    if run_validation or run_test_validation:
        files_to_process = []
        if run_test_validation:
            test_data_dir = 'test_data'
            if not os.path.exists(test_data_dir):
                st.error(f"Test data directory not found. Please create a '{test_data_dir}' folder and add test files.")
            else:
                test_files = os.listdir(test_data_dir)
                for file_name in test_files:
                    file_path = os.path.join(test_data_dir, file_name)
                    try:
                        with open(file_path, "rb") as f: files_to_process.append({"name": file_name, "bytes": f.read()})
                    except Exception as e:
                        st.error(f"Could not read test file {file_path}: {e}")
        else:
            uploaded_files = st.session_state.get('uploaded_files', [])
            if not uploaded_files:
                st.toast("📂 Please upload artwork files first!", icon="⚠️")
            else:
                files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

        if files_to_process:
            file_tuples = tuple(sorted(({"name": f["name"], "bytes": tuple(f["bytes"])} for f in files_to_process), key=lambda x: x['name']))
            
            analysis_results = run_analysis_cached(file_tuples, must_contain_text, must_not_contain_text, "", ai_provider)

            batch_key = analysis_results.get("skus", [])[0] if analysis_results.get("skus") else f"Batch-{int(time.time())}"
            st.session_state.current_batch_sku = batch_key
            
            st.session_state.batches[batch_key] = {
                "uploaded_files_data": [dict(f, bytes=tuple(f['bytes'])) for f in files_to_process],
                "brand": brand_selection,
                **analysis_results
            }
            st.session_state.messages = []
            st.rerun()

    # --- Display Area ---
    main_display_area = st.container()
    
    with main_display_area:
        current_batch_sku = st.session_state.get('current_batch_sku')
        current_batch_data = st.session_state.batches.get(current_batch_sku)

        if current_batch_data:
            display_results_page(current_batch_data)
            col1, col2 = st.columns(2)
            with col1:
                display_dynamic_checklist(current_batch_data.get('brand', 'Vive'), current_batch_sku)
            with col2:
                display_pdf_previews(current_batch_data.get('uploaded_files_data', []))
            display_chat_interface(current_batch_data)
        else:
            display_instructions()
            st.session_state['uploaded_files'] = display_file_uploader()
            display_dynamic_checklist(st.session_state.brand_selection, "standalone_checklist")
            
if __name__ == "__main__":
    main()
