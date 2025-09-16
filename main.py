import streamlit as st
import time
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

@st.cache_data
def run_analysis_cached(_processor_results, checklist_text, custom_instructions):
    """
    Performs all validation and AI analysis. Caching this function prevents
    re-running AI calls for the same data. The underscore on _processor_results
    tells Streamlit to not hash this complex object, relying on the caller's caching.
    """
    processed_docs, skus = _processor_results
    all_text = "\n\n".join(doc['text'] for doc in processed_docs)
    
    validator = ArtworkValidator(all_text, checklist_text)
    global_results, per_doc_results = validator.validate(processed_docs)
    
    api_keys = check_api_keys()
    if not api_keys.get('openai'):
        return {
            "error": "OpenAI API key not found.", "global_results": global_results,
            "per_doc_results": per_doc_results, "skus": skus, "ai_processing_complete": True
        }

    reviewer = AIReviewer(api_keys)
    # The OCR correction is now part of the cached analysis
    for doc in processed_docs:
        doc['text'] = reviewer.run_ai_ocr_correction(doc['text'])
        
    packaging_doc_text = next((d['text'] for d in processed_docs if d['doc_type'] == 'packaging_artwork'), None)
    
    # Run all AI analyses
    ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text) if packaging_doc_text else {"success": True, "data": {}}
    compliance_results = reviewer.run_ai_compliance_check(ai_facts.get('data', {}), checklist_text)
    quality_results = reviewer.run_ai_quality_check(all_text)
    ai_summary = reviewer.generate_executive_summary(processed_docs, global_results, compliance_results.get('data', []), quality_results, custom_instructions)
    
    return {
        "global_results": global_results, "per_doc_results": per_doc_results, "skus": skus,
        "ai_summary": ai_summary, "ai_facts": ai_facts, "compliance_results": compliance_results,
        "quality_results": quality_results, "processed_docs": processed_docs, "ai_processing_complete": True
    }

def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    
    # Initialize session state with the default checklist
    if 'checklist_text_input' not in st.session_state:
        st.session_state.checklist_text_input = AppConfig.DEFAULT_CHECKLIST
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    current_batch_data = st.session_state.batches.get(st.session_state.get('current_batch_sku')) if 'batches' in st.session_state else None

    display_header()
    display_instructions()

    api_keys = check_api_keys()
    run_validation, checklist_text, custom_instructions, run_test_validation = display_sidebar(api_keys, current_batch_data)
    uploaded_files = display_file_uploader()

    if run_validation and not uploaded_files:
        st.toast("📂 Please upload artwork files first!", icon="⚠️")
    
    elif run_validation or run_test_validation:
        files_to_process = []
        if run_test_validation:
            test_files = [
                "Wheelchair Bag Advanced 020625.xlsx - Black.csv", "Wheelchair_Bag_Black_Shipping_Mark.pdf",
                "wheelchair_bag_advanced_purple_floral_240625.pdf", "wheelchair_bag_advanced_quickstart_020625.pdf",
                "wheelchair_bag_purple_flower_shipping_mark.pdf", "wheelchair_bag_tag_black_250625.pdf",
                "wheelchair_bag_tag_purple_250625.pdf", "wheelchair_bag_washtag.pdf"
            ]
            for file_path in test_files:
                try:
                    with open(file_path, "rb") as f: files_to_process.append({"name": file_path, "bytes": f.read()})
                except FileNotFoundError: st.error(f"Test file not found: {file_path}")
        else:
            files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

        if files_to_process:
            with st.spinner("Processing files... (This may take a moment on first run)"):
                processor_results = process_files_cached(files_to_process)
            
            with st.spinner("Running validation and AI analysis..."):
                analysis_results = run_analysis_cached(processor_results, checklist_text, custom_instructions)

            batch_sku = analysis_results["skus"][0] if analysis_results.get("skus") else f"Batch-{int(time.time())}"
            st.session_state.current_batch_sku = batch_sku
            
            if 'batches' not in st.session_state: st.session_state.batches = {}
            st.session_state.batches[batch_sku] = {
                "uploaded_files_data": [dict(f, bytes=tuple(f['bytes'])) for f in files_to_process], # Make bytes hashable for state
                "checklist_text": checklist_text,
                "custom_instructions": custom_instructions,
                **analysis_results
            }
            st.rerun()

    if current_batch_data:
        display_results_page(current_batch_data)
        display_dynamic_checklist(current_batch_data.get('checklist_text', ''), st.session_state.current_batch_sku, current_batch_data)
        display_pdf_previews(current_batch_data.get('uploaded_files_data', []))
    elif st.session_state.checklist_text_input.strip():
        display_dynamic_checklist(st.session_state.checklist_text_input, "standalone")
    else:
        st.info("👋 Welcome to the Co-pilot! Start by editing the checklist or uploading artwork files.")

    st.divider()
    display_chat_interface(current_batch_data)

if __name__ == "__main__":
    main()
