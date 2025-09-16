import streamlit as st
import time
from config import AppConfig
from file_processor import process_files_cached # UPGRADE: Import the cached function
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

# UPGRADE: Cache the entire AI analysis and validation process.
@st.cache_data
def run_analysis_cached(_processor_results, checklist_text, custom_instructions):
    """
    Performs all validation and AI analysis. Caching this function prevents
    re-running AI calls for the same data.
    """
    processed_docs, skus = _processor_results
    all_text = "\n\n".join(doc['text'] for doc in processed_docs)
    
    # Run rule-based validation
    validator = ArtworkValidator(all_text, checklist_text)
    global_results, per_doc_results = validator.validate(processed_docs)
    
    # Run AI analysis
    api_keys = check_api_keys()
    if not api_keys.get('openai'):
        return {
            "error": "OpenAI API key not found.",
            "global_results": global_results,
            "per_doc_results": per_doc_results,
            "skus": skus
        }

    reviewer = AIReviewer(api_keys)
    for doc in processed_docs:
        doc['text'] = reviewer.run_ai_ocr_correction(doc['text'])
        
    packaging_doc_text = next((d['text'] for d in processed_docs if d['doc_type'] == 'packaging_artwork'), None)
    ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text) if packaging_doc_text else {"success": True, "data": {}}
    compliance_results = reviewer.run_ai_compliance_check(ai_facts.get('data', {}), checklist_text)
    quality_results = reviewer.run_ai_quality_check(all_text)
    ai_summary = reviewer.generate_executive_summary(processed_docs, global_results, compliance_results.get('data', []), quality_results, custom_instructions)
    
    return {
        "global_results": global_results,
        "per_doc_results": per_doc_results,
        "skus": skus,
        "ai_summary": ai_summary,
        "ai_facts": ai_facts,
        "compliance_results": compliance_results,
        "quality_results": quality_results,
        "processed_docs": processed_docs,
        "ai_processing_complete": True
    }


def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    
    # Simplified session state initialization
    if 'current_batch_sku' not in st.session_state:
        st.session_state.current_batch_sku = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    display_header()
    display_instructions()

    api_keys = check_api_keys()
    run_validation, checklist_text, custom_instructions, run_test_validation = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    if run_validation and not uploaded_files:
        st.toast("📂 Please upload artwork files first!", icon="⚠️")

    elif run_validation or run_test_validation:
        files_to_process = []
        if run_test_validation:
            # For test validation, we can use a hardcoded batch key for caching
            st.session_state.current_batch_sku = "test_validation_batch"
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
            # For user uploads, clear the test batch key
            if st.session_state.current_batch_sku == "test_validation_batch":
                st.session_state.current_batch_sku = None
            files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

        if files_to_process:
            with st.spinner("Processing files... (This may take a moment on first run)"):
                # Use the cached file processor
                processor_results = process_files_cached(files_to_process)
            
            with st.spinner("Running validation and AI analysis..."):
                # Use the cached analysis function
                analysis_results = run_analysis_cached(processor_results, checklist_text, custom_instructions)

            # Determine the batch SKU and store results
            batch_sku = analysis_results["skus"][0] if analysis_results.get("skus") else f"Batch-{int(time.time())}"
            st.session_state.current_batch_sku = batch_sku
            
            # Use a dictionary to hold all batch data for cleaner session state
            if 'batches' not in st.session_state: st.session_state.batches = {}
            st.session_state.batches[batch_sku] = {
                "uploaded_files_data": files_to_process,
                **analysis_results
            }

    # Display logic remains the same, pulling data from the current batch in session state
    current_batch_data = st.session_state.batches.get(st.session_state.current_batch_sku)
    
    if current_batch_data:
        display_results_page(current_batch_data)
        display_dynamic_checklist(current_batch_data.get('checklist_text', ''), st.session_state.current_batch_sku, current_batch_data)
        display_pdf_previews(current_batch_data.get('uploaded_files_data', []))
    else:
        st.info("👋 Welcome to the Co-pilot! Start by pasting a checklist in the sidebar or uploading artwork files.")

    st.divider()
    display_chat_interface(current_batch_data)

if __name__ == "__main__":
    main()
