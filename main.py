import streamlit as st
import time
from config import AppConfig
from file_processor import DocumentProcessor
from validator import ArtworkValidator
from ai_analyzer import AIReviewer, check_api_keys
from ui_components import (
    display_header,
    display_instructions,
    display_sidebar,
    display_file_uploader,
    display_results_page,
    display_pdf_previews,
    display_chat_interface
)

# --- Security: Rate Limiting Constants ---
VERIFICATION_COOLDOWN_SECONDS = 10

def initialize_session_state():
    """Initializes the session state variables."""
    st.session_state.setdefault('validation_complete', False)
    st.session_state.setdefault('global_results', [])
    st.session_state.setdefault('per_doc_results', {})
    st.session_state.setdefault('skus', [])
    st.session_state.setdefault('ai_summary', "")
    st.session_state.setdefault('ai_facts', {})
    st.session_state.setdefault('compliance_results', [])
    st.session_state.setdefault('quality_results', {})
    st.session_state.setdefault('processed_docs', [])
    st.session_state.setdefault('uploaded_files_data', [])
    st.session_state.setdefault('messages', []) 
    st.session_state.setdefault('last_verification_time', 0)

def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    initialize_session_state()

    if st.session_state.get('run_ai_processing'):
        st.session_state.run_ai_processing = False
        api_keys = check_api_keys()
        reviewer = AIReviewer(api_keys)
        
        with st.spinner("AI is correcting OCR text..."):
            for doc in st.session_state.processed_docs:
                doc['text'] = reviewer.run_ai_ocr_correction(doc['text'])
        
        all_text = "\n\n".join(doc['text'] for doc in st.session_state.processed_docs)

        with st.spinner("AI is analyzing... Step 1/3: Extracting key facts..."):
            packaging_doc_text = next((d['text'] for d in st.session_state.processed_docs if d['doc_type'] == 'packaging_artwork'), None)
            if packaging_doc_text:
                st.session_state.ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text)
        
        with st.spinner("AI is analyzing... Step 2/3: Running compliance checks..."):
            reference_text = st.session_state.get('reference_text', '')
            st.session_state.compliance_results = reviewer.run_ai_compliance_check(st.session_state.ai_facts, reference_text)
        
        with st.spinner("AI is analyzing... Step 3/3: Proofreading for quality..."):
            st.session_state.quality_results = reviewer.run_ai_quality_check(all_text)

        with st.spinner("AI is generating executive summary..."):
            custom_instructions = st.session_state.get('custom_instructions', '')
            st.session_state.ai_summary = reviewer.generate_executive_summary(
                st.session_state.processed_docs, st.session_state.global_results, 
                st.session_state.compliance_results, st.session_state.quality_results, custom_instructions
            )
        
        st.session_state.ai_processing_complete = True
        st.rerun()

    display_header()
    display_instructions()
    
    api_keys = check_api_keys()
    run_validation, custom_instructions, reference_text, run_test_validation = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    if run_validation or run_test_validation:
        current_time = time.time()
        if current_time - st.session_state.last_verification_time < VERIFICATION_COOLDOWN_SECONDS:
            st.toast(f"Please wait {VERIFICATION_COOLDOWN_SECONDS} seconds before running another verification.", icon="⏳")
        else:
            st.session_state.last_verification_time = current_time
            
            files_to_process = []
            if run_test_validation:
                test_files = [
                    "Wheelchair Bag Advanced 020625.xlsx - Black.csv",
                    "Copy of Wheelchair_Bag_Black_Shipping_Mark.pdf",
                    "Copy of wheelchair_bag_advanced_purple_floral_240625.pdf",
                    "Copy of wheelchair_bag_advanced_quickstart_020625.pdf",
                    "Copy of wheelchair_bag_purple_flower_shipping_mark.pdf",
                    "Copy of wheelchair_bag_tag_black_250625.pdf",
                    "wheelchair_bag_tag_purple_250625.pdf",
                    "wheelchair_bag_washtag.pdf"
                ]
                for file_path in test_files:
                    try:
                        with open(file_path, "rb") as f:
                            files_to_process.append({"name": file_path, "bytes": f.read()})
                    except FileNotFoundError:
                        st.error(f"Test file not found: {file_path}")
            elif uploaded_files:
                files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

            if not files_to_process:
                st.session_state.validation_complete = True
                st.session_state.processed_docs = []
            else:
                st.session_state.uploaded_files_data = files_to_process
                st.session_state.custom_instructions = custom_instructions
                st.session_state.reference_text = reference_text

                with st.spinner("Extracting text and running rule-based checks..."):
                    processor = DocumentProcessor(st.session_state.uploaded_files_data)
                    st.session_state.processed_docs, st.session_state.skus = processor.process_files()
                    all_text = "\n\n".join(doc['text'] for doc in st.session_state.processed_docs)
                    validator = ArtworkValidator(all_text)
                    st.session_state.global_results, st.session_state.per_doc_results = validator.validate(st.session_state.processed_docs)
                
                st.session_state.validation_complete = True
                st.session_state.run_ai_processing = True
            
            st.rerun()

    if st.session_state.validation_complete:
        display_results_page(**st.session_state)
        display_pdf_previews(st.session_state.uploaded_files_data)

    st.divider()
    display_chat_interface()

if __name__ == "__main__":
    main()
