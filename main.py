import streamlit as st
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
    display_pdf_previews
)

def initialize_session_state():
    """Initializes the session state variables."""
    if "validation_complete" not in st.session_state:
        st.session_state.validation_complete = False
        st.session_state.global_results = []
        st.session_state.per_doc_results = {}
        st.session_state.skus = []
        st.session_state.ai_summary = ""
        st.session_state.ai_facts = {}
        st.session_state.processed_docs = []
        st.session_state.uploaded_files_data = []

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    initialize_session_state()

    # --- AI Processing (runs on rerun after initial validation) ---
    if st.session_state.get('run_ai_processing'):
        st.session_state.run_ai_processing = False  # Prevent re-running
        api_keys = check_api_keys()
        reviewer = AIReviewer(api_keys)
        
        # 1. AI Fact Extraction on primary packaging
        packaging_doc_text = next((d['text'] for d in st.session_state.processed_docs if d['doc_type'] == 'packaging_artwork'), None)
        if packaging_doc_text:
            st.session_state.ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text)
        
        # 2. Final AI Summary
        custom_instructions = st.session_state.get('custom_instructions', '')
        st.session_state.ai_summary = reviewer.generate_summary(
            docs=st.session_state.processed_docs,
            custom_instructions=custom_instructions
        )
        st.session_state.ai_processing_complete = True
        st.rerun()

    # --- UI Rendering ---
    display_header()
    display_instructions()
    
    api_keys = check_api_keys()
    run_validation, custom_instructions, reference_text = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    # --- Initial Validation Run ---
    if run_validation and uploaded_files:
        st.session_state.uploaded_files_data = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]
        st.session_state.custom_instructions = custom_instructions # Save for the AI run

        with st.spinner("Step 1/2: Analyzing documents and running checks..."):
            processor = DocumentProcessor(st.session_state.uploaded_files_data)
            st.session_state.processed_docs, st.session_state.skus = processor.process_files()
            
            all_text = "\n\n".join(doc['text'] for doc in st.session_state.processed_docs)
            validator = ArtworkValidator(all_text, reference_text=reference_text)
            st.session_state.global_results, st.session_state.per_doc_results = validator.validate(st.session_state.processed_docs)

        st.session_state.validation_complete = True
        st.session_state.run_ai_processing = True  # Flag to start AI processing on the next run
        st.rerun()

    # --- Display Results Page ---
    if st.session_state.validation_complete:
        display_results_page(
            st.session_state.global_results,
            st.session_state.per_doc_results,
            st.session_state.processed_docs,
            st.session_state.skus,
            st.session_state.ai_summary,
            st.session_state.ai_facts
        )
        display_pdf_previews(st.session_state.uploaded_files_data)

if __name__ == "__main__":
    main()
