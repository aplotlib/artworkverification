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
    st.session_state.setdefault('validation_complete', False)
    st.session_state.setdefault('global_results', [])
    st.session_state.setdefault('per_doc_results', {})
    st.session_state.setdefault('skus', [])
    st.session_state.setdefault('ai_summary', "")
    st.session_state.setdefault('ai_facts', {})
    st.session_state.setdefault('compliance_results', [])
    st.session_state.setdefault('processed_docs', [])
    st.session_state.setdefault('uploaded_files_data', [])

def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    initialize_session_state()

    if st.session_state.get('run_ai_processing'):
        st.session_state.run_ai_processing = False
        api_keys = check_api_keys()
        reviewer = AIReviewer(api_keys)
        
        with st.spinner("AI is analyzing... Step 1/3: Extracting key facts..."):
            packaging_doc_text = next((d['text'] for d in st.session_state.processed_docs if d['doc_type'] == 'packaging_artwork'), None)
            if packaging_doc_text:
                st.session_state.ai_facts = reviewer.run_ai_fact_extraction(packaging_doc_text)
        
        with st.spinner("AI is analyzing... Step 2/3: Running compliance checks..."):
            reference_text = st.session_state.get('reference_text', '')
            st.session_state.compliance_results = reviewer.run_ai_compliance_check(st.session_state.ai_facts, reference_text)
        
        with st.spinner("AI is analyzing... Step 3/3: Generating executive summary..."):
            custom_instructions = st.session_state.get('custom_instructions', '')
            st.session_state.ai_summary = reviewer.generate_executive_summary(
                st.session_state.processed_docs, st.session_state.global_results, 
                st.session_state.compliance_results, custom_instructions
            )
        
        st.session_state.ai_processing_complete = True
        st.rerun()

    display_header()
    display_instructions()
    
    api_keys = check_api_keys()
    run_validation, custom_instructions, reference_text = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    if run_validation and uploaded_files:
        st.session_state.uploaded_files_data = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]
        st.session_state.custom_instructions = custom_instructions
        st.session_state.reference_text = reference_text

        with st.spinner("Running rule-based checks..."):
            processor = DocumentProcessor(st.session_state.uploaded_files_data)
            st.session_state.processed_docs, st.session_state.skus = processor.process_files()
            all_text = "\n\n".join(doc['text'] for doc in st.session_state.processed_docs)
            validator = ArtworkValidator(all_text) # No longer needs reference_text
            st.session_state.global_results, st.session_state.per_doc_results = validator.validate(st.session_state.processed_docs)
        
        st.session_state.validation_complete = True
        st.session_state.run_ai_processing = True
        st.rerun()

    if st.session_state.validation_complete:
        display_results_page(**st.session_state)
        display_pdf_previews(st.session_state.uploaded_files_data)

if __name__ == "__main__":
    main()
