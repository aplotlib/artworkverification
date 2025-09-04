import streamlit as st
from io import BytesIO
from config import AppConfig
from file_processor import DocumentProcessor
from validator import ArtworkValidator
from ai_analyzer import AIReviewer, check_api_keys
from ui_components import (
    display_header,
    display_sidebar,
    display_manual_review_section,
    display_file_uploader,
    display_report,
    display_ai_review,
    display_pdf_previews,
    display_extracted_text,
)

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide"
    )

    # Initialize session state
    if "validation_complete" not in st.session_state:
        st.session_state.validation_complete = False
        st.session_state.results = []
        st.session_state.skus = []
        st.session_state.ai_summary = ""
        st.session_state.processed_docs = []
        st.session_state.uploaded_files_data = []

    display_header()

    # Sidebar controls
    api_keys = check_api_keys()
    run_validation, ai_provider, custom_instructions = display_sidebar(api_keys)

    # Main page content
    display_manual_review_section()
    uploaded_files = display_file_uploader()

    if run_validation and uploaded_files:
        # Store file bytes in session state to persist across reruns
        st.session_state.uploaded_files_data = [
            {"name": f.name, "bytes": f.getvalue()} for f in uploaded_files
        ]

        # --- Core Processing ---
        with st.spinner("Step 1/3: Analyzing all documents..."):
            processor = DocumentProcessor(st.session_state.uploaded_files_data)
            processed_docs, skus = processor.process_files()
            st.session_state.processed_docs = processed_docs
            st.session_state.skus = skus
        
        with st.spinner("Step 2/3: Running rule-based validation..."):
            all_text = "\n\n".join(doc['text'] for doc in processed_docs)
            validator = ArtworkValidator(all_text)
            st.session_state.results = validator.validate()

        st.session_state.validation_complete = True
        st.session_state.run_ai_review = True # Flag to run AI on the next pass
        st.rerun() # Rerun to separate AI processing from core logic

    # --- AI Review (runs on the rerun after validation) ---
    if st.session_state.get('run_ai_review'):
        st.session_state.run_ai_review = False # Prevent re-running
        if ai_provider and st.session_state.processed_docs:
            with st.spinner(f"Step 3/3: Sending documents to AI for review... (Provider: {ai_provider})"):
                reviewer = AIReviewer(api_keys)
                st.session_state.ai_summary = reviewer.generate_summary_in_batches(
                    provider=ai_provider,
                    docs=st.session_state.processed_docs,
                    custom_instructions=custom_instructions
                )
            st.rerun() # Rerun one last time to display the final report cleanly

    # --- Display Results ---
    if st.session_state.validation_complete:
        display_report(st.session_state.results, st.session_state.skus)
        display_ai_review(st.session_state.ai_summary)
        display_pdf_previews(st.session_state.uploaded_files_data)
        display_extracted_text(st.session_state.processed_docs)

if __name__ == "__main__":
    main()
