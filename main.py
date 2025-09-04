import streamlit as st
from config import AppConfig
from file_processor import DocumentProcessor
from validator import ArtworkValidator
from ai_analyzer import AIReviewer, check_api_keys
from ui_components import (
    display_header,
    display_sidebar,
    display_file_uploader,
    display_dashboard,
    display_ai_review,
    display_pdf_previews
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
    api_keys = check_api_keys()
    run_validation, custom_instructions = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    if run_validation and uploaded_files:
        st.session_state.uploaded_files_data = [
            {"name": f.name, "bytes": f.getvalue()} for f in uploaded_files
        ]

        with st.spinner("Step 1/3: Analyzing and extracting text from documents..."):
            processor = DocumentProcessor(st.session_state.uploaded_files_data)
            processed_docs, skus = processor.process_files()
            st.session_state.processed_docs = processed_docs
            st.session_state.skus = skus
        
        with st.spinner("Step 2/3: Running rule-based validation..."):
            all_text = "\n\n".join(doc['text'] for doc in processed_docs)
            validator = ArtworkValidator(all_text)
            st.session_state.results = validator.validate()

        st.session_state.validation_complete = True
        
        # Run AI review if keys are available
        if api_keys.get('openai') and api_keys.get('anthropic'):
            with st.spinner("Step 3/3: Generating AI-powered summary..."):
                reviewer = AIReviewer(api_keys)
                st.session_state.ai_summary = reviewer.generate_summary(
                    docs=st.session_state.processed_docs,
                    custom_instructions=custom_instructions
                )
        else:
            st.session_state.ai_summary = "AI review skipped. Missing OpenAI or Anthropic API keys."
            
        st.rerun()

    # --- Display Results ---
    if st.session_state.validation_complete:
        display_dashboard(st.session_state.results, st.session_state.processed_docs, st.session_state.skus)
        display_ai_review(st.session_state.ai_summary)
        display_pdf_previews(st.session_state.uploaded_files_data)

if __name__ == "__main__":
    main()
