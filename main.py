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
    display_dynamic_checklist,
    display_chat_interface
)

# --- Security: Rate Limiting Constants ---
VERIFICATION_COOLDOWN_SECONDS = 10

def initialize_session_state():
    """Initializes session state variables."""
    st.session_state.setdefault('batches', {})
    st.session_state.setdefault('current_batch_sku', None)
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('last_verification_time', 0)
    st.session_state.setdefault('checklist_state', {})
    st.session_state.setdefault('checklist_text_input', "")


def get_current_batch_data():
    """Safely retrieves data for the currently selected batch."""
    if st.session_state.current_batch_sku and st.session_state.current_batch_sku in st.session_state.batches:
        return st.session_state.batches[st.session_state.current_batch_sku]
    return None

def run_ai_processing(batch_data):
    """Runs the full AI analysis pipeline on a given batch."""
    batch_data['run_ai_processing'] = False
    api_keys = check_api_keys()
    if not api_keys.get('openai'):
        st.warning("Cannot run AI processing: OpenAI API key not found.")
        return

    reviewer = AIReviewer(api_keys)
    with st.spinner("AI is correcting OCR text..."):
        for doc in batch_data['processed_docs']:
            doc['text'] = reviewer.run_ai_ocr_correction(doc['text'])

    all_text = "\n\n".join(doc['text'] for doc in batch_data['processed_docs'])

    with st.spinner("AI is analyzing... Step 1/3: Extracting key facts..."):
        packaging_doc_text = next((d['text'] for d in batch_data['processed_docs'] if d['doc_type'] == 'packaging_artwork'), None)
        batch_data['ai_facts'] = reviewer.run_ai_fact_extraction(packaging_doc_text) if packaging_doc_text else {}

    with st.spinner("AI is analyzing... Step 2/3: Running compliance checks..."):
        batch_data['compliance_results'] = reviewer.run_ai_compliance_check(
            batch_data['ai_facts'], batch_data.get('checklist_text', '')
        )

    with st.spinner("AI is analyzing... Step 3/3: Proofreading for quality..."):
        batch_data['quality_results'] = reviewer.run_ai_quality_check(all_text)

    with st.spinner("AI is generating executive summary..."):
        batch_data['ai_summary'] = reviewer.generate_executive_summary(
            batch_data['processed_docs'], batch_data['global_results'],
            batch_data['compliance_results'], batch_data['quality_results'],
            batch_data.get('custom_instructions', '')
        )
    batch_data['ai_processing_complete'] = True


def main():
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.PAGE_ICON, layout="wide")
    initialize_session_state()

    current_batch_data = get_current_batch_data()
    if current_batch_data and current_batch_data.get('run_ai_processing'):
        run_ai_processing(current_batch_data)
        st.rerun()

    display_header()
    display_instructions()

    api_keys = check_api_keys()
    run_validation, checklist_text, custom_instructions, run_test_validation = display_sidebar(api_keys)
    uploaded_files = display_file_uploader()

    if run_validation or run_test_validation:
        current_time = time.time()
        if current_time - st.session_state.last_verification_time < VERIFICATION_COOLDOWN_SECONDS:
            st.toast(f"Please wait {VERIFICATION_COOLDOWN_SECONDS} seconds.", icon="⏳", duration=VERIFICATION_COOLDOWN_SECONDS * 1000)
        else:
            st.session_state.last_verification_time = current_time
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
                        with open(file_path, "rb") as f:
                            files_to_process.append({"name": file_path, "bytes": f.read()})
                    except FileNotFoundError:
                        st.error(f"Test file not found: {file_path}")
            elif uploaded_files:
                files_to_process = [{"name": f.name, "bytes": f.getvalue()} for f in uploaded_files]

            if files_to_process:
                with st.spinner("Processing files and running checks..."):
                    processor = DocumentProcessor(files_to_process)
                    processed_docs, skus = processor.process_files()
                    all_text = "\n\n".join(doc['text'] for doc in processed_docs)
                    validator = ArtworkValidator(all_text, checklist_text)
                    global_results, per_doc_results = validator.validate(processed_docs)

                    batch_sku = skus[0] if skus else f"Batch-{len(st.session_state.batches) + 1}"
                    st.session_state.current_batch_sku = batch_sku
                    st.session_state.batches[batch_sku] = {
                        'uploaded_files_data': files_to_process, 'processed_docs': processed_docs, 'skus': skus,
                        'global_results': global_results, 'per_doc_results': per_doc_results,
                        'checklist_text': checklist_text, 'custom_instructions': custom_instructions,
                        'run_ai_processing': True, 'ai_summary': "", 'ai_facts': {},
                        'compliance_results': [], 'quality_results': {},
                    }
                st.rerun()

    # --- Main Display Logic ---
    # This logic decides what to show on the main page.
    if current_batch_data:
        # If a batch is active, show the full report page and its checklist
        display_results_page(current_batch_data)
        display_dynamic_checklist(current_batch_data.get('checklist_text', ''), st.session_state.current_batch_sku)
        display_pdf_previews(current_batch_data.get('uploaded_files_data', []))
    elif checklist_text.strip():
        # If no batch is active but there is text in the checklist, show only the checklist
        display_dynamic_checklist(checklist_text, "standalone")
    else:
        st.info("👋 Welcome to VAVE! Start by pasting a checklist in the sidebar or uploading artwork files.")

    st.divider()
    display_chat_interface(current_batch_data)

if __name__ == "__main__":
    main()
