import streamlit as st
import base64
from typing import List, Dict, Any, Tuple

def display_header():
    """Displays the main application header."""
    st.markdown("<h1>Artwork Verification Tool</h1>", unsafe_allow_html=True)

def display_sidebar(api_keys: Dict[str, str]) -> Tuple[bool, str, str]:
    """Displays the sidebar controls and returns their current values."""
    with st.sidebar:
        st.header("⚙️ Controls")
        run_validation = st.button("🔍 Run Validation", type="primary")
        
        st.header("🤖 AI Review Configuration")
        custom_instructions = st.text_area("Custom Instructions for AI (Optional)", 
                                           help="Guide the AI's focus, e.g., 'Check for a 1-year warranty statement.'")
        
        # AI Provider Selection
        options = {"Select AI Provider...": None}
        if 'openai' in api_keys: options["OpenAI"] = 'openai'
        if 'anthropic' in api_keys: options["Anthropic"] = 'anthropic'
        if 'openai' in api_keys and 'anthropic' in api_keys: 
            options["Both (Anthropic -> OpenAI)"] = 'both'
        
        if len(options) > 1:
            selected = st.selectbox("Choose AI Provider", options=options.keys())
            ai_provider = options[selected]
        else:
            st.warning("No AI API keys found in secrets. AI review is disabled.")
            ai_provider = None
        
        if st.button("Clear & Reset"):
            st.session_state.clear()
            st.rerun()
            
    return run_validation, ai_provider, custom_instructions

def display_manual_review_section():
    """Displays the manual checklist for high-risk areas."""
    st.markdown("### Manually Review High-Risk Areas")
    st.info("""
    Based on common errors, please manually check these critical points:
    - **Country of Origin**: Ensure "Made in China" (or correct country) is present and accurate.
    - **Color Matching**: Confirm that colors on packaging match labels and specifications.
    - **UDI Formatting**: Verify that all UDIs are present, correct, and scannable.
    """)
    st.markdown("---")

def display_file_uploader() -> List[st.runtime.uploaded_file_manager.UploadedFile]:
    """Displays the file uploader and returns the list of uploaded files."""
    return st.file_uploader("Upload all artwork files for one product", 
                             type=['pdf', 'csv', 'xlsx'], 
                             accept_multiple_files=True)

def display_report(results: List[Tuple[str, str, str]], skus: List[str]):
    """Displays the main validation report."""
    st.header("📊 Automated Validation Report")
    if len(skus) > 1:
        st.warning(f"**Multiple Variants Detected**: SKUs `{', '.join(skus)}` were found. Please manually ensure differences are correctly reflected on labels.")

    with st.expander("UDI & Serial Number Analysis", expanded=True):
        st.info("Checks for matching UPCs (12-digit) and UDIs across all files.")
        if not results:
            st.markdown("- No specific issues found.")
        for status, msg, _ in results:
            icon = '✅' if status == 'passed' else '❌'
            st.markdown(f"{icon} {msg}")

def display_ai_review(summary: str):
    """Displays the AI-generated review."""
    if summary:
        st.markdown("---")
        st.header("🤖 AI-Powered Review")
        st.markdown(summary, unsafe_allow_html=True)

def display_pdf_previews(files: List[Dict[str, Any]]):
    """Displays uploaded PDFs using an iframe."""
    pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
    if pdf_files:
        st.header("📄 PDF Previews")
        for pdf_file in pdf_files:
            with st.expander(f"View: {pdf_file['name']}"):
                base64_pdf = base64.b64encode(pdf_file['bytes']).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

def display_extracted_text(docs: List[Dict[str, str]]):
    """Displays the combined raw text extracted from all files."""
    with st.expander("📄 View Combined Extracted Text"):
        all_text = "\n\n".join(d['text'] for d in docs)
        st.text_area("Combined Extracted Text", all_text, height=300, label_visibility="collapsed")
