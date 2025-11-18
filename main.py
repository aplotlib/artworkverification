import streamlit as st
import os
from config import Config, load_css
from file_processor import FileProcessor
from checklist_manager import ChecklistManager
from validator import ArtworkValidator
from ai_analyzer import AIAnalyzer

# --- Setup ---
st.set_page_config(page_title=Config.PAGE_TITLE, page_icon=Config.PAGE_ICON, layout=Config.LAYOUT)
load_css()

# Initialize Session State
if 'validation_results' not in st.session_state:
    st.session_state['validation_results'] = None

# Sidebar - Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=50)
    st.title("Artwork Pro")
    
    # Check for API Key in secrets
    if "OPENAI_API_KEY" in st.secrets:
        st.success("OpenAI API Key Connected")
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("Missing 'OPENAI_API_KEY' in secrets.toml")
        api_key = None
    
    st.divider()
    
    brand_choice = st.selectbox("Select Brand Checklist", ["Vive Health", "Coretech"])
    
    st.divider()
    st.info("System ready. Upload artwork to begin validation.")

# --- Main Content ---
st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
st.write("Automated verification against Brand Checklists and Historical Error Tracker.")

# 1. Upload Section
uploaded_file = st.file_uploader("Upload Artwork (PDF/Image)", type=Config.ALLOWED_EXTENSIONS)

if uploaded_file and api_key:
    col1, col2 = st.columns([1, 1])
    
    # 2. Processing
    processor = FileProcessor()
    text, img_parts, preview_image = processor.process_file(uploaded_file)
    
    with col1:
        st.subheader("Artwork Preview")
        if preview_image:
            st.image(preview_image, use_column_width=True)
        
        with st.expander("View Extracted Text"):
            st.text(text[:1000] + "...")

    # 3. Validation Logic
    with col2:
        st.subheader("Analysis & Validation")
        
        if st.button("Run Verification"):
            with st.spinner("Consulting Knowledge Base & Analyzing Visuals (GPT-4o)..."):
                
                # A. Load Context
                checklist_mgr = ChecklistManager()
                checklist_path = Config.VIVE_CHECKLIST_PATH if brand_choice == "Vive Health" else Config.CORETECH_CHECKLIST_PATH
                
                rules = checklist_mgr.load_checklist(checklist_path, brand_choice)
                errors = checklist_mgr.get_common_errors(Config.ERROR_TRACKER_PATH)
                
                # B. AI Analysis
                analyzer = AIAnalyzer(api_key, Config.MODEL_NAME)
                checklist_str = "\n".join([r['requirement'] for r in rules[:10]]) 
                
                # Pass text context + visual parts
                ai_results = analyzer.analyze_artwork(img_parts, checklist_str, errors)
                
                # C. Logic Validation
                validator = ArtworkValidator(rules, errors)
                final_report = validator.run_validation(text, uploaded_file.name, ai_results)
                
                st.session_state['validation_results'] = final_report

        # 4. Display Results
        if st.session_state['validation_results']:
            res = st.session_state['validation_results']
            
            # Scorecards
            s1, s2, s3 = st.columns(3)
            s1.metric("PASS", res['summary']['pass'])
            s2.metric("FAIL", res['summary']['fail'])
            s3.metric("WARNING", res['summary']['warning'])
            
            st.divider()
            
            # Detailed List
            for item in res['details']:
                icon = "✅" if item['status'] == "PASS" else "❌" if item['status'] == "FAIL" else "⚠️"
                with st.expander(f"{icon} {item['check']} ({item['category']})"):
                    st.write(f"**Status:** {item['status']}")
                    st.write(f"**Observation:** {item['observation']}")

elif not api_key:
    st.warning("Please add your OpenAI API Key to .streamlit/secrets.toml to proceed.")
