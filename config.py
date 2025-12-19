import streamlit as st

class Config:
    PAGE_TITLE = "EconWiz: Homework Assistant"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # AI Configuration
    # Using GPT-4o for strong reasoning and vision capabilities
    MODEL_NAME = "gpt-4o" 
    
    # File Upload Settings
    ALLOWED_EXTENSIONS = ["pdf", "jpg", "jpeg", "png"]
    
    # System Prompts - Optional, mostly handled in the analyzer now
    SYSTEM_PROMPT = """
    You are an expert Economics Professor.
    """

def load_css():
    st.markdown("""
        <style>
        .stAlert { padding: 10px; border-radius: 5px; }
        .formula-box { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #2A9D8F; }
        </style>
    """, unsafe_allow_html=True)
