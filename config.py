import streamlit as st

class Config:
    PAGE_TITLE = "EconWiz Assistant"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # Using GPT-4o. If you don't have access, change to "gpt-4-turbo" or "gpt-3.5-turbo"
    MODEL_NAME = "gpt-4o" 
    
    ALLOWED_EXTENSIONS = ["pdf", "jpg", "jpeg", "png"]

def load_css():
    st.markdown("""
        <style>
        .stAlert { padding: 10px; border-radius: 5px; }
        </style>
    """, unsafe_allow_html=True)
