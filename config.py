import os
import streamlit as st

class Config:
    PAGE_TITLE = "Artwork Verification Pro"
    PAGE_ICON = "🎨"
    LAYOUT = "wide"
    
    # AI Configuration
    # Using the Flash model for speed and vision capabilities
    MODEL_NAME = "gemini-2.0-flash-exp" 
    
    # File Upload Settings
    ALLOWED_EXTENSIONS = ["pdf", "jpg", "jpeg", "png", "csv"]
    
    # System Prompts
    # We inject the "Persona" here to ensure the AI acts like a QC Engineer
    SYSTEM_PROMPT = """
    You are an expert Quality Control Specialist for medical device packaging (Vive Health/Coretech). 
    Your goal is to catch errors before production. 
    
    CRITICAL RULES:
    1. ACCURACY: Do not hallucinate text. If you can't read it, say "Unreadable".
    2. CONTEXT: Compare the visual design against the text content.
    3. SENSITIVITY: Be highly critical of "Made in China" placement, Barcode readability, and Spelling.
    4. BRANDING: Ensure fonts and logos match the brand style (Vive vs Coretech).
    """
    
    # Paths to reference files (assuming they are in the root or a data folder)
    VIVE_CHECKLIST_PATH = "Artwork Checklist.xlsx - Vive.csv"
    CORETECH_CHECKLIST_PATH = "Artwork Checklist.xlsx - Coretech.csv"
    ERROR_TRACKER_PATH = "Artwork Error Tracker (1).xlsx - Sheet1.csv"

def load_css():
    st.markdown("""
        <style>
        .stAlert { padding: 10px; border-radius: 5px; }
        .pass-badge { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .fail-badge { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        .warning-badge { background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
