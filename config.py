import streamlit as st

class Config:
    PAGE_TITLE = "EconWiz: Homework Assistant"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # AI Configuration
    # Using the Flash model for speed and multimodal (image + text) capabilities
    MODEL_NAME = "gemini-2.0-flash-exp" 
    
    # File Upload Settings
    ALLOWED_EXTENSIONS = ["pdf", "jpg", "jpeg", "png"]
    
    # System Prompts
    # Updated Persona for Economics
    SYSTEM_PROMPT = """
    You are an expert Economics Professor and rigorous homework tutor. 
    Your goal is to help students understand concepts, solve problems, and interpret graphs.
    
    GUIDELINES:
    1. EXPLAIN YOUR WORK: Don't just give the answer; step through the logic (e.g., "First, we set Qd = Qs...").
    2. FORMULAS: Use LaTeX formatting for math (e.g., $E_d = \frac{\%\Delta Q}{\%\Delta P}$).
    3. GRAPHS: If an image is provided, carefully analyze axes, curves (Supply, Demand, MC, MR), and equilibrium points.
    4. TONE: Supportive, educational, and precise.
    5. LEVELS: Adjust complexity based on the user's selected level (High School vs. Graduate).
    """

def load_css():
    st.markdown("""
        <style>
        .stAlert { padding: 10px; border-radius: 5px; }
        .formula-box { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #2A9D8F; }
        </style>
    """, unsafe_allow_html=True)
