import streamlit as st
import os
from config import Config, load_css
from file_processor import FileProcessor
from ai_analyzer import AIAnalyzer

# --- Setup ---
st.set_page_config(page_title=Config.PAGE_TITLE, page_icon=Config.PAGE_ICON, layout=Config.LAYOUT)
load_css()

# --- Helper: Common Formulas Database ---
BASIC_FORMULAS = {
    "Price Elasticity of Demand": r"E_d = \frac{\% \Delta Q_d}{\% \Delta P}",
    "GDP (Expenditure Approach)": r"Y = C + I + G + (X - M)",
    "Profit Maximization": r"MR = MC",
    "Average Total Cost": r"ATC = \frac{TC}{Q} = AFC + AVC"
}

ADVANCED_FORMULAS = {
    "Cobb-Douglas Utility": r"U(x,y) = x^\alpha y^\beta",
    "MRS (Marginal Rate of Substitution)": r"MRS_{xy} = \frac{MU_x}{MU_y}",
    "Solow Steady State": r"s f(k^*) = (n + g + \delta)k^*",
    "Keynesian Multiplier": r"k = \frac{1}{1 - MPC}"
}

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=50) # Graph icon
    st.title("EconWiz Settings")
    
    # API Key
    if "OPENAI_API_KEY" in st.secrets:
        # We use the variable name OPENAI_API_KEY for convenience if you have it set, 
        # but this code uses Google Gemini. Ensure your secrets.toml has the right key or paste it below.
        st.success("API Key Found")
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    st.divider()
    
    # Context Settings
    st.header("📚 Context")
    level = st.select_slider("Academic Level", options=["High School", "Undergraduate (Intro)", "Undergraduate (Advanced)", "Graduate/PhD"])
    topic = st.selectbox("Topic", [
        "Microeconomics: Supply & Demand",
        "Microeconomics: Consumer Theory",
        "Microeconomics: Market Structures",
        "Macroeconomics: GDP & Inflation",
        "Macroeconomics: Fiscal/Monetary Policy",
        "Game Theory",
        "Econometrics",
        "International Trade"
    ])
    
    st.divider()
    
    # Formula Cheat Sheet in Sidebar
    st.subheader("🧮 Quick Formulas")
    formulas = ADVANCED_FORMULAS if "Advanced" in level or "Graduate" in level else BASIC_FORMULAS
    for name, latex in formulas.items():
        with st.expander(name):
            st.latex(latex)

# --- Main Content ---
st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
st.write("Upload a screenshot of a graph, a PDF of a problem set, or type your question below.")

# 1. Inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Problem")
    # File Uploader
    uploaded_file = st.file_uploader("Upload Graph/Homework (Optional)", type=Config.ALLOWED_EXTENSIONS)
    
    # Text Input
    user_query = st.text_area("Type your question here:", height=150, placeholder="Example: Calculate the deadweight loss from a tax of $5 given the supply function Qs = 2P...")
    
    run_btn = st.button("🚀 Solve / Explain", type="primary")

# 2. Processing & Results
with col2:
    st.subheader("2. Solution")
    
    if run_btn and api_key and (user_query or uploaded_file):
        with st.spinner(f"Analyzing {topic} problem..."):
            
            # A. Process File (if any)
            text_from_file = ""
            img_parts = []
            preview_img = None
            
            if uploaded_file:
                processor = FileProcessor()
                # We reuse the existing processor
                text_from_file, img_parts, preview_img = processor.process_file(uploaded_file)
                
                # Show preview
                if preview_img:
                    st.image(preview_img, caption="Uploaded Problem", width=300)
            
            # Combine context
            full_query = user_query
            if text_from_file:
                full_query += f"\n\n[CONTEXT FROM FILE]:\n{text_from_file[:2000]}"
            
            # B. Call AI
            analyzer = AIAnalyzer(api_key, Config.MODEL_NAME)
            response_text = analyzer.analyze_problem(img_parts, full_query, topic, level)
            
            # C. Display Output
            st.markdown("### Answer")
            st.markdown(response_text)
            
    elif run_btn and not api_key:
        st.error("Please provide an API Key in the sidebar to proceed.")
    elif run_btn:
        st.warning("Please enter a question or upload a file.")

# --- Footer ---
st.divider()
st.caption("EconWiz uses AI to assist with homework. Always verify calculations manually.")
