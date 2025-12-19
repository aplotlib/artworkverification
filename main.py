import streamlit as st
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
    "MRS": r"MRS_{xy} = \frac{MU_x}{MU_y}",
    "Solow Steady State": r"s f(k^*) = (n + g + \delta)k^*",
    "Keynesian Multiplier": r"k = \frac{1}{1 - MPC}"
}

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666505.png", width=50)
    st.title("EconWiz Settings")
    
    # --- API KEY LOGIC ---
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success(f"Key Loaded: ...{api_key[-4:]}")
    else:
        st.error("Missing 'OPENAI_API_KEY' in secrets.toml")
        api_key = st.text_input("Or enter key manually:", type="password")
    # ---------------------
    
    st.divider()
    
    st.header("📚 Context")
    level = st.select_slider("Level", options=["High School", "Undergrad (Intro)", "Undergrad (Adv)", "Grad/PhD"])
    topic = st.selectbox("Topic", [
        "Micro: Supply & Demand",
        "Micro: Consumer Theory",
        "Macro: GDP & Inflation",
        "Macro: Fiscal/Monetary Policy",
        "Game Theory",
        "Econometrics"
    ])
    
    st.divider()
    st.subheader("🧮 Quick Formulas")
    formulas = ADVANCED_FORMULAS if "Adv" in level or "Grad" in level else BASIC_FORMULAS
    for name, latex in formulas.items():
        with st.expander(name):
            st.latex(latex)

# --- Main Content ---
st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
st.write("Upload a graph or homework problem to get started.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input")
    uploaded_file = st.file_uploader("Upload File", type=Config.ALLOWED_EXTENSIONS)
    user_query = st.text_area("Question", height=150, placeholder="e.g., Calculate equilibrium price...")
    run_btn = st.button("🚀 Solve", type="primary")

with col2:
    st.subheader("2. Solution")
    
    if run_btn:
        if not api_key:
            st.error("Please configure your OpenAI API Key.")
        elif not (user_query or uploaded_file):
            st.warning("Please provide a file or a question.")
        else:
            with st.spinner("Analyzing with GPT-4o..."):
                # Process File
                text_from_file = ""
                img_parts = []
                preview_img = None
                
                if uploaded_file:
                    processor = FileProcessor()
                    text_from_file, img_parts, preview_img = processor.process_file(uploaded_file)
                    if preview_img:
                        st.image(preview_img, caption="Preview", width=300)
                
                # Combine Query
                full_query = user_query
                if text_from_file:
                    full_query += f"\n\n[FILE CONTENT]:\n{text_from_file[:2000]}"
                
                # Run AI
                analyzer = AIAnalyzer(api_key, Config.MODEL_NAME)
                result = analyzer.analyze_problem(img_parts, full_query, topic, level)
                
                st.markdown("### Answer")
                st.markdown(result)
