import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# fetch Backend URL from .env or use default (localhost)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Research Assistant AI",
    page_icon="🔬",
    layout="wide"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Dark mode premium styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1f6f8b, #134e5e);
        color: white;
        border: none;
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(31, 111, 139, 0.4);
        color: white;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #58a6ff !important;
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s;
        text-align: center;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    /* Hide default Streamlit elements if needed */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def call_research_api(query: str) -> dict:
    """
    FastAPI backend call
    Args:
        query: research question
    Returns:
        API response dict or None if error
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/research",
            json={"query": query},
            timeout=120
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            error = response.json().get("detail", "Bad request")
            st.error(f"Error: {error}")
            return None
        else:
            st.error(f"Server error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the backend server. Please check if it's running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None


def check_backend_health() -> bool:
    """check weather backend is working"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# ---------------- SideBar ----------------

with st.sidebar:
    st.header("Settings")

    st.subheader("Backend Status")
    if check_backend_health():
        st.success("Connected")
    else:
        st.error("Disconnected")
        st.write("Run this command to start the server:")
        st.code("uvicorn main:app --host 0.0.0.0 --port 8000 --reload")

    st.divider()

    st.subheader("Example Queries")
    example_queries = [
        "Compare transformer and LSTM for NLP",
        "RAG vs fine-tuning for LLMs",
        "Diffusion models vs GANs",
    ]

    for eq in example_queries:
        if st.button(eq, key=eq):
            st.session_state.selected_query = eq


# ---------------- Main Page ----------------

st.title("Multi-Agent Research Assistant")
st.write("Powered by ArXiv + RAG + LangGraph + Groq")

default_query = st.session_state.get("selected_query", "")

query = st.text_input(
    "Research Query",
    placeholder="e.g. Compare transformer and LSTM architectures for NLP tasks",
    value=default_query
)

search_clicked = st.button("Research")

if "selected_query" in st.session_state:
    del st.session_state.selected_query


# ---------------- Search Logic ----------------

if search_clicked and query:

    if not check_backend_health():
        st.error("Backend server is down. Please start the server first.")
        st.code("uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        st.stop()

    with st.spinner("Agents are researching..."):
        result = call_research_api(query)

    if result:
        final = result.get("final_answer", {})

        st.divider()
        st.header("Research Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Papers Analyzed", final.get("papers_analyzed", 0))
        with col2:
            st.metric("Quality Score", f"{final.get('quality_score', 0)}/10")
        with col3:
            st.metric("Iterations", final.get("iterations_taken", 0))
        with col4:
            st.metric("Time Taken", f"{result.get('time_taken', 0)}s")

        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Methodology", "Research Gaps", "Recommendations", "Sources"])

        with tab1:
            st.subheader("Key Findings")
            findings = final.get("key_findings", [])
            if findings:
                for i, finding in enumerate(findings):
                    st.write(f"{i+1}. {finding}")
            else:
                st.info("No findings available")

        with tab2:
            st.subheader("Methodology Comparison")
            methodology = final.get("methodology_comparison", "")
            if methodology:
                st.write(methodology)
            else:
                st.info("No methodology comparison available")

        with tab3:
            st.subheader("Research Gaps")
            gaps = final.get("research_gaps", [])
            if gaps:
                for gap in gaps:
                    st.write(f"- {gap}")
            else:
                st.info("No gaps identified")

        with tab4:
            st.subheader("Recommendations")
            recommendations = final.get("recommendations", [])
            if recommendations:
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.info("No recommendations available")

        with tab5:
            st.subheader("Source Papers")
            sources = final.get("sources", [])
            if sources:
                for i, source in enumerate(sources):
                    with st.expander(f"{i+1}. {source.get('title', 'Unknown')}"):
                        st.write(f"**Published:** {source.get('published', 'N/A')}")
                        authors = source.get('authors', [])
                        if authors:
                            st.write(f"**Authors:** {', '.join(authors[:3])}")
                        url = source.get('url', '')
                        if url:
                            st.write(f"[View on ArXiv]({url})")
            else:
                st.info("No sources available")

        st.divider()
        with st.expander("Raw JSON Response (Debug)"):
            st.json(result)

elif search_clicked and not query:
    st.warning("Please enter a research query first!")