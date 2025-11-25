import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Multi-Tool Chatbot Agent",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multi-Tool Chatbot Agent")
st.caption("AI agent with web search, Wikipedia, and ArXiv capabilities")

# =========================================================
# SESSION STATE
# =========================================================
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "tavily_key" not in st.session_state:
    st.session_state.tavily_key = ""

if "agent" not in st.session_state:
    st.session_state.agent = None

# =========================================================
# SIDEBAR: API KEYS
# =========================================================
with st.sidebar:
    st.header("🔑 API Keys")

    st.session_state.openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key,
    )

    st.session_state.tavily_key = st.text_input(
        "Tavily API Key (Tavily Search)",
        type="password",
        value=st.session_state.tavily_key,
    )

    # Status messages
    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")

    if st.session_state.tavily_key:
        st.success("✅ Tavily Connected")
    else:
        st.warning("⚠️ Tavily Not Connected")

    # Show available tools if both keys exist
    if st.session_state.openai_key and st.session_state.tavily_key:
        st.subheader("🛠️ Available Tools")
        st.write("✅ **Tavily Search** – Web search")
        st.write("✅ **Wikipedia** – Encyclopedia")
        st.write("✅ **ArXiv** – Research papers")

    # Button to clear keys
    if st.session_state.openai_key or st.session_state.tavily_key:
        if st.button("Change API Keys"):
            st.session_state.openai_key = ""
            st.session_state.tavily_key = ""
            st.session_state.agent = None
            st.rerun()

# =========================================================
# BUILD AGENT WHEN KEYS ARE PRESENT
# =========================================================
if st.session_state.openai_key and st.session_state.tavily_key and st.session_state.agent is None:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=st.session_state.openai_key,
    )

    # Tavily search tool
    search_tool = TavilySearchResults(
        max_results=3,
        api_key=st.session_state.tavily_key,
    )

    # Wikipedia tool
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500,
        ),
        name="wikipedia",
        description=(
            "Search Wikipedia for encyclopedia articles, historical information, "
            "biographies, and general knowledge."
        ),
    )

    # ArXiv tool
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500,
        ),
        name="arxiv",
        description=(
            "Search ArXiv for academic papers, research articles, and scientific publications."
        ),
    )

    tools = [search_tool, wikipedia, arxiv]
    st.session_state.agent = create_react_agent(llm, tools)

# =========================================================
# CHAT INTERFACE
# =========================================================
if not st.session_state.openai_key or not st.session_state.tavily_key:
    st.info("🔧 Enter your OpenAI and Tavily API keys in the sidebar to start chatting.")
else:
    user_input = st.chat_input("Ask me anything…")

    if user_input and st.session_state.agent is not None:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            # Simple one-shot call (no streaming)
            result = st.session_state.agent.invoke({"messages": [("user", user_input)]})
            # result format depends on create_react_agent; adjust if needed
            st.write(result)
