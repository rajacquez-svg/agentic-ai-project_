# create_react_agent: Creates an agent that can reason and use tools
from langgraph.prebuilt import create_react_agent

# Wikipedia and ArXiv: Tools for encyclopedia and research papers
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


# =========================================================
# PAGE SETUP
 @@ -36,8 +40,11 @@
    layout="wide"  # Use full width of browser
)

st.title("🔍 Multi-Tool Chatbot Agent")
st.caption("AI agent with web search, Wikipedia, and ArXiv capabilities")





# =========================================================
 @@ -74,14 +81,19 @@
    else:
        st.warning("⚠️ Tavily Not Connected")

     if st.session_state.openai_key and st.session_state.tavily_key:
        st.subheader("🛠️ Available Tools")
        st.write("✅ **Tavily Search** - Web search")
        st.write("✅ **Wikipedia** - Encyclopedia")
        st.write("✅ **ArXiv** - Research papers")

    if st.session_state.openai_key or st.session_state.tavily_key:
        if st.button("Change API Keys"):
            # Reset everything to start fresh
            st.session_state.openai_key = ""
            st.session_state.tavily_key = ""
            st.rerun()

# =========================================================
# API KEYS INPUT
# =========================================================
 @@ -154,6 +166,34 @@
    # Create Tavily search tool
    search_tool = TavilySearchResults(max_results=3)

    # Create Wikipedia tool
    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        ),
        name="wikipedia",
        description="""Search Wikipedia for encyclopedia articles, historical information, 
        biographies, and general knowledge. Best for: 'Who was...', 'What is...', 
        'History of...', 'Explain...' queries."""
    )

    # Create ArXiv tool
    arxiv = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=500
        ),
        name="arxiv",
        description="""Search ArXiv for academic papers, research articles, and scientific 
        publications. Best for: 'Latest research on...', 'Papers about...', 
        'Scientific studies on...' queries."""
    )

    # Create agent with all tools
    tools = [search_tool, wikipedia, arxiv]
    st.session_state.agent = create_react_agent(llm, tools)

    # Create agent with search capability
    st.session_state.agent = create_react_agent(llm, [search_tool])
