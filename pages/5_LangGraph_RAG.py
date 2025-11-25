# RAG Agent App - Streamlit + LangGraph
# Data Science Dojo 2016–2025 (adapted to Streamlit chat UI)

# =========================================================
# IMPORTS
# =========================================================
import warnings
warnings.filterwarnings("ignore")

import os
import pprint
from typing import Annotated, Sequence, TypedDict, Literal

import streamlit as st

# LangChain / LangGraph
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition


# =========================================================
# STREAMLIT PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="RAG Agent (Lilian Weng Blog)",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Agent with LangGraph")
st.caption("Ask questions about Lilian Weng’s blog posts (Agents, Prompt Engineering, Adversarial Attacks)")

# =========================================================
# SESSION STATE
# =========================================================
if "openai_key" not in st.session_state:
    st.session_state.openai_key = ""

if "graph" not in st.session_state:
    st.session_state.graph = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = []  # [{"role": "user"/"assistant", "content": "..."}]


# =========================================================
# SIDEBAR - API KEY
# =========================================================
with st.sidebar:
    st.subheader("🔑 OpenAI API Key")

    if st.session_state.openai_key:
        st.success("✅ OpenAI Connected")
    else:
        st.warning("⚠️ OpenAI Not Connected")

    if st.session_state.openai_key:
        if st.button("Change API Key"):
            st.session_state.openai_key = ""
            st.session_state.graph = None
            st.session_state.retriever = None
            st.rerun()

# Ask for key if missing
if not st.session_state.openai_key:
    key_input = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

    if st.button("Connect"):
        if not key_input or not key_input.startswith("sk-"):
            st.error("❌ Invalid API key format")
        else:
            st.session_state.openai_key = key_input
            st.rerun()

    st.stop()  # Don’t continue until key is set


# =========================================================
# CORE RAG AGENT SETUP (only once per session)
# =========================================================

# --- 1. Environment & URLs ---
os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
os.environ["USER_AGENT"] = "MyApp/1.0"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


# --- 2. Define Agent State type (LangGraph) ---
class AgentState(TypedDict):
    # add_messages = append instead of replace
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --- 3. Build retriever, tools, and graph (once) ---
if st.session_state.graph is None:

    with st.spinner("🔄 Building vector index and RAG graph (first-time setup)..."):

        # 3.1 Load and split documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100,
            chunk_overlap=50,
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # 3.2 FAISS vectorstore + retriever
        vectorstore = FAISS.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(),
        )
        retriever = vectorstore.as_retriever()
        st.session_state.retriever = retriever

        # 3.3 Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
        )
        tools = [retriever_tool]

        # =====================================================
        # NODES & EDGES (functions for the graph)
        # =====================================================

        # ---- Edge: grade_documents (relevance check) ----
        def grade_documents(state) -> Literal["generate", "rewrite"]:
            """
            Decide whether retrieved documents are relevant to the question.
            Returns:
                "generate" if relevant, "rewrite" otherwise.
            """
            print("---CHECK RELEVANCE---")

            class Grade(BaseModel):
                """Binary score for relevance check."""
                binary_score: str = Field(description="Relevance score 'yes' or 'no'")

            model = ChatOpenAI(temperature=0.3, model="gpt-4o", streaming=True)
            llm_with_struct = model.with_structured_output(Grade)

            prompt = PromptTemplate(
                template=(
                    "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
                    "Here is the retrieved document:\n\n{context}\n\n"
                    "Here is the user question: {question}\n\n"
                    "If the document contains keyword(s) or semantic meaning related to the user question, "
                    "grade it as relevant.\n"
                    "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
                ),
                input_variables=["context", "question"],
            )

            chain = prompt | llm_with_struct

            messages = state["messages"]
            last_message = messages[-1]
            question = messages[0].content
            docs_content = getattr(last_message, "content", last_message)

            scored_result = chain.invoke(
                {"question": question, "context": docs_content}
            )
            score = scored_result.binary_score.strip().lower()

            if score == "yes":
                print("---DECISION: DOCS RELEVANT---")
                return "generate"
            else:
                print("---DECISION: DOCS NOT RELEVANT---")
                return "rewrite"

        # ---- Node: agent (decide whether to call tool) ----
        def agent(state):
            """
            Agent node: uses tools (retriever) or stops.
            """
            print("---CALL AGENT---")
            messages = state["messages"]
            model = ChatOpenAI(temperature=0.4, streaming=True, model="gpt-4o")
            model = model.bind_tools(tools)
            response = model.invoke(messages)
            return {"messages": [response]}

        # ---- Node: rewrite (improve question) ----
        def rewrite(state):
            """
            Rewrite the user question to be clearer / better for retrieval.
            """
            print("---TRANSFORM QUERY---")

            messages = state["messages"]
            original_question = messages[0].content

            msg = [
                HumanMessage(
                    content=(
                        "Look at the input and reason about the underlying semantic intent.\n"
                        "Here is the initial question:\n"
                        "-------\n"
                        f"{original_question}\n"
                        "-------\n"
                        "Formulate an improved question:"
                    )
                )
            ]

            model = ChatOpenAI(temperature=0.3, model="gpt-4o", streaming=True)
            response = model.invoke(msg)
            return {"messages": [response]}

        # ---- Node: generate (final answer) ----
        def generate(state):
            """
            Generate answer based on retrieved context and question.
            """
            print("---GENERATE---")
            messages = state["messages"]
            question = messages[0].content
            last_message = messages[-1]
            docs_content = getattr(last_message, "content", last_message)

            rag_prompt = hub.pull("rlm/rag-prompt")
            llm = ChatOpenAI(model="gpt-4o", temperature=0.4, streaming=True)

            rag_chain = rag_prompt | llm | StrOutputParser()

            response = rag_chain.invoke(
                {"context": docs_content, "question": question}
            )
            # StrOutputParser gives a string
            return {"messages": [response]}

        # Optional: print the RAG prompt in server logs
        print("*" * 20 + " Prompt [rlm/rag-prompt] " + "*" * 20)
        try:
            _ = hub.pull("rlm/rag-prompt").pretty_print()
        except Exception:
            pass

        # =====================================================
        # BUILD GRAPH
        # =====================================================
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("agent", agent)
        retrieve_node = ToolNode([retriever_tool])
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("rewrite", rewrite)
        workflow.add_node("generate", generate)

        # Edges
        workflow.add_edge(START, "agent")

        # From agent: either call tools (-> retrieve) or END
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        # From retrieve: grade docs and go to generate or rewrite
        workflow.add_conditional_edges("retrieve", grade_documents)

        # After generate: end
        workflow.add_edge("generate", END)

        # After rewrite: go back to agent
        workflow.add_edge("rewrite", "agent")

        # Compile graph
        graph = workflow.compile()

        st.session_state.graph = graph

# At this point, we have st.session_state.graph ready to use.


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================
for msg in st.session_state.rag_chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================================================
# HANDLE USER INPUT (CHAT LOOP)
# =========================================================
user_question = st.chat_input("Ask about Lilian Weng’s posts (agents, prompts, adversarial attacks)...")

if user_question:
    # Show user message
    st.session_state.rag_chat_history.append(
        {"role": "user", "content": user_question}
    )
    with st.chat_message("user"):
        st.write(user_question)

    # Run the RAG graph (fresh state per question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking with RAG..."):
            graph = st.session_state.graph

            inputs = {
                "messages": [
                    ("user", user_question),
                ]
            }

            # We’ll just get the final state (not streaming node-by-node)
            final_state = graph.invoke(inputs)
            messages = final_state["messages"]

            last_msg = messages[-1]

            # Extract text from BaseMessage or raw string
            if isinstance(last_msg, BaseMessage):
                answer = last_msg.content
            else:
                answer = str(last_msg)

            st.write(answer)

            st.session_state.rag_chat_history.append(
                {"role": "assistant", "content": answer}
            )

    # You could also (optionally) log intermediate node outputs like this:
    # for node_output in graph.stream(inputs):
    #     pprint.pp(node_output)
