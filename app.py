
import os
from getpass import getpass

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# --- API Key Setup ---
# For Streamlit deployment, it's recommended to use st.secrets or environment variables.
# If running locally for testing, you might use getpass.
if "GROQ_API_KEY" not in st.session_state:
    if os.environ.get("GROQ_API_KEY"):
        st.session_state.GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    elif "GROQ_API_KEY" in st.secrets:
        st.session_state.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    else:
        st.error("GROQ_API_KEY not found. Please set it as an environment variable or in Streamlit secrets.")
        st.stop()

# --- LLM Initialization ---
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=st.session_state.GROQ_API_KEY)

# --- Question Classification ---
CLASSIFY_PROMPT = """Classify the user's question as exactly one word:
either "policy" or "general".
policy = questions about company rules, leave, expenses, equipment,
benefits, conduct, or HR processes.
general = anything else (small talk, general knowledge, unrelated topics).
Examples:
Question: How many paid leave days do I get per year?
Answer: policy
Question: What's a good recipe for banana bread?
Answer: general
Question: Can I expense a client dinner?
Answer: policy
Question: What is the capital of France?
Answer: general
Now classify this question. Answer with exactly one word, nothing else.
Question: {question}
Answer:"""

def classify_question(question: str) -> str:
    prompt = CLASSIFY_PROMPT.format(question=question)
    response = llm.invoke(prompt).content.strip().lower()
    return "policy" if "policy" in response else "general"

# --- Handbook Data and Vectorstore ---
handbook_chunks = [
"""Leave Policy: Employees are entitled to 18 paid leave days per calendar
year, including casual and sick leave combined.""",
"""Laptop Policy: Company laptops are provided to all full-time employees
and must be returned upon exit. Personal use is permitted within
reasonable limits.""",
"""Remote Work Policy: Employees may work remotely up to 3 days per week
with manager approval, submitted via the HR portal.""",
"""Expense Policy: Business expenses including client meals and travel
are reimbursable with receipts submitted within 30 days.""",
"""Probation Policy: New employees undergo a 6-month probation period,
reviewed at the 3-month and 6-month marks.""",
"""Notice Period: Employees must serve a notice period of 60 days upon
resignation, unless otherwise agreed with HR.""",
"""Health Insurance: All employees are covered under group health
insurance from day one, extending to immediate family.""",
"""Working Hours: Standard working hours are 9:30 AM to 6:30 PM, Monday
to Friday, with flexible start times within a 1-hour window.""",
"""Grievance Redressal: Employees can raise workplace grievances
confidentially through the HR helpline, acknowledged within 2
working days.""",
"""Exit Process: Employees must complete a knowledge transfer plan
before their last working day; full settlement is processed
within 45 days.""",
]

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(handbook_chunks, embedding=embedding_model)
    return vectorstore

vectorstore = get_vectorstore()

def retrieve(query: str, k: int = 2):
    results = vectorstore.similarity_search(query, k=k)
    return [r.page_content for r in results]

# --- LangGraph Agent Definition ---
class HandbookState(TypedDict):
    question: str
    category: str
    context: str
    answer: str

def router_node(state: HandbookState):
    category = classify_question(state["question"])
    return {"category": category}

def rag_node(state: HandbookState):
    chunks = retrieve(state["question"], k=2)
    context = "\n".join(chunks)
    prompt = f"""Answer the question using ONLY the context below. Be concise.
Context:
{context}
Question: {state['question']}
Answer:"""
    answer = llm.invoke(prompt).content
    return {"context": context, "answer": answer}

def general_node(state: HandbookState):
    answer = llm.invoke(state["question"]).content
    return {"context": "(no retrieval -- general question)", "answer": answer}

def route_decision(state: HandbookState) -> str:
    return "rag_node" if state["category"] == "policy" else "general_node"

@st.cache_resource
def get_handbook_agent():
    builder = StateGraph(HandbookState)
    builder.add_node("router", router_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("general_node", general_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_decision, {"rag_node": "rag_node", "general_node": "general_node"})
    builder.add_edge("rag_node", END)
    builder.add_edge("general_node", END)
    return builder.compile()

handbook_agent = get_handbook_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="Employee Handbook Agent", layout="centered")
st.title("🧠 Employee Handbook Q&A Agent")
st.markdown("Ask me anything about the employee handbook or general knowledge questions!")

user_question = st.text_input("Your Question:", "How many paid leave days do I get?")

if user_question:
    with st.spinner("Thinking..."):
        result = handbook_agent.invoke({"question": user_question})

        st.subheader("Agent Response:")
        st.write(result["answer"])

        st.subheader("Details:")
        st.write(f"**Category:** {result['category']}")
        st.write(f"**Context Used:** {result['context']}")
