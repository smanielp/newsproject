from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.graph.workflow import NewsGenieWorkflow

st.set_page_config(page_title="NewsGenie", page_icon="🧞‍♂️", layout="wide")

st.title("NewsGenie: Your AI-Powered News Assistant")

# Initialize the workflow
workflow = NewsGenieWorkflow().build()

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for news category selection
st.sidebar.title("News Categories")
category = st.sidebar.radio(
    "Choose a category:",
    ("business", "technology", "sports")
)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to session state
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("human"):
        st.markdown(prompt)

    # Invoke the workflow with the entire conversation history
    with st.spinner("Thinking..."):
        graph_input = {
            "messages": st.session_state.messages,
            "category": category
        }

        response = workflow.invoke(graph_input)

        # The final response is the last message from the AI
        final_response = response['messages'][-1]

        st.session_state.messages.append(final_response)
        with st.chat_message("ai"):
            st.markdown(final_response.content)
