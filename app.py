import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from groq import GroqError

# Load environment variables if available
load_dotenv()

# Initialize API Wrappers and Tools
try:
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    search = DuckDuckGoSearchRun(name="Search")
except Exception as e:
    st.sidebar.error(f"⚠️ Error initializing tools: {str(e)}")
    st.stop()

# Streamlit UI
st.title("🔎 LangChain - Chat with Search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions 
of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at 
[github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    st.sidebar.warning("⚠️ Please enter your Groq API key to continue.")
    st.stop()

# Initialize LLM only if API key is provided
try:
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    st.sidebar.success("✅ API Key is valid! Model initialized.")
except GroqError as e:
    st.sidebar.error(f"❌ Groq API Error: {str(e)}")
    st.sidebar.warning("Possible causes:\n- Invalid or missing API key.\n- Network issues.\n- API rate limits or service downtime.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"⚠️ Unexpected Error: {str(e)}")
    st.stop()

# Initialize chat messages session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages from session
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input Processing
prompt = st.chat_input(placeholder="What is machine learning?")
if prompt:
    if not api_key:
        st.sidebar.warning("⚠️ Please enter your Groq API key to continue.")
    else:
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        try:
            # Re-initialize LLM
            llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

            # Define search tools
            tools = [search, arxiv, wiki]

            # Initialize agent
            search_agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
            )

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                
                # Run agent and handle errors gracefully
                try:
                    response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                except Exception as e:
                    st.error(f"⚠️ Error processing response: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": "⚠️ An error occurred while processing your request. Please try again."})
                    st.write("⚠️ An error occurred while processing your request. Please try again.")

        except GroqError as e:
            st.sidebar.error(f"❌ Groq API Error: {str(e)}")
            st.sidebar.warning("Possible causes:\n- Invalid API key.\n- Network issues.\n- API rate limits or downtime.")
        except Exception as e:
            st.sidebar.error(f"⚠️ Unexpected Error: {str(e)}")
