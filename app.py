import streamlit as st
import requests

# --- UI Configuration ---
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="centered")

# --- Custom CSS for a Visually Appealing UI ---
st.markdown("""
<style>
    /* General styles */
    .stApp {
        background-color: #e5e7eb; /* Soft gray background */
        font-family: 'Inter', sans-serif;
        padding: 20px;
    }
    /* Main content container */
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    /* Title style with gradient */
    h1 {
        background: linear-gradient(to right, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* Caption style */
    .stCaption {
        color: #4b5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Chat bubble styles */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #d1d5db;
        animation: fadeIn 0.3s ease-in;
        color: #1f2937; /* Dark text for readability */
    }
    /* User message */
    .stChatMessage[data-testid="chat-message-container-user"] {
        background-color: #93c5fd; /* Soft blue for user */
        border-left: 4px solid #1e40af;
    }
    /* Assistant message */
    .stChatMessage[data-testid="chat-message-container-assistant"] {
        background-color: #d1d5db; /* Light gray for assistant */
        border-left: 4px solid #4b5563;
    }
    /* Chat input style */
    .stChatInput > div > input {
        border-radius: 12px;
        border: 1px solid #9ca3af;
        padding: 12px 16px;
        font-size: 1rem;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: border-color 0.3s ease;
    }
    .stChatInput > div > input:focus {
        border-color: #3b82f6;
        outline: none;
    }
    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Ensure text contrast */
    .stMarkdown p {
        color: #1f2937 !important; /* Force dark text */
    }
</style>
""", unsafe_allow_html=True)

# --- App Title and Description ---
st.title("HR Policy Chatbot 🤖")
st.caption("🚀 Ask me anything about the company's HR policies, and I'll provide clear, concise answers!")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask something about the HR policy..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the user's question to the backend API
                response = requests.post("http://127.0.0.1:8000/query",
                                         json={"question": prompt})
                response.raise_for_status()  # Raise an error for bad status codes
                data = response.json()
                answer = data.get("answer", "Sorry, I couldn't get an answer.")
                sources = data.get("sources", [])
                if sources:
                    with st.expander("📄 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"**Source {i}:** {src}")
            except requests.exceptions.RequestException as e:
                answer = f"An error occurred: {e}"

            st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})