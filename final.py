import os
import streamlit as st
import torch
import dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="AI Insurance Chatbot",
    page_icon="💬",
    layout="centered"
)

# App title and description
st.title("AI Insurance Chatbot")
st.markdown("Ask about Health, Life, Auto, or Home insurance policies.")

# Load environment variables
@st.cache_resource
def load_environment():
    dotenv.load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables!")
    return api_key

# Initialize Groq client
@st.cache_resource
def get_groq_client():
    api_key = load_environment()
    return Groq(api_key=api_key)

# Load embedder model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load and process knowledge base
@st.cache_data
def load_chunks(file_path="knowledge.txt", chunk_size=500, overlap=100):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size - overlap)]
        return chunks
    except FileNotFoundError:
        st.error(f"Knowledge base file not found: {file_path}")
        return ["No knowledge base available."]

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize resources
groq_client = get_groq_client()
embedder = load_embedder()
chunks = load_chunks()
chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
MODEL_NAME = "llama-3.3-70b-versatile"

# RAG Functions
def get_groq_response(prompt):
    try:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert insurance assistant chatbot."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error from LLM API: {str(e)}")
        return "I'm having trouble connecting to my knowledge base. Please try again later."

def rag_chat(user_query):
    # Display thinking indicator
    with st.spinner("Thinking..."):
        # 1. Embed user query
        query_emb = embedder.encode([user_query], convert_to_tensor=True)
        
        # 2. Compute cosine similarity
        similarities = cosine_similarity(query_emb.cpu(), chunk_embeddings.cpu())[0]
        top_indices = similarities.argsort()[-3:][::-1]  # Get top 3 most relevant chunks
        context = "\n".join([chunks[i] for i in top_indices])
        
        # 3. Compose Prompt
        prompt = f"""
Use the context below to answer the user's question.

Context:
{context}

Question:
{user_query}
"""
        
        # 4. Get Groq LLM response
        return get_groq_response(prompt)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about premiums, claims, policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = rag_chat(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.subheader("About")
    st.write("This chatbot uses RAG (Retrieval-Augmented Generation) to answer your insurance questions based on a knowledge base.")
    
    st.subheader("Options")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Add model info
    st.subheader("Model Information")
    st.write(f"LLM: {MODEL_NAME}")
    st.write("Embeddings: all-MiniLM-L6-v2")
    
    # Add debug info toggle
    show_debug = st.toggle("Show Debug Info", value=False)
    
    if show_debug:
        st.subheader("Debug Information")
        st.write(f"Knowledge Base: {len(chunks)} chunks")
        if st.button("Show Example Chunks"):
            st.write("First chunk:")
            st.text(chunks[0][:200] + "...")