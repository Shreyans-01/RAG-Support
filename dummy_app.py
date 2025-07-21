
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import requests
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Hardware Sentry",
    page_icon="🛡️",
    layout="wide"
)

# --- Gemini API Key ---
# The RAG chatbot requires a Gemini API key.
# For security, this is handled by the environment and should be left as an empty string.
# In a deployed Streamlit environment, this would be set in the secrets management.
GEMINI_API_KEY = "AIzaSyCNXkfdzdN1WB1cbY5kQWC7GsJj9jahkag"


# --- Caching for Performance ---
@st.cache_data
def load_data(filepath):
    """Loads hardware data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def create_vector_store(document_path):
    """Creates a FAISS vector store from a text document for retrieval."""
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)

    # Use a popular, lightweight embedding model that runs locally.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# --- Core Logic for RAG with Gemini ---
def query_gemini_with_rag(user_query, vectorstore):
    """
    Performs Retrieval-Augmented Generation using the Gemini API.
    1. Retrieves relevant documents from the vector store.
    2. Constructs a prompt with the retrieved context and instructions for out-of-scope answers.
    3. Queries the Gemini API.
    """
    if not vectorstore:
        return "Error: Vector store is not available."

    # 1. Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    
    if not docs:
        context = "No relevant information found in the manual for this query."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Construct the new, more flexible prompt for Gemini
    prompt_template = f"""
    You are a helpful AI assistant for hardware diagnostics. Your primary source of information is the technical manual context provided below.

    First, try to answer the user's question using ONLY the provided context.

    If the context does not contain the answer to the question, then use your general knowledge to provide a helpful answer. If you use your general knowledge, you MUST begin your response with the following disclaimer:
    "DISCLAIMER: The following information is from outside the provided technical manual and is based on general knowledge."

    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {user_query}

    ANSWER:
    """

    # 3. Query the Gemini API
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt_template}]
        }]
    }
    
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and 
            result['candidates'][0]['content']['parts'][0].get('text')):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            # Handle cases of safety blocks or unexpected response structure
            return "The AI model returned an empty response. This could be due to a safety filter or an issue with the request."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


# --- Core Anomaly Detection Logic ---
def detect_anomalies(df):
    """
    Detects anomalies using Isolation Forest on key hardware metrics.
    """
    df['power_util_ratio'] = df['power_watts'] / (df['utilization_percent'] + 0.01)
    model = IsolationForest(contamination='auto', random_state=42)
    features = df[['utilization_percent', 'power_watts', 'power_util_ratio']]
    df['anomaly_score'] = model.fit_predict(features)
    df['is_anomaly'] = df['anomaly_score'] == -1
    return df

# --- UI & Main Application Flow ---
st.title("🛡️ Hardware Sentry: Anomaly Detection & AI Support")
st.markdown("A prototype tool for monitoring server health, flagging anomalies, and providing diagnostic support via **Google's Gemini API**.")

data = load_data('hardware_data.csv')
vector_store = create_vector_store('dummy_manual.txt')

if data is None:
    st.error("`hardware_data.csv` not found. Please run `generate_data.py` first.")
else:
    data_with_anomalies = detect_anomalies(data.copy())
    anomalies_df = data_with_anomalies[data_with_anomalies['is_anomaly']].sort_values(by='timestamp', ascending=False)

    st.header("Live System Monitoring")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CPU Utilization (%)")
        fig_util = go.Figure()
        fig_util.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['utilization_percent'], mode='lines', name='Normal', line=dict(color='royalblue')))
        fig_util.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['utilization_percent'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
        fig_util.update_layout(xaxis_title='Timestamp', yaxis_title='Utilization (%)', showlegend=True)
        st.plotly_chart(fig_util, use_container_width=True)

    with col2:
        st.subheader("Power Draw (Watts)")
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=data_with_anomalies['timestamp'], y=data_with_anomalies['power_watts'], mode='lines', name='Normal', line=dict(color='limegreen')))
        fig_power.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['power_watts'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
        fig_power.update_layout(xaxis_title='Timestamp', yaxis_title='Power (Watts)', showlegend=True)
        st.plotly_chart(fig_power, use_container_width=True)

    st.header("🚩 Flagged Anomalies")
    if anomalies_df.empty:
        st.success("No anomalies detected in the current dataset.")
    else:
        st.warning(f"Detected {len(anomalies_df)} potential anomalies. Please investigate.")
        st.dataframe(
            anomalies_df[['timestamp', 'utilization_percent', 'power_watts', 'power_util_ratio']].style.format({
                'utilization_percent': '{:.2f}%',
                'power_watts': '{:.2f}W',
                'power_util_ratio': '{:.2f}'
            }),
            use_container_width=True
        )

    st.header("🤖 AI Diagnostic Assistant (Powered by Gemini)")
    st.markdown("Ask a question about the anomalies or system behavior. The AI will answer based on the `dummy_manual.txt`.")

    if vector_store is None:
        st.warning("AI Assistant is offline. Please ensure the `dummy_manual.txt` file exists.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("e.g., What causes high power draw at low utilization?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consulting the manual and querying Gemini..."):
                    response = query_gemini_with_rag(prompt, vector_store)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})