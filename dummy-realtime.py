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
import time
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Hardware Sentry (Live)",
    page_icon="📡",
    layout="wide"
)

# --- Gemini API Key ---
# The RAG chatbot requires a Gemini API key.
# For security, this is handled by the environment and should be left as an empty string.
GEMINI_API_KEY = "AIzaSyCNXkfdzdN1WB1cbY5kQWC7GsJj9jahkag"

# --- Caching for Resource-Intensive Objects ---
@st.cache_resource
def create_vector_store(document_path):
    """Creates a FAISS vector store from a text document for retrieval."""
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        st.error(f"Error: The manual file '{document_path}' was not found.")
        return None

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# --- Real-Time Data Generation ---
def generate_new_data_point(step):
    """Generates a single new data point, simulating a real-time sensor reading."""
    utilization_base = 50
    power_base = 200
    noise_level = 5
    power_ratio = 3.5
    utilization = utilization_base + 15 * np.sin(np.pi * step / 50) + np.random.randn() * noise_level / 2
    power = power_base + (utilization * power_ratio) + np.random.randn() * (noise_level * 2)

    if 50 <= step < 55:
        power += 150
        utilization -= 20
    elif 100 <= step < 103:
        power *= 1.5
    
    utilization = np.clip(utilization, 0, 100)
    power = np.clip(power, 0, 1000)

    return pd.DataFrame([{'timestamp': datetime.datetime.now(), 'utilization_percent': utilization, 'power_watts': power}])

# --- Core Anomaly Detection Logic ---
def detect_anomalies(df):
    """Detects anomalies on the full dataset using Isolation Forest."""
    if len(df) < 2:
        df['is_anomaly'] = False
        df['power_util_ratio'] = 0
        return df

    df['power_util_ratio'] = df['power_watts'] / (df['utilization_percent'] + 0.01)
    model = IsolationForest(contamination='auto', random_state=42)
    features = df[['utilization_percent', 'power_watts', 'power_util_ratio']]
    df['is_anomaly'] = model.fit_predict(features) == -1
    return df

# --- Core Logic for RAG with Gemini ---
def query_gemini_with_rag(user_query, vectorstore, anomalies_df=None):
    """Performs Retrieval-Augmented Generation using the Gemini API and live data."""
    if not vectorstore:
        return "Error: Vector store is not available."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    manual_context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant information found in the manual."

    anomaly_context = "No anomalies are currently flagged."
    if anomalies_df is not None and not anomalies_df.empty:
        # Ensure the context string includes the power_util_ratio
        anomaly_context = f"Here is a summary of the currently flagged anomalies:\n{anomalies_df[['timestamp', 'utilization_percent', 'power_watts', 'power_util_ratio']].to_string()}"

    prompt_template = f"""
    You are an AI assistant for hardware diagnostics. You have two sources of information: a technical manual and live anomaly data.
    First, answer the user's question using the provided "TECHNICAL MANUAL CONTEXT" and "LIVE ANOMALY DATA".
    If these sources don't have the answer, use your general knowledge, but you MUST start your response with: "DISCLAIMER: This is based on general knowledge."

    ---
    TECHNICAL MANUAL CONTEXT:
    {manual_context}
    ---
    LIVE ANOMALY DATA:
    {anomaly_context}
    ---

    QUESTION: {user_query}
    ANSWER:
    """

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_template}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error connecting to Gemini API: {e}"

# --- Initialize Session State ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['timestamp', 'utilization_percent', 'power_watts'])
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
if 'simulation_step' not in st.session_state:
    st.session_state.simulation_step = 0

# --- Main Application UI ---
st.title("📡 Hardware Sentry: Live Anomaly Detection")
st.markdown("A real-time simulation of a server health monitor. Press 'Start' to begin or resume the data stream.")

# --- Simulation Controls ---
col1, col2, _ = st.columns([1, 1, 5])
if col1.button("▶️ Start / Resume", use_container_width=True):
    st.session_state.run_simulation = True
if col2.button("⏹️ Stop / Pause", use_container_width=True):
    st.session_state.run_simulation = False

# --- Dashboard Layout (Placeholders for live updates) ---
st.header("Live System Monitoring")
chart_col1, chart_col2 = st.columns(2)
util_placeholder = chart_col1.empty()
power_placeholder = chart_col2.empty()
anomaly_placeholder = st.empty()

# --- AI Chat Support Section (Always Visible) ---
st.header("🤖 AI Diagnostic Assistant (Powered by Gemini)")
st.markdown("Ask a question about the anomalies or system behavior. The AI will answer based on the `dummy_manual.txt` and live data.")

vector_store = create_vector_store('dummy_manual.txt')
if vector_store is None:
    st.warning("AI Assistant is offline. Please ensure the `dummy_manual.txt` file exists.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process chat input. This is kept outside the main simulation loop.
    if prompt := st.chat_input("e.g., Why was the latest anomaly flagged?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting the manual and querying Gemini..."):
                # We need the latest processed data for the context
                df_processed_for_chat = detect_anomalies(st.session_state.data.copy())
                anomalies_for_chat = df_processed_for_chat[df_processed_for_chat['is_anomaly']]
                response = query_gemini_with_rag(prompt, vector_store, anomalies_for_chat)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main Simulation and Rendering Loop ---
# This loop now controls the data generation and updates the UI placeholders.
while st.session_state.run_simulation:
    # Generate new data and add it to the session state
    new_data_point = generate_new_data_point(st.session_state.simulation_step)
    st.session_state.data = pd.concat([st.session_state.data, new_data_point], ignore_index=True)
    st.session_state.simulation_step += 1

    # Keep the DataFrame from growing indefinitely for performance
    if len(st.session_state.data) > 200:
        st.session_state.data = st.session_state.data.tail(200)
    
    # Process the data to find anomalies
    df_processed = detect_anomalies(st.session_state.data.copy())
    anomalies_df = df_processed[df_processed['is_anomaly']].sort_values(by='timestamp', ascending=False)

    # --- Update UI Placeholders ---
    with util_placeholder.container():
        st.subheader("CPU Utilization (%)")
        fig_util = go.Figure()
        fig_util.add_trace(go.Scatter(x=df_processed['timestamp'], y=df_processed['utilization_percent'], mode='lines', name='Normal', line=dict(color='royalblue')))
        if not anomalies_df.empty:
            fig_util.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['utilization_percent'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
        st.plotly_chart(fig_util, use_container_width=True)

    with power_placeholder.container():
        st.subheader("Power Draw (Watts)")
        fig_power = go.Figure()
        fig_power.add_trace(go.Scatter(x=df_processed['timestamp'], y=df_processed['power_watts'], mode='lines', name='Normal', line=dict(color='limegreen')))
        if not anomalies_df.empty:
            fig_power.add_trace(go.Scatter(x=anomalies_df['timestamp'], y=anomalies_df['power_watts'], mode='markers', name='Anomaly', marker=dict(color='red', size=10, symbol='x')))
        st.plotly_chart(fig_power, use_container_width=True)

    with anomaly_placeholder.container():
        st.header("🚩 Flagged Anomalies")
        if anomalies_df.empty:
            st.success("No anomalies detected in the current data stream.")
        else:
            st.warning(f"Detected {len(anomalies_df)} potential anomalies.")
            # Ensure the power_util_ratio is displayed in the table
            st.dataframe(anomalies_df[['timestamp', 'utilization_percent', 'power_watts', 'power_util_ratio']], use_container_width=True)

    # Pause for 1 second to simulate a real-time interval
    time.sleep(1)