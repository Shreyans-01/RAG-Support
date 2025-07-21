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

st.set_page_config(
    page_title="Hardware Sentry",
    page_icon="🛡️",
    layout="wide"
)

GEMINI_API_KEY = "AIzaSyCNXkfdzdN1WB1cbY5kQWC7GsJj9jahkag"


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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

def query_gemini_with_rag(user_query, vectorstore, anomalies_df=None):
  
    if not vectorstore:
        return "Error: Vector store is not available."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_query)
    
    if not docs:
        manual_context = "No relevant information found in the manual for this query."
    else:
        manual_context = "\n\n".join([doc.page_content for doc in docs])

    if anomalies_df is not None and not anomalies_df.empty:
        anomaly_context = f"""
Here is a summary of the currently flagged anomalies from the monitoring system:
{anomalies_df[['timestamp', 'utilization_percent', 'power_watts', 'power_util_ratio']].to_string()}
"""
    else:
        anomaly_context = "No anomalies are currently flagged by the system."

    prompt_template = f"""
    You are a helpful AI assistant for hardware diagnostics. You have two sources of information:
    1. A technical manual.
    2. Live anomaly data from the monitoring system.

    Your task is to answer the user's question by synthesizing information from both sources. Refer to specific data points from the live data if relevant.

    First, try to answer using the provided "TECHNICAL MANUAL CONTEXT" and the "LIVE ANOMALY DATA".

    If these sources do not contain the answer, then use your general knowledge. If you use your general knowledge, you MUST begin your response with the following disclaimer:
    "DISCLAIMER: The following information is from outside the provided technical manual and is based on general knowledge."

    ---
    TECHNICAL MANUAL CONTEXT:
    {manual_context}
    ---
    LIVE ANOMALY DATA:
    {anomaly_context}
    ---

    QUESTION:
    {user_query}

    ANSWER:
    """

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
        response.raise_for_status()
        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and 
            result['candidates'][0]['content']['parts'][0].get('text')):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "The AI model returned an empty response. This could be due to a safety filter or an issue with the request."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def detect_anomalies(df):
    
    df['power_util_ratio'] = df['power_watts'] / (df['utilization_percent'] + 0.01)
    model = IsolationForest(contamination='auto', random_state=42)
    features = df[['utilization_percent', 'power_watts', 'power_util_ratio']]
    df['anomaly_score'] = model.fit_predict(features)
    df['is_anomaly'] = df['anomaly_score'] == -1
    return df

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

    st.header("🤖 AI Diagnostic Assistant")
    st.markdown("Ask a question about the anomalies or system behavior. The AI will answer based on the `dummy_manual.txt` and live data.")

    if vector_store is None:
        st.warning("AI Assistant is offline. Please ensure the `dummy_manual.txt` file exists.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("e.g., Why was the first anomaly flagged?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Consulting the manual and querying Gemini..."):
                    response = query_gemini_with_rag(prompt, vector_store, anomalies_df)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})