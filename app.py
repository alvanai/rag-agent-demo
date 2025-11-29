import streamlit as st
import boto3
import json
import chromadb
import os
from pypdf import PdfReader

# --- PAGE CONFIG ---
st.set_page_config(page_title="My RAG Agent", layout="centered")
st.title("🤖 Document Chatbot (Llama 3)")

# --- 1. SETUP SYSTEMS (Cached for speed) ---
@st.cache_resource
def get_systems():
    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")
    collection = chroma_client.get_or_create_collection(name="web_uploads")
    
    # Setup AWS Bedrock
    if "AWS_ACCESS_KEY_ID" in st.secrets:
        bedrock = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1"
    )
else:
    # Fallback to local machine keys (~/.aws/credentials)
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

collection, bedrock = get_systems()

# --- 2. BACKEND FUNCTIONS ---
def ingest_pdf(uploaded_file):
    if uploaded_file is None:
        return
    
    # Save temp file because pypdf needs a real path
    temp_filename = "temp_upload.pdf"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    reader = PdfReader(temp_filename)
    chunks = []
    ids = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if len(text) > 50:
            chunks.append(text)
            ids.append(f"{uploaded_file.name}_page_{i}")
            
    if chunks:
        collection.upsert(documents=chunks, ids=ids)
        st.success(f"✅ Memorized {len(chunks)} pages from {uploaded_file.name}!")
        os.remove(temp_filename) # Cleanup

def ask_llm(question):
    # Search Memory
    results = collection.query(query_texts=[question], n_results=1)
    
    if not results['documents'][0]:
        return "I don't have enough info in the documents to answer that."
        
    context = results['documents'][0][0]
    
    # Prepare Prompt
    prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Answer using ONLY the context below.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
CONTEXT: {context}
QUESTION: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    
    # Call AWS
    payload = {"prompt": prompt, "max_gen_len": 512, "temperature": 0.3}
    response = bedrock.invoke_model(
        modelId="meta.llama3-8b-instruct-v1:0",
        body=json.dumps(payload)
    )
    result = json.loads(response['body'].read())
    return result['generation']

# --- 3. THE UI ---

# Sidebar for Uploads
with st.sidebar:
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Reading document..."):
                ingest_pdf(uploaded_file)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your PDF..."):
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Answer
    with st.chat_message("assistant"):
        response = ask_llm(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
