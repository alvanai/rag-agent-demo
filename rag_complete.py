import boto3
import json
import chromadb
import os
from pypdf import PdfReader

# --- CONFIGURATION ---
# IMPORTANT: You must have a file named 'notes.pdf' in this folder!
PDF_FILENAME = "notes.pdf" 
COLLECTION_NAME = "university_knowledge"

# --- 1. SETUP CLIENTS ---
print("--- INITIALIZING SYSTEMS ---")
# ChromaDB (The Memory - Saves to disk so it remembers next time)
chroma_client = chromadb.PersistentClient(path="./my_vector_db")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# AWS Bedrock (The Brain)
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# --- 2. INGESTION (Read PDF -> Vector DB) ---
def ingest_pdf(filename):
    if not os.path.exists(filename):
        print(f"ERROR: Could not find {filename}. Please put a PDF in this folder.")
        return

    print(f"Reading {filename}...")
    reader = PdfReader(filename)
    
    chunks = []
    ids = []
    
    # Split by page
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if len(text) > 50: # Ignore empty pages
            chunks.append(text)
            ids.append(f"page_{i}")
            
    print(f"Memorizing {len(chunks)} pages...")
    # .upsert() means "Insert if new, Update if exists"
    collection.upsert(documents=chunks, ids=ids)
    print("Ingestion Complete.")

# --- 3. RETRIEVAL (Search DB -> Ask AI) ---
def ask_agent(question):
    print(f"\nUser Question: {question}")
    
    # A. Search Memory
    print("Searching database...")
    results = collection.query(
        query_texts=[question],
        n_results=1 # Get the ONE best matching page
    )
    
    if not results['documents'][0]:
        return "I don't know the answer to that based on the documents."
        
    context_text = results['documents'][0][0]
    print("Found relevant context! Sending to Llama 3...")

    # B. Construct Prompt
    prompt = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a university tutor. Answer the question using ONLY the context below.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
CONTEXT:
{context_text}

QUESTION:
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    # C. Call AWS
    payload = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.3
    }
    
    response = bedrock.invoke_model(
        modelId="meta.llama3-8b-instruct-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )
    
    result = json.loads(response['body'].read())
    return result['generation']

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Ingest Data
    # Only runs if you have the file. 
    ingest_pdf(PDF_FILENAME)
    
    # 2. Start Chat Loop
    print("\nSystem Ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() == 'exit':
            break
            
        answer = ask_agent(user_input)
        print("\n=== AI ANSWER ===")
        print(answer)
        print("=================")
