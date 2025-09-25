import os
import json
import torch
import faiss
import numpy as np
import gradio as gr
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Paths
# -------------------------
INDEX_PATH = "data/vector_store/legal.index"
DOCS_PATH = "data/vector_store/docs.json"
MODEL_PATH = "models/flan_legal_model"
os.makedirs("data/vector_store", exist_ok=True)

# -------------------------
# Load models
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔹 Loading InLegalBERT embedder...")
embedder = SentenceTransformer("law-ai/InLegalBERT")

print("🔹 Loading summarizer/Q&A model...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

# -------------------------
# Load FAISS index + docs
# -------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        doc_map = json.load(f)
else:
    index = faiss.IndexFlatL2(768)  # InLegalBERT outputs 768-dim embeddings
    doc_map = []

# -------------------------
# Utils
# -------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF (fallback to OCR)."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    if not text:  # fallback OCR
        pages = convert_from_path(pdf_path)
        for p in pages:
            text.append(pytesseract.image_to_string(p, lang="eng"))
    return "\n".join(text)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split into overlapping chunks for FAISS indexing."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def summarize_text(text, max_tokens=250):
    """Summarize into Facts, Arguments, Decision."""
    prompt = f"""
You are a legal assistant. Summarize the following case into 3 sections:
1. Facts
2. Key Arguments
3. Decision / Verdict

Text:
{text}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=4,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def index_uploaded_pdfs(pdf_files):
    """Index uploaded PDFs into FAISS."""
    new_docs = []
    for pdf in pdf_files:
        text = extract_text_from_pdf(pdf.name)
        chunks = chunk_text(text)
        for chunk in chunks:
            emb = embedder.encode(chunk, convert_to_numpy=True).astype("float32")
            index.add(np.array([emb]))
            doc_map.append({"file": os.path.basename(pdf.name), "text": chunk})
        new_docs.append({"file": os.path.basename(pdf.name), "text": text})
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_map, f)
    return new_docs

def retrieve(query, top_k=3):
    """Retrieve relevant chunks from FAISS."""
    q_vec = embedder.encode(query, convert_to_numpy=True).astype("float32")
    D, I = index.search(q_vec.reshape(1, -1), top_k)
    return [doc_map[i] for i in I[0] if i < len(doc_map)]

def generate_answer(query, retrieved_docs):
    """Generate answer from retrieved chunks."""
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])
    prompt = f"""
You are a legal assistant. Use only the context below to answer the question.
If the answer is not in the context, say "Not found in the provided documents."

Context:
{context}

Question: {query}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=4,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Gradio Logic
# -------------------------
chat_history = []

def chatbot_fn(query, pdf_files):
    global chat_history
    if pdf_files:  # Upload → Index
        new_docs = index_uploaded_pdfs(pdf_files)
        summaries = [f"📄 {doc['file']} Summary:\n{summarize_text(doc['text'][:2000])}" for doc in new_docs]
        response = "\n\n".join(summaries)
    else:  # Query → Retrieve + Answer
        retrieved = retrieve(query, top_k=3)
        response = "❌ No relevant passages found." if not retrieved else generate_answer(query, retrieved)

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response})
    return chat_history, chat_history

def clear_chat():
    global chat_history
    chat_history = []
    return [], []

def summarizer_fn(pdf_file):
    """Standalone summarizer tab."""
    text = extract_text_from_pdf(pdf_file.name)
    return summarize_text(text[:3000])  # truncate for efficiency

# -------------------------
# UI
# -------------------------
with gr.Blocks(theme="default") as demo:
    gr.Markdown("## ⚖️ Legal Contract Assistant (Summarizer + Q&A Chatbot)")

    with gr.Tab("📑 Summarizer"):
        file_in = gr.File(label="Upload a legal PDF", file_types=[".pdf"])
        summary_out = gr.Textbox(label="Summary", lines=15)
        file_in.change(fn=summarizer_fn, inputs=file_in, outputs=summary_out)

    with gr.Tab("💬 Chatbot"):
        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(label="Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
                clear_btn = gr.Button("Clear Chat")
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat with Legal Docs", type="messages")
                query = gr.Textbox(label="Ask a question")
                send_btn = gr.Button("Ask")

        send_btn.click(
            chatbot_fn,
            inputs=[query, uploader],
            outputs=[chatbot, chatbot]
        )

        clear_btn.click(
            clear_chat,
            inputs=None,
            outputs=[chatbot, chatbot]
        )

# -------------------------
# Launch
# -------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
