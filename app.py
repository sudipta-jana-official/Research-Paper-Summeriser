from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import numpy as np

# AI Libraries
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# =========================
# LOAD MODELS (ONLY ONCE)
# =========================
print("🚀 Loading AI models...")

# 1. Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# 2. Embedding Model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# 3. Text Generation Model (Turns Context -> Real Answer)
generator = pipeline("text2text-generation", model="google/flan-t5-large")

print("✅ Models loaded!")

# =========================
# GLOBAL STORAGE (Single Session for Project Deployment)
# =========================
faiss_index = None
chunks_store = []

# =========================
# HELPER FUNCTIONS
# =========================
def extract_text_from_pdf_file(file):
    """Extracts text seamlessly from a file stream without saving it to disk."""
    file.seek(0)  
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into smaller chunks to fit into embedding models."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def summarize_long_text(text):
    """Iteratively summarizes text chunks and outputs a final high-quality summary."""
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    summaries = []
    
    # Process up to 8 chunks to cover longer documents safely
    for chunk in chunks[:8]:
        try:
            res = summarizer(chunk, max_length=140, min_length=60, do_sample=False)
            summaries.append(res[0]['summary_text'])
        except Exception:
            continue

    if not summaries:
        return "⚠️ Could not generate summary."

    combined = " ".join(summaries)[:3500]
    final = summarizer(combined, max_length=350, min_length=150, length_penalty=1.5, do_sample=False)
    return final[0]['summary_text']

def build_faiss(chunks):
    """Generates embeddings and builds the local FAISS index vector database."""
    global faiss_index, chunks_store

    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')

    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    chunks_store = chunks

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    """Renders the attractive project homepage portfolio."""
    return render_template('index.html')


@app.route('/summery')
def summery():
    """Renders the workspace operational interactive dashboard."""
    return render_template('summery.html')


@app.route('/summarize', methods=['POST'])
def summarize():
    """Processes incoming data to generate an AI summary and populate the FAISS context database."""
    try:
        text = request.form.get('text')
        pdf_file = request.files.get('pdf')

        if pdf_file and pdf_file.filename != "":
            final_text = extract_text_from_pdf_file(pdf_file)
        elif text:
            final_text = text
        else:
            return jsonify({"summary": "⚠️ No input provided"})

        if len(final_text.split()) < 30:
            return jsonify({"summary": "⚠️ Text too short to analyze properly (Min 30 words)."})

        # 1. Pipeline Summary Engine
        summary = summarize_long_text(final_text)

        # 2. Pipeline Local RAG Index Vectorization
        chunks = chunk_text(final_text)
        build_faiss(chunks)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"summary": f"Error during parsing: {str(e)}"})


@app.route('/chat', methods=['POST'])
def chat():
    """Answers context questions by cross-referencing user queries with the FAISS store via Flan-T5."""
    try:
        global faiss_index, chunks_store

        if faiss_index is None or not chunks_store:
            return jsonify({"reply": "⚠️ Context database missing. Please process a document first."})

        data = request.get_json()
        query = data.get("message")
        
        if not query:
            return jsonify({"reply": "⚠️ Please type a query."})

        # 1. Similarity Retrieval
        query_vector = embedder.encode([query]).astype('float32')
        D, I = faiss_index.search(query_vector, k=3)

        context_chunks = [chunks_store[i] for i in I[0] if i < len(chunks_store)]
        context = " ".join(context_chunks)

        # 2. Strict Prompt Instruction Setup
        prompt = f"Answer the question based strictly on the provided context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"

        # 3. Model Pipeline Execution
        generation_results = generator(prompt, max_length=150, min_length=10, do_sample=False)
        answer = generation_results[0]['generated_text']

        return jsonify({"reply": answer})

    except Exception as e:
        return jsonify({"reply": f"Error running context inference: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True, port=5001)