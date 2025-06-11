import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF for PDF text extraction
import ollama
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for knowledge base (in-memory for simplicity)
knowledge_base = []  # Stores (text_chunk, embedding) tuples
ollama_model = "llama3" # You can change this to your preferred Ollama model (e.g., "mistral", "phi3")
embedding_model = "nomic-embed-text" # Ollama embedding model

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF document."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks with a specified overlap."""
    chunks = []
    if not text:
        return chunks
    words = text.split()
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def get_ollama_embedding(text):
    """Generates embeddings for text using Ollama's embeddings API."""
    try:
        # Ensure the embedding model is pulled locally: ollama pull nomic-embed-text
        response = ollama.embeddings(model=embedding_model, prompt=text)
        return response['embedding']
    except Exception as e:
        logging.error(f"Error getting Ollama embedding for text: '{text[:50]}...' - {e}")
        return None

def get_ollama_response(prompt, context=""):
    """Gets a response from the Ollama model."""
    full_prompt = f"Using the following context, answer the question. If the question cannot be answered from the context, state that. If no context is provided, answer as a general AI assistant.\n\nContext: {context}\n\nQuestion: {prompt}\n\nAnswer:"
    try:
        response = ollama.generate(model=ollama_model, prompt=full_prompt)
        return response['response']
    except Exception as e:
        logging.error(f"Error getting Ollama response: {e}")
        return "Sorry, I am unable to process your request at the moment."

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handles PDF file uploads, extracts text, chunks it, and generates embeddings."""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        # Process PDF
        extracted_text = extract_text_from_pdf(filepath)
        if extracted_text:
            chunks = chunk_text(extracted_text)
            global knowledge_base
            knowledge_base = [] # Clear previous knowledge base
            for i, chunk in enumerate(chunks):
                embedding = get_ollama_embedding(chunk)
                if embedding:
                    knowledge_base.append((chunk, np.array(embedding)))
                    logging.info(f"Processed chunk {i+1}/{len(chunks)}")
                else:
                    logging.warning(f"Could not get embedding for chunk {i+1}")
            
            if knowledge_base:
                return jsonify({'message': f'PDF "{filename}" processed successfully. Knowledge base updated with {len(knowledge_base)} chunks.'}), 200
            else:
                return jsonify({'error': 'Failed to generate embeddings from PDF. Ensure Ollama embedding model is running.'}), 500
        else:
            return jsonify({'error': 'Failed to extract text from PDF.'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Please upload a PDF.'}), 400

@app.route('/ask', methods=['POST'])
def ask():
    """Handles user questions, performs RAG if a knowledge base exists, or answers generally."""
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    response_text = ""
    context = ""

    if knowledge_base:
        # Retrieve relevant chunks from the knowledge base
        question_embedding = get_ollama_embedding(question)
        if question_embedding is None:
            return jsonify({'answer': "Error: Could not generate embedding for your question."}), 500
        
        question_embedding = np.array(question_embedding).reshape(1, -1)
        
        similarities = []
        for chunk_text_item, chunk_embedding in knowledge_base:
            sim = cosine_similarity(question_embedding, chunk_embedding.reshape(1, -1))[0][0]
            similarities.append((sim, chunk_text_item))
        
        # Sort by similarity and get the top N chunks
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Concatenate top chunks to form context, adjust `top_n` as needed
        top_n = 3
        relevant_chunks = [chunk for sim, chunk in similarities[:top_n] if sim > 0.5] # Only include if similarity is above a threshold
        context = "\n\n".join(relevant_chunks)

        if not context:
            logging.info("No sufficiently similar context found, answering generally.")
            response_text = get_ollama_response(question, context="") # No relevant context, so answer generally
        else:
            response_text = get_ollama_response(question, context=context)
    else:
        # No knowledge base, answer general queries
        logging.info("No knowledge base loaded, answering general query.")
        response_text = get_ollama_response(question, context="")

    return jsonify({'answer': response_text})

if __name__ == '__main__':
    # Ensure a model for general responses is available
    logging.info(f"Attempting to pull Ollama model: {ollama_model}. This might take a while if not already present.")
    try:
        ollama.pull(ollama_model)
        logging.info(f"Ollama model '{ollama_model}' is ready.")
    except Exception as e:
        logging.error(f"Failed to pull Ollama model '{ollama_model}'. Please ensure Ollama server is running and the model name is correct. Error: {e}")
        logging.error("The application might not function correctly without the specified Ollama model.")

    # Ensure embedding model is available
    logging.info(f"Attempting to pull Ollama embedding model: {embedding_model}. This might take a while if not already present.")
    try:
        ollama.pull(embedding_model)
        logging.info(f"Ollama embedding model '{embedding_model}' is ready.")
    except Exception as e:
        logging.error(f"Failed to pull Ollama embedding model '{embedding_model}'. Please ensure Ollama server is running and the model name is correct. Error: {e}")
        logging.error("PDF processing will not work without the embedding model.")

    app.run(debug=True, port=5000)

