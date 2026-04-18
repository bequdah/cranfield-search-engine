import os
from flask import Flask, request, jsonify
from pathlib import Path

# Adjust path to find src module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import parse_cran_docs
from src.preprocess import preprocess_text, preprocess_collection
from src.vsm import VectorSpaceModel

# Initialize Flask app to serve static files from 'web' folder
app = Flask(__name__, static_url_path='', static_folder='web')

print("Starting Cranfield Vector Space Model Server...")
base_dir = Path(__file__).resolve().parent
dataset_dir = base_dir / "dataset"
docs_file = dataset_dir / "cran.all.1400"

print("Loading and indexing documents, please wait...")
# Load and index documents on startup
try:
    raw_docs = parse_cran_docs(docs_file)
    clean_docs = preprocess_collection(raw_docs)
    vsm = VectorSpaceModel()
    vsm.build_index(clean_docs)
    print(f"[SUCCESS] Index built with {len(vsm.inverted_index)} terms.")
except Exception as e:
    print(f"[ERROR] failed to build index: {e}")
    raw_docs = {}
    vsm = None

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    if not vsm:
        return jsonify({"error": "Index not loaded"}), 500
        
    data = request.get_json()
    query_text = data.get("query", "")
    k = data.get("k", 10)
    
    if not query_text.strip():
        return jsonify({"results": [], "query_tokens": []})
        
    # Apply exactly the same preprocessing to the user's query
    query_tokens = preprocess_text(query_text)
    
    # Retrieve top k documents
    start_time = time.time() if 'time' in globals() else 0
    import time
    start_time = time.time()
    
    top_k = vsm.get_top_k(query_tokens, k=k)
    search_time = time.time() - start_time
    
    results = []
    for doc_id, score in top_k:
        doc_text = raw_docs.get(doc_id, "")
        results.append({
            "doc_id": doc_id,
            "score": round(score, 4),
            "snippet": doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
        })
        
    return jsonify({
        "results": results,
        "query_tokens": query_tokens,
        "time_ms": round(search_time * 1000)
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Server running at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
