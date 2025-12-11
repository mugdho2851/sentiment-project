import os
import json
import numpy as np
from flask import Flask, request, jsonify
import faiss
import threading
import pickle

app = Flask(__name__)
LOCK = threading.Lock()

DATA_DIR = os.environ.get("VECTORSTORE_DATA_DIR", "/data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.pkl")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")  # New history file

EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

def init_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
        else:
            meta = []
    else:
        index = faiss.IndexFlatIP(EMBED_DIM)  # cosine via normalizing outside
        meta = []
    return index, meta

INDEX, META = init_index()

# Load history if exists
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, "r") as f:
        HISTORY = json.load(f)
else:
    HISTORY = []

def persist():
    """Persist index and metadata to disk."""
    with LOCK:
        faiss.write_index(INDEX, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(META, f)
        # persist history as JSON
        with open(HISTORY_PATH, "w") as f:
            json.dump(HISTORY, f, indent=2)

@app.route("/add", methods=["POST"])
def add():
    """
    Expects JSON:
    {
      "id": "<string id>",
      "embedding": [..],
      "metadata": {...}
    }
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "invalid json"}), 400
    id_ = payload.get("id")
    emb = payload.get("embedding")
    metadata = payload.get("metadata", {})
    if id_ is None or emb is None:
        return jsonify({"error": "missing id or embedding"}), 400

    vec = np.array(emb, dtype="float32").reshape(1, -1)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    vec = vec / norm

    with LOCK:
        INDEX.add(vec)
        META.append({"id": id_, "metadata": metadata})
        # Add to history log
        HISTORY.append({"id": id_, "embedding": emb, "metadata": metadata})
        persist()

    return jsonify({"status": "ok", "current_count": INDEX.ntotal})

@app.route("/query", methods=["POST"])
def query():
    """
    Expects:
    {
      "embedding": [...],
      "top_k": 5
    }
    """
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "invalid json"}), 400
    emb = payload.get("embedding")
    top_k = int(payload.get("top_k", 5))
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm==0] = 1.0
    vec = vec / norm
    with LOCK:
        if INDEX.ntotal == 0:
            return jsonify({"results": []})
        D, I = INDEX.search(vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            meta = META[idx] if idx < len(META) else {}
            results.append({"score": float(score), "id": meta.get("id"), "metadata": meta.get("metadata")})
    return jsonify({"results": results})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"count": INDEX.ntotal, "history_count": len(HISTORY)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
