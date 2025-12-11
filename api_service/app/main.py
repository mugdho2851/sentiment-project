# api_service/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from model import SentimentModel
from preprocess import clean_text
import uuid

VECTORSTORE_URL = os.environ.get("VECTORSTORE_URL", "http://vectorstore:8001")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/distilbert-sst2")

app = FastAPI(title="Sentiment Analysis API")

model = SentimentModel(model_dir=MODEL_DIR)

class PredictRequest(BaseModel):
    texts: List[str]
    add_to_index: bool = False

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/predict")
def predict(req: PredictRequest):
    texts = [clean_text(t) for t in req.texts]
    preds, probs = model.predict_sentiment(texts)
    embeddings = model.embed_texts(texts)
    results = []
    for t, p, pr, emb in zip(texts, preds, probs, embeddings):
        label = int(p)
        scores = pr
        obj = {"text": t, "label": label, "probs": scores}
        results.append(obj)
        if req.add_to_index:
            # create id
            id_ = str(uuid.uuid4())
            payload = {"id": id_, "embedding": emb, "metadata": {"text": t, "label": label}}
            try:
                requests.post(f"{VECTORSTORE_URL}/add", json=payload, timeout=10)
            except Exception as e:
                # if vectorstore not reachable, continue but inform in response
                obj["index_error"] = str(e)
    return {"results": results}

@app.post("/search")
def search(req: QueryRequest):
    q = clean_text(req.query)
    emb = model.embed_texts([q])[0]
    payload = {"embedding": emb, "top_k": req.top_k}
    try:
        r = requests.post(f"{VECTORSTORE_URL}/query", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
