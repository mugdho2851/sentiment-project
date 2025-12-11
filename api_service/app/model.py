# api_service/app/model.py
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/distilbert-sst2")
SENTIMENT_MODEL_NAME = MODEL_DIR
EMBED_MODEL_NAME = MODEL_DIR  # we will use same model's hidden states for embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentModel:
    def __init__(self, model_dir=SENTIMENT_MODEL_NAME):
        # tokenizer + classification model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
        # For embeddings, use AutoModel (returns hidden states). If classification model supports .distilbert, we can reuse.
        try:
            self.embed_model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        except Exception:
            # fallback: use the encoder part from cls_model if available
            self.embed_model = None
        self.max_length = 256

    def predict_sentiment(self, texts):
        # returns labels and probabilities
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = self.cls_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1).tolist()
        return preds, probs.tolist()

    def embed_texts(self, texts):
        """
        Create sentence-level embeddings using the last hidden state mean pool.
        """
        if self.embed_model is None:
            # fallback to using cls_model's base encoder if present
            base = getattr(self.cls_model, "distilbert", None)
            if base is None:
                raise RuntimeError("No embed_model available. Provide a model with encoder.")
            model = base
        else:
            model = self.embed_model

        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            mean_pooled = summed / counts
            embeddings = mean_pooled.cpu().numpy()
        return embeddings.tolist()
