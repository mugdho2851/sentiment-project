# Sentiment Analysis + FAISS Vectorstore (Offline)

## Summary
This repo provides an offline-capable sentiment analysis pipeline with:
- transformer-based classification (DistilBERT)
- embedding generation and semantic indexing using FAISS (vectorstore service)
- FastAPI ML API
- Gradio UI

Everything runs locally via Docker Compose. **You must download the transformer model and tokenizer files to `models/distilbert-sst2/` before running.**

## Pre-download (offline preparation)
1. Create folder:
2. Download the transformer files (`pytorch_model.bin`, `config.json`, tokenizer files etc.) into the folder above.
- If you cannot use the internet from the target machine, download the files on another machine and copy them into `models/distilbert-sst2/`.
- The model files must be from a transformer saved locally (e.g., DistilBERT fine-tuned on SST-2). The code uses `transformers` `local_files_only=True` to ensure offline loads.
3. (Optional) Pre-download Python wheels for all packages listed in `requirements.txt` if you must install packages offline. Place wheel files and install with `pip install --no-index --find-links=./wheels -r requirements.txt`.

## Running locally (Docker)
1. Ensure Docker is installed.
2. (Optional) Build images:
If your machine has no internet, ensure `python:3.10-slim` base images and necessary wheels are available offline on the machine (you may need to `docker save`/`docker load` images).
3. Run:
4. Open UI:
- Gradio: http://localhost:7880
- API: http://localhost:8000
- Vectorstore: http://localhost:8001/status

## Running without Docker (direct, offline)
1. Create virtualenv and install packages offline (see requirements).
2. Start vectorstore:
3. Start API:
4. Start UI:
