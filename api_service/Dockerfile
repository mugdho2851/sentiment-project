FROM python:3.12-slim
WORKDIR /app

# wheels are located in project root
COPY ./wheels /wheels

# requirements are inside api_service/app
COPY ./api_service/app/requirements.txt /app/requirements.txt

RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt

# now copy application code
COPY ./api_service/app /app

ENV MODEL_DIR=/models/distilbert-sst2
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
