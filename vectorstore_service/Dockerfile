FROM python:3.12-slim
WORKDIR /app

COPY ./wheels /wheels
COPY ./vectorstore_service/app/requirements.txt /app/requirements.txt

RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt

COPY ./vectorstore_service/app /app

EXPOSE 8001
CMD ["gunicorn", "--bind", "0.0.0.0:8001", "vectorstore:app", "--workers", "1", "--threads", "2"]
