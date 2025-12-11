FROM python:3.12-slim
WORKDIR /app

COPY ./wheels /wheels
COPY ./ui_service/app/requirements.txt /app/requirements.txt

RUN pip install --no-index --find-links=/wheels -r /app/requirements.txt

COPY ./ui_service/app /app

EXPOSE 7880
CMD ["python", "gradio_app.py"]
