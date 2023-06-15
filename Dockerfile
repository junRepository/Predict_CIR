FROM python:3.10
COPY /app /app
COPY requirements.txt requirements.txt

RUN api-get update && \
    python -m pip install --upgrade pip && \
    pip install -r requirement.txt

CMD uvicorn api:app --host=0.0.0.0