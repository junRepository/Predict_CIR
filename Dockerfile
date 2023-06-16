FROM python:3.10
COPY app app
COPY models models
COPY requirements.txt requirements.txt

RUN apt-get update && \
    python -m pip install --upgrade pip && \
    pip install -r requirements.txt

CMD uvicorn app.api:app --host=0.0.0.0