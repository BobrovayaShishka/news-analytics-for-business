FROM python:3.12-slim

WORKDIR /app

COPY requirements-news.txt .
RUN pip install --no-cache-dir -r requirements-news.txt

COPY . .

ENV ALFA_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "pipelines.news", "ingest", "--loop"]
