# Gunakan base image Python yang ringan
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY prometheus_exporter.py .

CMD ["uvicorn", "prometheus_exporter:app", "--host", "0.0.0.0", "--port", "8001"]