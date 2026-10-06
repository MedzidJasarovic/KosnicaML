# Build iz KosnicaML/ foldera (kontekst mora da sadrži models/):
#   docker build -f service/Dockerfile -t kosnica-ml .
#   docker run -p 8000:8000 kosnica-ml
FROM python:3.12-slim

WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

COPY service/__init__.py service/main.py service/inference.py service/preprocessing.py service/
COPY models/bee_health.onnx models/bee_health.meta.json models/

EXPOSE 8000
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
