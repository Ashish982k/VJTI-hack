FROM python:3.11-slim
WORKDIR /app
# OpenCV (PaddleOCR) needs these shared libraries in slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[semantic,observability,ocr]'
ENV KB_DATA_DIR=/data HF_HOME=/models PADDLE_PDX_CACHE_HOME=/models/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
VOLUME ["/data", "/models"]
EXPOSE 8000
CMD ["uvicorn", "knowledge_base.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
