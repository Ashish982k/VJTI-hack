FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[semantic,observability]'
ENV KB_DATA_DIR=/data HF_HOME=/models
VOLUME ["/data", "/models"]
EXPOSE 8000
CMD ["uvicorn", "knowledge_base.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
