FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 safety
COPY requirements.lock.txt ./
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.lock.txt
COPY --chown=safety:safety app ./app
COPY --chown=safety:safety paper/model_source ./paper/model_source
COPY --chown=safety:safety paper/model_weights ./paper/model_weights
COPY --chown=safety:safety paper/data_processing/llm_extract.py ./paper/data_processing/llm_extract.py
COPY --chown=safety:safety paper/processed_data/extracted_data/_all_extractions.json ./paper/processed_data/extracted_data/_all_extractions.json
COPY --chown=safety:safety paper/processed_data/extracted_text ./paper/processed_data/extracted_text
COPY --chown=safety:safety paper/original_results/comparison_results.json ./paper/original_results/comparison_results.json
USER safety
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "16"]
