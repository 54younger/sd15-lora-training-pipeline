FROM python:3.12-slim-bookworm AS runtime
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=300
ARG PIP_RETRIES=10
ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT} PIP_RETRIES=${PIP_RETRIES} \
    LORA_DATA_DIR=/data HF_HOME=/models HF_HUB_DISABLE_TELEMETRY=1
RUN useradd --create-home --uid 10001 pipeline \
    && mkdir -p /data /models /app \
    && chown -R pipeline:pipeline /data /models /app
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install torch==2.8.0 torchvision==0.23.0 --index-url ${TORCH_INDEX_URL}
COPY pyproject.toml ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url ${PIP_INDEX_URL} .
USER pipeline
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live',timeout=3)"
CMD ["lora-pipeline", "api", "--host", "0.0.0.0"]
