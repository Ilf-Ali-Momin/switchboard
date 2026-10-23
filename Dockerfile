# Switchboard routing proxy. CPU only: the classifier is small enough for CPU inference.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv

# Dependencies from the lock file. torch comes from the CPU wheel index; the CUDA packages a
# Linux torch would pull in from PyPI are left out.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/requirements.txt \
    && TORCH_VERSION=$(sed -n 's/^torch==\([^ ;]*\).*/\1/p' /tmp/requirements.txt) \
    && pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E '^(torch==|nvidia-|triton|cuda-)' /tmp/requirements.txt > /tmp/rest.txt \
    && pip install -r /tmp/rest.txt \
    && rm /tmp/requirements.txt /tmp/rest.txt

COPY config ./config
COPY switchboard ./switchboard

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/models /app/results \
    && chown app /app/data
USER app

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/healthz')"
CMD ["uvicorn", "switchboard.serve.app:app", "--host", "0.0.0.0", "--port", "8100"]
