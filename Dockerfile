# Stage 1: Build virtual environment and application
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY wyoming_vietnamese/ ./wyoming_vietnamese/

# Stage 2: Minimal runtime image
FROM python:3.14-slim-trixie

WORKDIR /app

RUN groupadd -g 1000 app && \
    useradd -u 1000 -g app -d /app -s /bin/bash app

RUN apt-get update && \
    apt-get install --yes --no-install-recommends espeak-ng-data libgomp1 tini && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/.cache /app/models && \
    chown -R app:app /app

COPY --from=builder --chown=app:app /app/.venv/ /app/.venv/
COPY --from=builder --chown=app:app /app/wyoming_vietnamese/ /app/wyoming_vietnamese/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/app/.cache" \
    HF_HUB_DISABLE_TELEMETRY=1

USER app

EXPOSE 10300

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
  CMD ["python", "-m", "wyoming_vietnamese.healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "wyoming_vietnamese"]
