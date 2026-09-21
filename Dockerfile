# Stage 1: Build virtual environment and application
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY wyoming_vietnamese/__init__.py wyoming_vietnamese/const.py wyoming_vietnamese/cpu.py ./wyoming_vietnamese/
COPY tools/build_zerotts.py ./tools/build_zerotts.py

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,id=aptlib,sharing=locked <<EOF
set -eux
apt-get update && apt-get install --yes --no-install-recommends build-essential cmake git
uv sync --locked --no-install-project --no-default-groups
mkdir -p /app/lib
python3 /app/tools/build_zerotts.py /app/lib --all-variants
EOF

COPY wyoming_vietnamese/ ./wyoming_vietnamese/

# Stage 2: Minimal runtime image
FROM python:3.14-slim-trixie AS runtime

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,id=aptlib,sharing=locked <<EOF
set -eux
groupadd -g 1000 app && useradd -u 1000 -g app -d /app -s /usr/sbin/nologin app
apt-get update && apt-get install --yes --no-install-recommends espeak-ng-data libgomp1 tini
mkdir -p /app/.cache /app/models /app/lib && chown -R app:app /app
EOF

COPY --from=builder --chown=app:app /app/.venv/ /app/.venv/
COPY --from=builder --chown=app:app /app/lib/ /app/lib/
COPY --from=builder --chown=app:app /app/wyoming_vietnamese/ /app/wyoming_vietnamese/

ENV PATH="/app/.venv/bin:$PATH" \
    LD_LIBRARY_PATH="/app/lib" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/app/.cache" \
    HF_HUB_DISABLE_TELEMETRY=1

USER app

EXPOSE 10300

HEALTHCHECK --interval=30s \
    --timeout=15s \
    --start-period=10m \
    --retries=3 \
    CMD ["python", "-m", "wyoming_vietnamese.healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python", "-m", "wyoming_vietnamese"]
