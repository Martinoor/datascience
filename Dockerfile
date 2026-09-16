# syntax=docker/dockerfile:1.7
#
# Churn Explorer — Streamlit app image.
#
#   docker build -t churn-explorer .
#   docker run --rm -p 8501:8501 churn-explorer
#   docker run --rm -p 8501:8501 -v "$PWD/churn-prediction-25-26:/data:ro" churn-explorer
#
# Base images are pinned by digest (kept fresh by Dependabot) and Python
# dependencies are installed from uv.lock with --frozen, so rebuilding the same
# commit produces the same environment.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.15@sha256:62f8c047d0a0e9ece6b53fc63df902585a67a47a7f318ddec4a37db586edc8e3

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src
# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="churn-explorer" \
      org.opencontainers.image.description="Streamlit explorer and churn model for streaming-service event logs" \
      org.opencontainers.image.source="https://github.com/Martinoor/datascience"

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app \
 && mkdir -p /data /app \
 && chown app:app /data /app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app app ./app

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

USER app
EXPOSE 8501
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py"]
