FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

# libgl/glib for opencv (pulled in by aiortc's WebRTC stack)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra db --extra jobs --extra telemetry

COPY parley ./parley
COPY evals ./evals
COPY examples ./examples
RUN uv sync --frozen --no-dev --extra db --extra jobs --extra telemetry

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "parley", "serve", "--host", "0.0.0.0", "--port", "8000"]
