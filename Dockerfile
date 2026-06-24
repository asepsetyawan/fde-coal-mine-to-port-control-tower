# ── Stage 1: dependency installer ──────────────────────────────────────────
# Uses the official uv image so we get the exact same resolver as locally.
FROM ghcr.io/astral-sh/uv:0.7.21-python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy only the files uv needs to resolve & install before the source code.
# This keeps the layer cached on every build that doesn't change dependencies.
COPY pyproject.toml uv.lock ./

# Install dependencies into an isolated venv inside /app/.venv.
# --frozen  : use the lockfile exactly (no re-resolution)
# --no-dev  : skip any dev extras if added later
# --no-install-project : install deps only; we install the project in the next copy
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself (editable-like, no network).
COPY src ./src
RUN uv sync --frozen --no-dev


# ── Stage 2: runtime base ───────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

WORKDIR /app

# Create runtime directories that will be mounted as volumes in compose.
RUN mkdir -p data models

# Copy the pre-built venv and source from the builder stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src   /app/src

# Put the venv on PATH so `python` / `uvicorn` resolve directly.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATABASE_URL="sqlite:////app/data/control_tower.db"

# ── Stage 3a: api ────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000
CMD ["uvicorn", "control_tower.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 3b: React frontend (build) ────────────────────────────────────────
FROM node:22-alpine AS frontend-build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=/api
RUN npm run build

# ── Stage 3c: React frontend (serve via nginx) ───────────────────────────────
FROM nginx:1.27-alpine AS frontend

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /app/dist /usr/share/nginx/html

EXPOSE 80
