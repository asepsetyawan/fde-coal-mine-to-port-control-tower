.PHONY: help install seed train api frontend smoke clean all \
        frontend-install frontend-build \
        docker-build docker-up docker-down docker-logs docker-seed docker-smoke

# ---------------------------------------------------------------------------
# Local (uv + React)
# ---------------------------------------------------------------------------
help:
	@echo "Coal Mine-to-Port Control Tower"
	@echo ""
	@echo "  Local:"
	@echo "    make install          uv sync (Python backend)"
	@echo "    make frontend-install npm install (React UI)"
	@echo "    make seed             Simulate dataset into DB"
	@echo "    make train            Train AI models"
	@echo "    make api              FastAPI backend  → http://localhost:8000"
	@echo "    make frontend         React UI (dev)     → http://localhost:5173"
	@echo "    make smoke            End-to-end smoke test"
	@echo "    make all              install + frontend-install + seed + train"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build     Build images"
	@echo "    make docker-up        Start all services"
	@echo "    make docker-seed      Seed DB inside running stack"
	@echo "    make docker-smoke     Smoke-test inside running stack"
	@echo "    make docker-logs      Tail all service logs"
	@echo "    make docker-down      Stop and remove containers"

install:
	uv sync

frontend-install:
	cd frontend && npm install

frontend-build:
	cd frontend && npm run build

seed:
	uv run python -m control_tower.pipeline.seed

train:
	uv run python -m control_tower.ai.train

api:
	uv run uvicorn control_tower.api.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

smoke:
	uv run python -m control_tower.smoke

all: install frontend-install seed train

clean:
	rm -f data/*.db data/*.csv data/*.parquet models/*.joblib
	rm -rf frontend/dist frontend/node_modules

# ---------------------------------------------------------------------------
# Docker (compose v2)
# ---------------------------------------------------------------------------
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-seed:
	docker compose exec api python -m control_tower.pipeline.seed

docker-smoke:
	docker compose exec api python -m control_tower.smoke

docker-logs:
	docker compose logs -f

docker-down:
	docker compose down -v
