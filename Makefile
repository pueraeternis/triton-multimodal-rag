.PHONY: help export-models up down init-qdrant smoke-test test client check-config

UV := uv run

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

export-models: ## Export YOLO ONNX model (QUICKSTART step 3)
	$(UV) scripts/export_yolo.py

up: ## Start Qdrant and Triton containers (QUICKSTART steps 4–5)
	docker compose up -d qdrant
	docker compose up -d --build triton

down: ## Stop all compose services
	docker compose down

init-qdrant: ## Initialize Qdrant collection and upload documents (QUICKSTART step 4)
	docker compose up -d qdrant
	$(UV) scripts/init_qdrant.py

smoke-test: ## Run smoke validation (offline by default; use MODE=online or MODE=full)
	$(UV) scripts/smoke_test.py $(if $(MODE),--$(MODE),)

test: ## Run CPU-only pytest suite
	$(UV) pytest

client: ## Run inference client with default test image and query (QUICKSTART step 6)
	$(UV) client.py \
		--image data/test_image.jpg \
		--query "Red status LED is blinking continuously on my Router. What to do?"

check-config: ## Verify .env.example matches docs/CONFIGURATION.md
	$(UV) pytest tests/test_configuration.py -q
