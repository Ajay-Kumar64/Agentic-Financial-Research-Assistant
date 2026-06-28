.PHONY: run ui mcp test eval eval-ragas docker docker-down clean trace adversarial redis redis-cli redis-logs

# =============================================================================
# DEVELOPMENT
# =============================================================================

# Run API server (development with reload)
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit UI
ui:
	streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0

# Run MCP server (stdio transport)
mcp:
	python -m mcp_server.run

# =============================================================================
# TESTING
# =============================================================================

# Run all tests
test:
	python -m pytest tests/ -v

# Run legacy evaluation suite (golden traces + adversarial)
eval:
	python -m evaluation.run_eval

# Run RAGAS evaluation on golden traces
eval-ragas:
	python -m eval.ragas_eval --all

# Run RAGAS on a single trace (usage: make eval-ragas-single ID=ST-01)
eval-ragas-single:
	python -m eval.ragas_eval --trace-id $(ID)

# Run single trace (usage: make trace ID=ST-01)
trace:
	python tests/test_single_trace.py $(ID)

# Run adversarial tests only
adversarial:
	python tests/test_adversarial.py

# =============================================================================
# REDIS
# =============================================================================

# Start Redis via Docker (for local development)
redis:
	docker run -d --name redis-finagent -p 6379:6379 redis:7-alpine \
		--maxmemory 256mb --maxmemory-policy allkeys-lru || docker start redis-finagent
	@echo "Redis running at localhost:6379"

# Stop Redis container
redis-stop:
	docker stop redis-finagent || true

# Redis CLI
redis-cli:
	docker exec -it redis-finagent redis-cli

# Redis logs
redis-logs:
	docker logs redis-finagent -f

# =============================================================================
# DOCKER COMPOSE (Full Stack)
# =============================================================================

# Start all services (API + UI + MCP + Redis)
up:
	docker compose up --build

# Start in detached mode
up-d:
	docker compose up --build -d

# Stop all services
down:
	docker compose down

# Stop and remove volumes
down-v:
	docker compose down -v

# View logs
logs:
	docker compose logs -f

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	rm -rf logs/*.json eval/results/*.json __pycache__ .pytest_cache .mypy_cache