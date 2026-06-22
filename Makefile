.PHONY: run ui mcp test eval docker docker-down clean trace adversarial

# Run API server (development with reload)
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit UI
ui:
	streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0

# Run MCP server (stdio transport)
mcp:
	python -m mcp_server.run

# Run all tests
test:
	python -m pytest tests/ -v

# Run evaluation suite (golden traces + adversarial)
eval:
	python -m eval.run_eval

# Run single trace (usage: make trace ID=ST-01)
trace:
	python tests/test_single_trace.py $(ID)

# Run adversarial tests only
adversarial:
	python tests/test_adversarial.py

# Docker: build and start all services
docker:
	docker compose up --build

# Docker: stop all services
docker-down:
	docker compose down

# Clean generated files
clean:
	rm -rf logs/*.json eval/results/*.json __pycache__ .pytest_cache
