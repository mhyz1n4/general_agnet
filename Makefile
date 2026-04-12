.PHONY: setup run dev dev-redis-start dev-redis-stop test test-unit test-integration test-llm test-benchmark test-all test-infra-start test-infra-stop clean help

DEV_REDIS_CONTAINER  := agent-redis-dev
TEST_REDIS_CONTAINER := agent-redis-test

# Default target
help:
	@echo "Dev commands:"
	@echo "  setup              - Create venv and install dependencies"
	@echo "  dev-redis-start    - Start Redis for local dev (Docker, port 6379)"
	@echo "  dev-redis-stop     - Stop and remove the dev Redis container"
	@echo "  run                - Run the agent (requires .env and dev Redis)"
	@echo "  dev                - dev-redis-start then run (one-shot)"
	@echo ""
	@echo "Test commands:"
	@echo "  test               - Run unit tests (no infra needed)"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests (requires test-infra-start first)"
	@echo "  test-llm           - Run LLM integration tests (requires running vLLM endpoint)"
	@echo "  test-benchmark     - Run latency benchmarks (P50/P95 targets)"
	@echo "  test-all           - Run unit + integration + LLM tests"
	@echo "  test-infra-start   - Start test Redis on port 6380 (Docker)"
	@echo "  test-infra-stop    - Stop test Redis and clean test dirs"
	@echo ""
	@echo "  clean              - Remove virtual environment and cached files"

# Setup the environment using standard venv
setup:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "\nEnvironment setup complete. To activate: source .venv/bin/activate"

# Start dev Redis (Docker)
dev-redis-start:
	@docker inspect $(DEV_REDIS_CONTAINER) >/dev/null 2>&1 \
		&& echo "Dev Redis already running ($(DEV_REDIS_CONTAINER))" \
		|| (docker run -d --name $(DEV_REDIS_CONTAINER) -p 6379:6379 redis:7-alpine \
			&& echo "Dev Redis started on localhost:6379")

# Stop dev Redis
dev-redis-stop:
	@docker stop $(DEV_REDIS_CONTAINER) && docker rm $(DEV_REDIS_CONTAINER) || true
	@echo "Dev Redis stopped"

# Run the agent loop
run: 
	@[ -f .env ] || (echo "ERROR: .env not found — copy .env.example to .env and set LLM_API_KEY" && exit 1)
	. .venv/bin/activate && PYTHONPATH=. python main.py

# Start dev Redis then run the agent (exits agent → Redis keeps running)
dev: dev-redis-start run

# Run unit tests (no infra needed)
test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -v --ignore=tests/integration

# Run unit tests only
test-unit:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/unit/ tests/test_memory_system.py tests/test_redis_storage.py -v

# Run integration tests (requires Redis on port 6380)
test-integration:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/integration/ -v -m integration

# Run LLM integration tests (requires live vLLM endpoint)
test-llm:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/integration/test_llm_memorize.py -v -m llm -s

# Run latency benchmarks
test-benchmark:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/benchmarks/ -v -m benchmark -s

# Run all tests
test-all: test-unit test-integration test-llm

# Start test infrastructure
test-infra-start:
	@docker inspect $(TEST_REDIS_CONTAINER) >/dev/null 2>&1 \
		&& echo "Test Redis already running ($(TEST_REDIS_CONTAINER))" \
		|| (docker run -d --name $(TEST_REDIS_CONTAINER) -p 6380:6379 redis:7-alpine \
			&& echo "Test Redis started on localhost:6380")

# Stop test infrastructure and clean test data
test-infra-stop:
	@docker stop $(TEST_REDIS_CONTAINER) && docker rm $(TEST_REDIS_CONTAINER) || true
	@echo "Test Redis stopped"
	rm -rf test_memory_data/
	@echo "Test filesystem cleaned"

# Clean the workspace
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
