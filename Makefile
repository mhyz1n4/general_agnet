.PHONY: setup run dev test test-unit test-integration test-llm test-benchmark test-eval test-all clean help

# Default target
help:
	@echo "Dev commands:"
	@echo "  setup              - Install dependencies via uv"
	@echo "  run                - Run the agent (requires .env)"
	@echo "  dev                - Alias for run"
	@echo ""
	@echo "Test commands:"
	@echo "  test               - Run unit tests"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests"
	@echo "  test-llm           - Run LLM integration tests (requires a live LLM endpoint)"
	@echo "  test-benchmark     - Run latency benchmarks (P50/P95 targets)"
	@echo "  test-eval          - Run golden-fixture eval harness (requires live LLM)"
	@echo "  test-all           - Run unit + integration + LLM tests"
	@echo ""
	@echo "  clean              - Remove virtual environment and cached files"

# Install dependencies via uv.
setup:
	@echo "Installing dependencies via uv..."
	uv pip install -r requirements.txt
	@echo "\nSetup complete."

# Run the agent loop
run:
	@[ -f .env ] || (echo "ERROR: .env not found — copy .env.example to .env and set LLM_API_KEY" && exit 1)
	uv run --active python main.py

# Dev alias — V1.2 uses ReMeLight's filesystem store, so no Redis is needed.
dev: run

# Run unit tests
test:
	uv run --active pytest tests/ -v --ignore=tests/integration

# Run unit tests only
test-unit:
	uv run --active pytest tests/unit/ -v

# Run integration tests
test-integration:
	uv run --active pytest tests/integration/ -v -m integration

# Run LLM integration tests (requires live vLLM endpoint)
test-llm:
	uv run --active pytest tests/integration/test_llm_memorize.py -v -m llm -s

# Run latency benchmarks
test-benchmark:
	uv run --active pytest tests/benchmarks/ -v -m benchmark -s

# Run the golden-fixture eval harness (requires live LLM endpoint)
test-eval:
	uv run --active pytest tests/eval/ -v -m eval -s

# Run all tests
test-all: test-unit test-integration test-llm

# Clean the workspace
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
