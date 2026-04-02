.PHONY: setup test test-unit test-integration test-all test-infra-start test-infra-stop clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  setup              - Setup python environment, create venv and install dependencies"
	@echo "  test               - Run unit tests (no infra needed)"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests (requires test-infra-start first)"
	@echo "  test-all           - Run unit + integration tests"
	@echo "  test-infra-start   - Start test Redis on port 6380"
	@echo "  test-infra-stop    - Stop test Redis and clean test dirs"
	@echo "  clean              - Remove virtual environment and cached files"

# Setup the environment using standard venv
setup:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "\nEnvironment setup complete. To activate: source .venv/bin/activate"

# Run unit tests (no infra needed)
test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -v --ignore=tests/integration

# Run unit tests only
test-unit:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/unit/ tests/test_memory_system.py tests/test_redis_storage.py -v

# Run integration tests (requires Redis on port 6380)
test-integration:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/integration/ -v -m integration

# Run all tests
test-all: test-unit test-integration

# Start test infrastructure
test-infra-start:
	redis-server --port 6380 --daemonize yes --logfile /tmp/redis-test.log --save ""
	@echo "Test Redis started on port 6380 (log: /tmp/redis-test.log)"

# Stop test infrastructure and clean test data
test-infra-stop:
	redis-cli -p 6380 shutdown nosave || true
	@echo "Test Redis stopped"
	rm -rf test_memory_data/
	@echo "Test filesystem cleaned"

# Clean the workspace
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
