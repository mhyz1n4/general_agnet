.PHONY: setup clean help

# Default target
help:
	@echo "Available commands:"
	@echo "  setup    - Setup python environment using uv, create venv and install dependencies"
	@echo "  test     - Run tests using pytest"
	@echo "  clean    - Remove virtual environment and cached files"

# Setup the environment using standard venv
setup:
	@echo "Creating virtual environment and installing dependencies..."
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "\nEnvironment setup complete. To activate: source .venv/bin/activate"

# Run tests
test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/

# Clean the workspace
clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +
