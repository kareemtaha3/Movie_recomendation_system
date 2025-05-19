.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3 test setup help

.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help: ## Show this help message
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

setup: ## Set up the development environment
	conda env create -f environment.yml

# Data management
data: ## Run data processing pipeline
	dvc repro prepare_data

train: ## Train the model
	dvc repro train_model

evaluate: ## Evaluate the model
	dvc repro evaluate_model

run_all: ## Run the complete pipeline
	dvc repro

# Code quality
lint: ## Check style with flake8, black, and isort
	flake8 src tests
	black --check src tests
	isort --check-only --profile black src tests

format: ## Format code with black and isort
	black src tests
	isort --profile black src tests

# Testing
test: ## Run tests
	pytest --cov=src tests/

# Cleaning
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/

# Documentation
docs: ## Generate Sphinx HTML documentation
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html