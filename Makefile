# MLOps Makefile for GenAI Infrastructure

.PHONY: install format lint test docker-build clean

install:
	pip install -r requirements.txt

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/

test:
	pytest tests/

docker-build:
	docker build -t genai-hpc-env:latest .

clean:
	rm -rf __pycache__ .pytest_cache outputs/
