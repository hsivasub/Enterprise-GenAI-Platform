.PHONY: help install test format lint run docker-up docker-down clean

help:
	@echo "Enterprise GenAI Platform - Developer Commands"
	@echo "----------------------------------------------"
	@echo "make install    - Install dependencies"
	@echo "make test       - Run all tests with pytest"
	@echo "make format     - Format code with black and isort"
	@echo "make lint       - Run static analysis with flake8"
	@echo "make run        - Run FastAPI server locally"
	@echo "make dash       - Run Streamlit dashboard locally"
	@echo "make docker-up  - Start full infrastructure via Docker Compose"
	@echo "make docker-down- Stop Docker infrastructure"
	@echo "make clean      - Remove __pycache__ and generated data/logs"

install:
	pip install -r requirements.txt
	pip install black isort flake8 pytest pytest-cov

test:
	pytest tests/ -v

format:
	black .
	isort .

lint:
	flake8 .

run:
	python -m uvicorn api.main:app --reload --port 8000

dash:
	streamlit run dashboards/streamlit_app.py

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/faiss_index/*
	rm -f data/analytical.duckdb*
	rm -rf mlruns/*
	rm -rf logs/*
