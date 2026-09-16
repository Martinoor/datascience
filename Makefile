IMAGE ?= churn-explorer:local

.PHONY: install lint format test run sample docker-build docker-run clean

install:        ## Create .venv from uv.lock (exact versions)
	uv sync --locked

lint:           ## Ruff lint + format check + lockfile check
	uv lock --check
	uv run ruff check .
	uv run ruff format --check .

format:         ## Auto-fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

test:           ## Unit + app tests with coverage
	uv run pytest

run:            ## Start the app on http://localhost:8501
	uv run streamlit run app/streamlit_app.py

sample:         ## Write a synthetic dataset to data/sample_events.parquet
	uv run python -m churn_app.sample_data --users 500 --out data/sample_events.parquet

docker-build:   ## Build the container image
	docker build -t $(IMAGE) .

docker-run:     ## Run the image, mounting ./churn-prediction-25-26 as /data
	docker run --rm -p 8501:8501 -v "$(CURDIR)/churn-prediction-25-26:/data:ro" $(IMAGE)

clean:
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml junit.xml
