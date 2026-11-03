.PHONY: install data train test lint clean

install:
	pip install -e ".[dev]"

data:
	python -m src.data.synthetic

train:
	python -m pipelines.train

test:
	pytest -v

lint:
	ruff check src tests pipelines

clean:
	rm -rf .pytest_cache .ruff_cache mlruns data/interim data/processed
