.PHONY: install test lint serve compose offline report

install:
	python3 -m pip install -e ".[dev]"

test:
	PYTHONPATH=src python3 -m pytest

lint:
	ruff check src tests

serve:
	PYTHONPATH=src uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000

compose:
	docker compose up --build

offline:
	PYTHONPATH=src python3 -m ranking.run

report:
	PYTHONPATH=src python3 -m evaluation.report
