# Convenience targets
.PHONY: setup lint test run-notebook

setup:
	pip install -r requirements.txt
	pre-commit install

lint:
	pre-commit run --all-files

test:
	pytest -q

run-notebook:
	papermill notebooks/THz_TDS_AI_Classification_Demo.ipynb artifacts/nb_executed.ipynb
