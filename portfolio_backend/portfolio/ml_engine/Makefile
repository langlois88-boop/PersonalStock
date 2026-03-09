.PHONY: train-stable train-penny train-crypto train-all test lint push-stable push-penny clean

PYTHON=.venv/bin/python

train-stable:
	$(PYTHON) -m portfolio.ml_engine.pipelines.stable_pipeline

train-penny:
	$(PYTHON) -m portfolio.ml_engine.pipelines.penny_pipeline

train-crypto:
	$(PYTHON) -m portfolio.ml_engine.pipelines.crypto_pipeline

train-all: train-stable train-penny train-crypto

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m ruff check portfolio/ml_engine/
	$(PYTHON) -m mypy portfolio/ml_engine/ --ignore-missing-imports

push-stable:
	AUTO_PUSH_MODEL=1 $(PYTHON) -m portfolio.ml_engine.pipelines.stable_pipeline

push-penny:
	AUTO_PUSH_MODEL=1 $(PYTHON) -m portfolio.ml_engine.pipelines.penny_pipeline

clean:
	find . -name "*.onnx.tmp" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
