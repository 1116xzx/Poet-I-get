PYTHON ?= python

prepare:
	$(PYTHON) -m src.data.preprocess --download
	$(PYTHON) -m src.data.tokenizer

train-gru:
	$(PYTHON) -m src.engine.train --config configs/gru_base.yaml

train-lstm:
	$(PYTHON) -m src.engine.train --config configs/lstm_base.yaml

evaluate:
	$(PYTHON) -m src.engine.evaluate --checkpoint checkpoints/gru_best.pt

demo:
	$(PYTHON) -m src.engine.demo --checkpoint checkpoints/gru_best.pt

plot:
	$(PYTHON) -m src.utils.plotting --metrics runs/gru_base/metrics.csv --out_dir runs/gru_base

test:
	pytest -q
