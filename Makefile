PYTHON ?= python

prepare:
	$(PYTHON) -m src.data.preprocess --download
	$(PYTHON) -m src.data.prepare_chengyu

train-baseline:
	$(PYTHON) -m src.engine.train --config configs/gru_plain_baseline.yaml

train-weighted:
	$(PYTHON) -m src.engine.train --config configs/gru_plain_weighted.yaml

train-structured:
	$(PYTHON) -m src.engine.train --config configs/gru_base.yaml

train-prefix:
	$(PYTHON) -m src.engine.train_global_prefix_scorer --config configs/global_prefix_bigru_20e.yaml

evaluate-structured:
	$(PYTHON) -m src.engine.evaluate --checkpoint checkpoints/gru_best.pt --out runs/moxing/jiegou/evaluation.csv

plot:
	$(PYTHON) -m src.utils.comparison_plot --comparison runs/duibi/biaoge/san_moxing_duibi.json --out_dir runs/duibi/tupian
	$(PYTHON) -m src.utils.mode_model_bar_plot
	$(PYTHON) -m src.utils.model_strategy_bar_plot

web:
	$(PYTHON) src/web/app.py

test:
	$(PYTHON) -m pytest tests/test_smoke.py -p no:cacheprovider
