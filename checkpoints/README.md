# Checkpoints

当前保留的可复现模型权重：

- `gru_plain_best.pt`: baseline GRU
- `gru_plain_weighted_best.pt`: weighted GRU
- `gru_best.pt`: structured GRU
- `global_prefix_bigru_20e.pt`: 成积似涵使用的 Prefix Global BiGRU Scorer

如果重新训练，会按 `configs/` 中对应配置覆盖或生成新的 checkpoint。
