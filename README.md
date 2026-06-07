# Poet-I-get — 诗人我持，七绝诗句生成模型

本项目严格完成课程要求：只使用四句、每句七字的七言绝句；支持”首句续写”和”藏头诗”；实现 temperature、top-k、top-p；输出 Test PPL、格式合规率，并额外统计 Distinct-2、重复率与训练集近邻复写风险。

## 设计亮点

- **统一多任务模型**：一个 checkpoint 同时支持自由建模、首句续写、藏头诗，不为两个任务分别训练模型。
- **显式结构 token**：训练序列中加入 `<L1> ... <L4>`，模型知道句界在哪里。
- **受约束解码**：每句严格生成 7 个汉字；藏头位置硬约束写入，因此结构稳定，不依赖后处理补救。
- **增量推理缓存**：生成时复用 GRU/LSTM 隐状态，每生成一个字只执行一步 RNN，不重复计算整个前缀。
- **公平采样比较**：同一 checkpoint 的 PPL 固定；采样参数只比较格式、多样性、重复和主观质量。
- **轻量复写检测**：用倒排 4-gram Jaccard 索引检查生成结果是否过度贴近训练诗。

## 目录

```text
configs/                 GRU/LSTM 配置
src/data/                下载、清洗、词表、Dataset
src/models/              字符级 GRU/LSTM
src/engine/              训练、评测、生成、样例导出
src/metrics/             格式、多样性、复写风险
src/utils/               公共工具、训练曲线
report/                  报告模板
```

## 运行步骤

```bash
pip install -r requirements.txt

# 1. 下载并严格筛选七绝
python -m src.data.preprocess --download

# 2. 只用训练集构建字符表
python -m src.data.tokenizer

# 3. 训练 GRU 主模型
python -m src.engine.train --config configs/gru_base.yaml

# 4. 画训练曲线
python -m src.utils.plotting \
  --metrics runs/gru_base/metrics.csv \
  --out_dir runs/gru_base

# 5. 自动评测三组采样参数
python -m src.engine.evaluate \
  --checkpoint checkpoints/gru_best.pt \
  --out runs/gru_base/evaluation.csv

# 6. 导出报告所需的 5 组首句续写 + 5 组藏头诗样例
python -m src.engine.demo \
  --checkpoint checkpoints/gru_best.pt \
  --out_dir runs/gru_base/demo
```

单条生成：

```bash
python -m src.engine.generate \
  --checkpoint checkpoints/gru_best.pt \
  --mode continue \
  --prompt 春风又过江南岸 \
  --temperature 0.9 --top_k 20

python -m src.engine.generate \
  --checkpoint checkpoints/gru_best.pt \
  --mode acrostic \
  --prompt 春江花月 \
  --temperature 0.9 --top_k 20
```

## 训练建议

首次跑通可将 `configs/gru_base.yaml` 中的 `epochs` 改成 `2`，确认完整流程无误后再改回 `30`。显存不足时，将 `batch_size` 从 `256` 改为 `128`。正式报告主表建议对比 GRU 与 LSTM；采样策略表建议对比：`T=0.7`、`T=0.9 + top-k=20`、`T=1.1 + top-p=0.95`。

## checkpoint 说明

训练完成后，最优权重会自动保存为 `checkpoints/gru_best.pt` 或 `checkpoints/lstm_best.pt`。checkpoint 中已经包含模型配置与词表，不需要额外拼接文件即可生成。
