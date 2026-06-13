# Poet-I-get

七言绝句条件生成系统，支持首句续写、藏头诗生成，以及成语格诗扩展模块。

项目主体使用字符级 GRU 完成课程要求的主任务，并在此基础上加入了：

- `baseline`：普通字符级 GRU
- `weighted`：对藏头关键位置加权的 GRU
- `structured`：加入句位标记和结构约束的 GRU
- `成积似涵`：基于成语库、Beam Search、GRU 评分和 Prefix Global BiGRU Scorer 的扩展模块

## 任务完成情况

本项目完成的课程主任务包括：

- 只使用七言绝句子集进行训练
- 支持首句续写
- 支持四字藏头诗生成
- 使用字符级 GRU 建模
- 实现多种采样策略：`stable`、`balanced`、`creative`
- 评测指标包含 `PPL` 和格式合规率
- 展示每种任务下的多组生成样例
- 输出训练曲线、指标表和实验报告

## 数据集

数据来源：

```text
https://dicalab-scu.github.io/nlp/post/ancient-poems-dataset/
```

处理规则：

- 只保留四句七字的七言绝句
- 保留原始标点信息用于训练
- 删除情绪、题材等额外标签

处理后统计：

- 严格七绝：`136631`
- 训练集：`122967`
- 验证集：`6831`
- 测试集：`6833`

## 目录说明

```text
configs/                训练配置
checkpoints/            训练好的模型权重
src/data/               数据预处理、词表、数据集
src/models/             GRU 和扩展评分器模型
src/engine/             训练、生成、评测、成语格搜索
src/metrics/            PPL、格式、押韵等指标
src/utils/              绘图和实验脚本
src/web/                Flask 前端
static/                 前端静态资源（如背景图）
runs/moxing/            三组主模型训练与评测结果
runs/duibi/             三模型对比图表
report/                 报告、导出脚本和成品文件
```

## 三组主模型

### 1. baseline

普通字符级 GRU，不使用句位标记。

配置文件：

```text
configs/gru_plain_baseline.yaml
```

权重文件：

```text
checkpoints/gru_plain_best.pt
```

### 2. weighted

在 baseline 基础上，对藏头位置增加 loss 权重，增强藏头控制能力。

配置文件：

```text
configs/gru_plain_weighted.yaml
```

权重文件：

```text
checkpoints/gru_plain_weighted_best.pt
```

### 3. structured

加入 `<L1><L2><L3><L4>` 句位标记，并在生成阶段使用结构约束。

配置文件：

```text
configs/gru_base.yaml
```

权重文件：

```text
checkpoints/gru_best.pt
```

主实验默认使用 `structured`。

## 主实验结果

三模型主对比结果：

| 模型 | 生成方式 | Test PPL | 格式合规率 | 藏头正确率 |
|---|---|---:|---:|---:|
| baseline | raw | 83.690 | 1.000 | 0.000 |
| weighted | raw | 78.264 | 1.000 | 0.930 |
| structured | constrained | 51.185 | 1.000 | 1.000 |

相关文件：

- `runs/duibi/biaoge/san_moxing_duibi.csv`
- `runs/duibi/tupian/san_moxing_ppl_duibi.png`
- `runs/duibi/tupian/san_moxing_zhuyao_duibi.png`
- `runs/duibi/tupian/san_moxing_zhiliang_duibi.png`

## 采样策略

系统使用三组采样参数：

- `stable`：`temperature = 0.7`
- `balanced`：`temperature = 0.9, top-k = 20`
- `creative`：`temperature = 1.1, top-p = 0.95`

九组合对比图：

- `runs/duibi/tupian/xuxie_moshi_jiu_zuhe.png`
- `runs/duibi/tupian/cangtou_moshi_jiu_zuhe.png`

## 成积似涵扩展模块

除课程主任务外，项目还包含一个成语格诗增强模块。

输入四字藏头后，系统会：

1. 固定首列为四个藏头字
2. 从成语库中抽取后六列候选成语
3. 用 Beam Search 逐列扩展候选字阵
4. 用训练好的 GRU 计算横向诗句 NLL
5. 结合重复惩罚、短语惩罚、风格分、押韵分进行重排序
6. 使用 Prefix Global BiGRU Scorer 对半成品字阵进行前缀一致性评分

相关代码：

- `src/engine/chengyu_grid.py`
- `src/engine/chengyu_global_beam_experiment.py`
- `src/models/global_prefix_bigru_scorer.py`
- `src/engine/global_prefix_score.py`

相关权重：

- `checkpoints/global_prefix_bigru_20e.pt`
- `checkpoints/global_bigru_scorer_20e.pt`

## 前端

前端基于 Flask，入口文件：

```text
src/web/app.py
```

模板文件：

```text
src/web/templates/index.html
```

背景图资源：

```text
static/beijing.png
```

启动方式：

```bash
python src/web/app.py
```

如果本地已经配了 PowerShell 快捷命令，也可以直接使用你自己的快捷启动方式。

## 训练与评测命令

### 数据处理

```bash
python -m src.data.preprocess --download
python -m src.data.tokenizer
```

### 训练主模型

```bash
python -m src.engine.train --config configs/gru_plain_baseline.yaml
python -m src.engine.train --config configs/gru_plain_weighted.yaml
python -m src.engine.train --config configs/gru_base.yaml
```

### 评测与样例导出

```bash
python -m src.engine.evaluate --checkpoint checkpoints/gru_best.pt --out runs/moxing/jiegou/evaluation.csv
python -m src.engine.demo --checkpoint checkpoints/gru_best.pt --out_dir runs/moxing/jiegou/demo
```

### 绘图

```bash
python -m src.utils.comparison_plot --comparison runs/duibi/biaoge/san_moxing_duibi.json --out_dir runs/duibi/tupian
python -m src.utils.mode_model_bar_plot
```

## 报告

当前仓库中已经包含多版报告与导出脚本。

推荐先看：

- `report/gru_report.md`
- `report/初版报告.md`
- `report/七言绝句生成实验报告.docx`
- `report/七言绝句条件生成系统实验报告_BP风格.docx`

报告生成脚本：

- `report/build_poem_report.py`
- `report/build_bp_style_poem_report.py`

## 当前默认建议

如果只展示课程主任务，建议使用：

- 主模型：`structured`
- 采样策略：`balanced`

如果展示扩展创新模块，建议同时展示：

- 成积似涵
- Prefix Global BiGRU Scorer
- 多候选重排序与押韵评分

