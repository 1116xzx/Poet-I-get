# Poet-I-get

Poet-I-get 是一个基于字符级 GRU 的七言绝句条件生成系统。项目完成了课程要求中的首句续写、四字藏头诗生成、采样策略比较、PPL 与格式合规率评测，并额外实现了一个带展示效果的创新模块：**成积似涵**，即“藏头字 + 竖向成语格”的七言绝句生成。

## 1. 项目任务

本项目面向七言绝句生成，输入和输出形式如下：

- 首句续写：输入 7 个汉字作为第一句，模型生成后三句。
- 藏头诗：输入 4 个汉字，分别作为四句首字，模型生成完整四句。
- 输出格式：四句，每句 7 个汉字，不含标点。

数据集来自：

```text
https://dicalab-scu.github.io/nlp/post/ancient-poems-dataset/
```

项目只保留七言绝句子集，即四句、每句七字的诗。处理后的数据位于：

```text
data/processed/qijue/
```

主要文件：

| 文件 | 说明 |
|---|---|
| `train.jsonl` | 训练集 |
| `valid.jsonl` | 验证集 |
| `test.jsonl` | 测试集 |
| `vocab.json` | structured 模型使用的词表 |
| `vocab_plain.json` | baseline / weighted 模型使用的词表 |
| `stats.json` | 数据统计信息 |

## 2. 模型设计

项目最终保留三组 GRU 模型：

| 模型 | 配置文件 | checkpoint | 说明 |
|---|---|---|---|
| baseline | `configs/gru_plain_baseline.yaml` | `checkpoints/gru_plain_best.pt` | 普通字符级 GRU，多任务训练，不加入句位标记 |
| weighted | `configs/gru_plain_weighted.yaml` | `checkpoints/gru_plain_weighted_best.pt` | 在 baseline 基础上，对藏头任务中的四个句首位置加大 loss 权重 |
| structured | `configs/gru_base.yaml` | `checkpoints/gru_best.pt` | 加入 `<L1><L2><L3><L4>` 句位标记，推理时使用结构约束 |

三个模型的关系是递进的：

1. baseline 作为基础模型，验证普通字符级 GRU 是否能学到七言绝句的语言和格式。
2. weighted 在不加入句位标记的情况下，通过提高藏头句首位置的训练权重，增强藏头控制能力。
3. structured 在训练序列中显式加入句位标记，并在推理阶段约束每句 7 字、藏头字固定到句首，使格式和控制更稳定。

## 3. 训练序列

模型采用字符级语言模型训练方式，即根据前文预测下一个字符。

以藏头诗为例，baseline / weighted 的训练序列类似：

```text
<BOS> <TASK_ACRO> <SEP> 春 江 花 月 <SEP>
春......，江......。花......，月......。 <EOS>
```

structured 模型额外加入句位标记：

```text
<BOS> <TASK_ACRO> <SEP> 春 江 花 月 <SEP>
<L1> 春......，
<L2> 江......。
<L3> 花......，
<L4> 月......。 <EOS>
```

句位标记的作用是告诉模型当前正在生成第几句，使模型更容易学习四句结构。

## 4. 采样策略

项目最终保留 temperature 采样，并设置三档：

| 策略 | 参数 | 特点 |
|---|---|---|
| stable | `temperature=0.7` | 更保守，重复风险较低，但变化少 |
| balanced | `temperature=1.0` | 默认采样强度 |
| creative | `temperature=1.3` | 更随机，结果更有变化，但格式和控制可能下降 |

说明：PPL 是在测试集上计算的模型指标，不依赖采样策略；格式率和藏头率来自生成样例，因此会受到采样策略影响。

## 5. 评测指标

项目主要使用以下指标：

| 指标 | 含义 |
|---|---|
| Test PPL | 测试集困惑度，越低表示模型越能预测测试集中的古诗字符 |
| 格式合规率 | 生成结果是否满足四句、每句七字 |
| 藏头正确率 | 四句首字是否与输入的四个藏头字一致 |

PPL 的计算公式为：

```text
PPL = exp(平均交叉熵 loss)
```

也就是说，模型在测试集上预测下一个字符越准确，loss 越低，PPL 也越低。

## 6. 最终实验结果

最终结果文件位于：

```text
runs/duibi/biaoge/
runs/duibi/tupian/
```

三模型主对比表：

```text
runs/duibi/biaoge/san_moxing_duibi.csv
runs/duibi/biaoge/san_moxing_duibi.json
runs/duibi/biaoge/san_moxing_duibi.md
```

续写和藏头模式的三模型、三采样策略对比：

```text
runs/duibi/biaoge/xuxie_moshi_jiu_zuhe.csv
runs/duibi/biaoge/cangtou_moshi_jiu_zuhe.csv
```

训练曲线和单模型评测位于：

```text
runs/moxing/jichu/      baseline
runs/moxing/jiaquan/    weighted
runs/moxing/jiegou/     structured
```

每个模型目录中保留：

| 文件 | 说明 |
|---|---|
| `metrics.csv` | 训练过程中的 train loss、train PPL、val loss、val PPL |
| `evaluation.csv` | 测试集评测结果 |
| `evaluation.json` | 测试集评测结果 JSON 版 |
| `train_ppl_curve.png` | 训练集 PPL 曲线 |
| `val_ppl_curve.png` | 验证集 PPL 曲线 |

## 7. 创新点

### 7.1 藏头位置加权训练

weighted 模型对藏头任务中的四个句首字设置更高的 loss 权重。普通训练时每个字符的重要性相同，但藏头诗最关键的是四句句首字是否正确。因此该模型在训练阶段让句首藏头位置受到更大惩罚，从而增强模型对藏头条件的学习。

### 7.2 句位标记与结构约束

structured 模型在训练序列中加入 `<L1><L2><L3><L4>`，让模型显式感知当前句子位置。推理时再配合结构约束，保证每句 7 字，并在藏头模式下固定四句首字。该方法使格式合规率和藏头正确率更稳定。

需要说明的是，结构约束不是训练模型本身的参数变化，而是推理阶段的解码策略。报告中应表述为：structured 模型结合结构化训练和约束解码，提高了格式和藏头控制稳定性。

### 7.3 押韵评分重排序

前端支持押韵优化。系统会一次生成多首候选诗，再根据押韵规则进行重排序，优先返回韵脚更自然的结果。

押韵评分主要考虑：

- 第 2 句和第 4 句是否押韵。
- 第 1 句是否入韵。
- 第 3 句是否避开主韵。
- 声调一致性作为小幅奖励。

对于 baseline 和 weighted，押韵优化只是“多候选生成 + 押韵分重排”，不会偷偷切换模型；对于 structured，开启押韵优化时可以进一步使用结构化押韵约束。

### 7.4 成积似涵：成语格藏头诗增强模块

“成积似涵”是项目的扩展展示模块。它面向四字藏头输入，将诗表示为一个 `4 x 7` 字阵：

```text
藏 ? ? ? ? ? ?
头 ? ? ? ? ? ?
诗 ? ? ? ? ? ?
字 ? ? ? ? ? ?
```

第一列固定为用户输入的四个藏头字，后六列从成语库中选择四字成语。这样横向读是四句七言诗，纵向读是藏头字和六个四字成语。

生成过程分两步：

1. Beam Search 中途筛选半成品  
   每加入一列成语，就形成一个半成品字阵。系统使用 Prefix Global BiGRU Scorer 判断当前半成品整体是否协调，同时结合重复惩罚、短语惩罚和风格惩罚，只保留较好的候选路径继续扩展。

2. 完整候选最终重排序  
   当字阵填满 7 列后，系统使用训练好的 GRU 计算横向四句诗的 NLL，并结合重复惩罚、短语惩罚、风格惩罚和押韵分进行综合排序，选出最终结果。

这里有两个模型分工：

| 模型 | 作用 |
|---|---|
| GRU 七绝语言模型 | 判断完整候选横向读起来是否像七言绝句 |
| Prefix Global BiGRU Scorer | 在 Beam Search 中途判断半成品字阵是否协调 |

相关代码：

| 文件 | 作用 |
|---|---|
| `src/engine/chengyu_grid.py` | 成语库读取、候选打分、基础成语格搜索 |
| `src/engine/chengyu_global_beam_experiment.py` | Prefix Global BiGRU 引导的 Beam Search |
| `src/engine/global_prefix_score.py` | Prefix Global BiGRU 评分接口 |
| `src/models/global_prefix_bigru_scorer.py` | Prefix Global BiGRU 模型 |
| `src/data/global_prefix_scorer_dataset.py` | Prefix Global BiGRU 训练数据构造 |

## 8. 前端

前端使用 Flask 实现，入口为：

```text
src/web/app.py
src/web/templates/index.html
```

启动方式：

```bash
python src/web/app.py
```

页面支持：

- 首句续写
- 藏头诗
- 成积似涵
- 三模型切换
- temperature 调节
- 押韵优化
- 多候选展示

背景图位于：

```text
static/beijing.png
```

## 9. 复现命令

安装依赖：

```bash
pip install -r requirements.txt
```

重新处理数据：

```bash
python -m src.data.preprocess --download
python -m src.data.prepare_chengyu
```

训练三个 GRU 模型：

```bash
python -m src.engine.train --config configs/gru_plain_baseline.yaml
python -m src.engine.train --config configs/gru_plain_weighted.yaml
python -m src.engine.train --config configs/gru_base.yaml
```

训练 Prefix Global BiGRU：

```bash
python -m src.engine.train_global_prefix_scorer --config configs/global_prefix_bigru_20e.yaml
```

评测 structured 模型：

```bash
python -m src.engine.evaluate --checkpoint checkpoints/gru_best.pt --out runs/moxing/jiegou/evaluation.csv
```

重画对比图：

```bash
python -m src.utils.comparison_plot --comparison runs/duibi/biaoge/san_moxing_duibi.json --out_dir runs/duibi/tupian
python -m src.utils.mode_model_bar_plot
python -m src.utils.model_strategy_bar_plot
```

启动前端：

```bash
python src/web/app.py
```

也可以使用 Makefile：

```bash
make train-baseline
make train-weighted
make train-structured
make train-prefix
make web
```

## 10. 目录结构

```text
configs/       最终训练配置
checkpoints/   最终模型权重
data/          原始数据、成语库、处理后的七绝数据
src/           数据处理、训练、生成、评测、前端代码
runs/          最终训练曲线、评测表和对比图
static/        前端背景图
```