# Poet-I-get

基于字符级 GRU 的七言绝句条件生成系统。项目支持首句续写、四字藏头诗生成，并额外实现了一个面向展示的成语格诗增强模块“成积似涵”。

## 1. 项目任务

本项目完成课程要求的七言绝句生成任务：

- 数据集使用 ancient-poems-dataset。
- 只筛选四句、每句七字的七言绝句。
- 支持首句续写：输入 7 字首句，生成后 3 句。
- 支持藏头诗：输入 4 个藏头字，生成完整 4 句。
- 模型主体使用字符级 GRU。
- 实现 `temperature` 采样策略。
- 评测指标包含困惑度 `PPL` 和格式合规率。
- 输出训练曲线、指标表、生成样例和实验报告。

数据来源：

```text
https://dicalab-scu.github.io/nlp/post/ancient-poems-dataset/
```

处理后数据规模：

| 数据项 | 数量 |
|---|---:|
| 严格七绝 | 136631 |
| 训练集 | 122967 |
| 验证集 | 6831 |
| 测试集 | 6833 |

## 2. 模型设置

项目训练了三组 GRU 模型，用于比较不同结构设计对生成效果的影响。

| 模型 | 配置文件 | checkpoint | 说明 |
|---|---|---|---|
| baseline | `configs/gru_plain_baseline.yaml` | `checkpoints/gru_plain_best.pt` | 普通字符级 GRU，不使用句位标记 |
| weighted | `configs/gru_plain_weighted.yaml` | `checkpoints/gru_plain_weighted_best.pt` | 在藏头句首位置提高 loss 权重 |
| structured | `configs/gru_base.yaml` | `checkpoints/gru_best.pt` | 加入 `<L1><L2><L3><L4>` 句位标记，并在推理时使用结构约束 |

主实验结果：

| 模型 | 生成方式 | Test PPL | 格式合规率 | 藏头正确率 |
|---|---|---:|---:|---:|
| baseline | raw | 83.690 | 1.000 | 0.000 |
| weighted | raw | 78.264 | 1.000 | 0.890 |
| structured | constrained | 51.185 | 1.000 | 1.000 |

采样策略：

| 策略 | 参数 |
|---|---|
| stable | `temperature=0.7` |
| balanced | `temperature=1.0` |
| creative | `temperature=1.3` |

## 3. 创新点

### 3.1 藏头位置加权训练

`weighted` 模型在训练时对藏头任务中的四个句首位置设置更高 loss 权重。这样模型会更重视“输入藏头字”和“句首生成位置”之间的对应关系，从而提高无结构标记情况下的藏头正确率。

### 3.2 句位标记与结构约束

`structured` 模型在训练序列中加入 `<L1><L2><L3><L4>`，使模型显式感知当前处于第几句。推理阶段再配合结构约束，控制每句 7 个汉字，并在藏头模式中固定四句句首字，从而保证七言绝句格式稳定。

### 3.3 押韵评分重排序

前端支持多候选生成，并通过押韵评分对候选结果排序。评分重点考虑第 2、4 句是否押韵，第 1 句是否入韵，第 3 句是否避韵，用于从多首候选诗中选择韵脚更自然的结果。

### 3.4 成积似涵：成语格诗增强模块

“成积似涵”是项目的扩展创新模块。它将四字藏头诗表示为一个 `4 x 7` 字阵：横向读是四句七言诗，纵向读则是首列藏头字和后六列四字成语。

真实生成流程如下：

1. 将四个藏头字固定为字阵第一列。
2. 从成语库中筛选模型词表内可识别的四字成语。
3. 默认从全量可用成语池中按类别抽样 240 个候选成语。
4. 使用 Beam Search 逐列扩展后六列成语。
5. 在 Beam Search 中，使用 Prefix Global BiGRU Scorer 对当前半成品字阵的新生成列进行前缀一致性评分。
6. 在 Beam Search 中同时加入重复惩罚、短语惩罚和风格惩罚，提前削弱横向表达生硬的路径。
7. 完整字阵生成后，使用训练好的 GRU 七绝语言模型计算横向四句诗的 NLL。
8. 最终结合 GRU NLL、重复惩罚、短语惩罚、风格惩罚和押韵分进行综合排序。

这里需要区分两个模型的作用：

- `GRU`：用于完整候选结果的横向诗句 NLL 评分，判断横着读是否像七言绝句。
- `Prefix Global BiGRU Scorer`：用于 Beam Search 中途，对尚未完成的半成品字阵进行前缀一致性评分。

相关代码：

| 文件 | 作用 |
|---|---|
| `src/engine/chengyu_grid.py` | 成语库读取、候选评分、基础成语格搜索 |
| `src/engine/chengyu_global_beam_experiment.py` | Prefix Global BiGRU 引导的 Beam Search |
| `src/engine/global_prefix_score.py` | 前缀一致性评分接口 |
| `src/models/global_prefix_bigru_scorer.py` | Prefix Global BiGRU Scorer 模型 |

相关 checkpoint：

```text
checkpoints/global_prefix_bigru_20e.pt
checkpoints/global_bigru_scorer_20e.pt
```

## 4. 前端

前端使用 Flask 实现，支持三种功能：

- 首句续写
- 藏头诗
- 成积似涵

启动方式：

```bash
python src/web/app.py
```

前端入口：

```text
src/web/app.py
src/web/templates/index.html
```

背景图：

```text
static/beijing.png
```

## 5. 训练与评测命令

训练三组主模型：

```bash
python -m src.engine.train --config configs/gru_plain_baseline.yaml
python -m src.engine.train --config configs/gru_plain_weighted.yaml
python -m src.engine.train --config configs/gru_base.yaml
```

评测 structured 模型：

```bash
python -m src.engine.evaluate --checkpoint checkpoints/gru_best.pt --out runs/moxing/jiegou/evaluation.csv
```

导出生成样例：

```bash
python -m src.engine.demo --checkpoint checkpoints/gru_best.pt --out_dir runs/moxing/jiegou/demo
```

绘制对比图：

```bash
python -m src.utils.comparison_plot --comparison runs/duibi/biaoge/san_moxing_duibi.json --out_dir runs/duibi/tupian
python -m src.utils.mode_model_bar_plot
```

## 6. 报告与结果文件

主要报告：

```text
report/gru_report.md
report/初版报告.md
report/七言绝句生成实验报告.docx
report/七言绝句条件生成系统实验报告_BP风格.docx
```

主要结果：

```text
runs/duibi/biaoge/san_moxing_duibi.csv
runs/duibi/tupian/san_moxing_ppl_duibi.png
runs/duibi/tupian/xuxie_moshi_jiu_zuhe.png
runs/duibi/tupian/cangtou_moshi_jiu_zuhe.png
runs/moxing/jiegou/demo/samples.md
```

## 7. 推荐展示方式

课程主任务建议展示：

- `structured` 模型
- `balanced` 采样策略
- 首句续写和藏头诗各 5 组样例
- PPL、格式合规率、藏头正确率对比表

创新模块建议展示：

- 成积似涵字阵
- 竖向成语列
- GRU NLL 评分
- Prefix Global BiGRU Scorer 的中途路径筛选作用
