# 基于统一多任务字符级 GRU/LSTM 的七言绝句条件生成

## 1. 任务概述

本项目面向七言绝句生成，要求输出四句、每句七个汉字。系统支持两种条件生成模式：给定首句续写后三句；给定四个藏头字生成四句藏头诗。

## 2. 数据处理

数据来自 DICAlab Ancient Poems Dataset。预处理脚本完成 Unicode 归一化、鲁棒分句、严格 4×7 过滤、去重、随机划分，并输出 `stats.json`。请将脚本实测统计粘贴到此处。

| 项目 | 数值 |
|---|---:|
| 严格七绝数量 | TODO |
| 训练集 | TODO |
| 验证集 | TODO |
| 测试集 | TODO |
| 字符表大小 | TODO |

## 3. 模型结构

主模型为字符级 GRU。每个字符先映射为字符嵌入，再与位置嵌入拼接，输入两层 GRU；输出经过 LayerNorm 与线性层后得到下一个字符概率。训练序列使用 `<TASK_CONT>`、`<TASK_ACRO>`、`<L1>...<L4>` 等结构 token，从而在单一 checkpoint 中统一支持多个条件任务。

## 4. 训练设置

| 超参数 | GRU | LSTM |
|---|---:|---:|
| embedding dim | 256 | 256 |
| position dim | 32 | 32 |
| hidden size | 512 | 512 |
| layers | 2 | 2 |
| dropout | 0.2 | 0.2 |
| optimizer | AdamW | AdamW |
| learning rate | 3e-4 | 3e-4 |
| epochs | 30 | 30 |

插入：`loss_curve.png` 与 `val_ppl_curve.png`。

## 5. 指标

PPL 按测试集自由建模任务的平均负对数似然计算。格式合规率要求生成结果恰好四句且每句恰好七个汉字。额外统计 Distinct-2、重复率、藏头正确率、与训练集最近邻的 4-gram Jaccard 相似度。

## 6. 结果

### 6.1 模型比较

| 模型 | Test PPL | 格式合规率 | Distinct-2 | 重复率 |
|---|---:|---:|---:|---:|
| GRU | TODO | TODO | TODO | TODO |
| LSTM | TODO | TODO | TODO | TODO |

### 6.2 采样策略比较

注意：同一个 checkpoint 的 Test PPL 不随采样参数变化。

| 采样设置 | Test PPL | 格式合规率 | Distinct-2 | 重复率 | 简评 |
|---|---:|---:|---:|---:|---|
| T=0.7 | TODO | TODO | TODO | TODO | 稳定但较保守 |
| T=0.9 + top-k=20 | TODO | TODO | TODO | TODO | 平衡性最好 |
| T=1.1 + top-p=0.95 | TODO | TODO | TODO | TODO | 新鲜但更易跳跃 |

## 7. 生成样例与分析

运行 `python -m src.engine.demo --checkpoint checkpoints/gru_best.pt`，将 `runs/moxing/jiegou/demo/samples.md` 中的五组首句续写和五组藏头结果粘贴到此处。逐组分析：结构是否完整、是否跑题、是否有重复、藏头是否自然、是否疑似复写训练集。

## 8. 局限与改进

受约束解码保证了结构完整，但格式正确不等于内容优秀。功能字藏头、较高温度采样和低频字符可能造成语义跳跃。后续可增加小型 decoder-only Transformer、押韵约束或候选重排序器。

## 9. 结论

本项目以统一多任务字符级 GRU 为主模型，通过结构 token 与受约束采样实现了可复现、格式稳定的七言绝句条件生成系统。
