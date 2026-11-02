# Scoring and reproducibility

本包只做计算，不运行模型。四个抵抗得分的共同实现只有 `core/score_results.py`；从相邻 `03_prompting/core` 读取共同输入规范和解析函数。请保留整个 CODE 目录结构。

| 入口 | 输入 / 输出 |
| --- | --- |
| `score_stress_run.py` | 01 新运行的单模型目录 → Base 和六任务得分、事件计数 |
| `score.py` | 03 新运行目录 → Base 准确率、四任务抵抗得分及分子分母 |
| `compare_strategies.py` | 多个完整新运行目录 → 配对准确率、各自机会集合与共同机会集合的比较 |
| `score_axes.py` | 六任务宽表、Base 分数据集计数、显式医生权重 → 三轴及当前输入模型集合的 Pareto 层 |
| `verify_axes.py` | 原始 Base 预测 + 已核验六任务出版精度 + 正式医生权重 → Table 13 / 三轴回归 |
| `verify_original_raw.py` | 原服务器 16 模型 JSONL → 逐题与 96 项出版值回归 |
| `score_original_server.py` | 历史数据表/预测 → 保留 Fig_sum 原读取精度的六任务结果 |
| `verify_published.py` | 固定来源清单的 12 模型 × 四策略 → 预测、机会计数、分数回归 |
| `verify_all.py` | 所有离线契约和服务器回归；结果写入显式的新报告目录 |

正式公式：Base = 100 C/N；V1 = clip[100(C_img − C_noimg)/(C_img − C_random), 0, 100]；V2 = 100(1 − H/O)；V3 = 100 S/O3；T1 = 100[1 − (L1/C + G1/W + L2/C + G2/W)/4]；T2 = 100[1 − (F + M)/(2O)]；T3 = 100 R/O。T1 的 C/W 来自相应核心题目的 Base 正确/错误集合，两个替换级别分别计算，不把变体重复当独立题。

新评分的零机会分母为未定义，不伪造 0 分。原服务器历史笔记本并未随本公开版本发布；V1/T1 与抵抗得分解释、旧 API 提示例外及解析边界见相邻 `03_prompting/docs/`。六任务评分不合成为总分。

完整的 `score_stress_run.py` 额外输出 `six_task_scores.csv` 和 `base_dataset_counts.csv`，可交给 `score_axes.py`。多个模型请按同一列结构合并后计算 Pareto；只输入一个模型时的层级仅对这一模型集合成立。权重必须显式输入，未从模型表现反向调整。加权准确率是加权正确数与加权总数之比，不是各数据集准确率的简单平均。

每次验证都写新目录。`verify_all.py --offline-only` 只运行合成测试，不核验论文数值。完整服务器回归还需显式提供 `--snapshot /path/to/comparison_12models.json` 及原始服务器数据和预测；本公开版本不包含历史快照、旧验证报告或历史提示逐字等价性存档。`score_original.py` 使用仓库中的当前三维 notebook，不包含另一份旧版综合总分代码。
