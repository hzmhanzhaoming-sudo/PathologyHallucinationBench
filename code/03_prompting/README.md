# Three prompting strategies

策略：`zero_shot_cot`、`few_shot_direct`、`few_shot_cot`。

- 本地模型入口：`run_local.py`；API：`run_api.py`。
- 新运行均用 `standard`，历史 API 例外只通过 `run_historical_api.py`，另存结果。
- `--dry-run` 不加载模型、不调用 API、不写预测。去掉后会执行真实任务。
- `--task all` = Base、V2、V3_with、V3_without、T2_present、T2_absent、T3，共 6,600 次初始请求，不包含 V1/T1。
- 核心评分单一来源位于相邻 `02_scoring/core/score_results.py`，不要复制回本目录。
- Base/V2/T3 的 few-shot 使用三条源示例；V3/T2 使用一组成对示例。两种 few-shot 保持相同示例题和正确答案，解释文本按策略变化。
- 历史协议差异保存在 `docs/`；本公开版本不包含历史输出、重复代码存档或 few-shot 原图，不能据此声称完整重现旧实验。示例原图须按已有授权另行提供，不可替换为测试集图片。

示例命令和评分方法见上一级 README。内部 `execute.py`/`core/` 不是供用户绕过公开入口直接执行的命令。搬迁使元数据指纹变化，必须另开新结果目录。
