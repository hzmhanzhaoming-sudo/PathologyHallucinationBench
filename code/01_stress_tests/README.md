# 01：原始六项压力测试（Original，无新增提示策略）

本目录负责**生成模型回答**，不负责评分。论文的六项任务是 **V1、V2、V3、T1、T2、T3**；Base 是共享干净对照，不是第七项压力测试。历史服务器目录 `Benchmark_T1` 至 `Benchmark_T7` 与论文任务名并不一一同名。

## 已保留的内容

- `vendor/` 是 2026-10-26 从服务器只读取得的 60 个原始评估脚本，逐字保留，哈希在 `provenance/original_source_manifest.json`。
- `run_local.py` 与 `run_api.py` 分离。只负责选输入、模型和新的输出位置；提示词、模型加载、原始输出解析和重试流程仍由所选原脚本处理。
- 支持论文 16 模型：12 个本地模型、4 个 API 模型。不会把 12 模型的提示策略子实验错当原始 16 模型实验。
- V1 采用 **完整 1200 题**，不是 `Benchmark_T2_data` 的 900 题；T1 仅采用论文报告的 `unknown1_2` 变体，不混入未知选项 3–5 级。

|论文任务|原服务器数据|本入口条件|初始请求数|
|---|---|---|---:|
|Base|Benchmark_T1_data|Base，有图|1200|
|V1|Benchmark_T1_data|V1_without，无图；与共享 Base 配对|1200|
|V2|Benchmark_T4_data|错误图像描述|900|
|V3|Benchmark_T6_data|V3_with / V3_without|1800|
|T1|Benchmark_T3_data|四、六、七选项的 unknown1_2 文件|1800|
|T2|Benchmark_T5_data|T2_present / T2_absent|1800|
|T3|Benchmark_T7_data|错误答案提示|900|
|合计|—|11 个条件文件；共享 Base 只运行一次|9600|

每项独立调用也会自动包含 Base 对照。不要直接运行 `vendor/`，这些原脚本内保留了历史默认结果路径；使用根目录包装入口才会把结果隔离到本包内。

## 默认只预览，不加载模型、不联网、不写结果

```bash
cd /data5/hzm/Benchmark/CODE/01_stress_tests
python3 run_local.py --model qwen35-4b --task all --dry-run --validate-inputs
python3 run_api.py --model gemini-3.6-flash --task all --dry-run --validate-inputs
python3 -B test_offline.py
```

预览显示每项确切输入、命令、原脚本参数默认值和代码哈希。`--validate-inputs` 会只读核对输入行数、唯一 ID 和审计快照 SHA256；不需要 GPU 或 API key。

确实要新运行时，必须显式提供 `--execute --run-id <新名字>`。例如以下命令会实际加载模型：

```bash
python3 run_local.py --model qwen35-4b --task all --run-id original_v1 --execute
```

API 同理，但会产生真实调用和费用。密钥仅从 `OPENAI_API_KEY` / `GEMINI_API_KEY` 环境变量读取；包装入口不接受密钥参数。可用 `--thinking-level` 指定 Gemini 原脚本已有的思考配置，不支持把该参数强加到 GPT。若通过环境变量配置 API 地址，必须为无嵌入凭据的 HTTPS 地址。

新输出为 `results/<run-id>/<model>/`，含 `run_manifest.json` 和各条件 JSONL。`--limit` 的小规模试跑放在 `pilot_results/`，不会混入完整结果。已存在同名模型输出目录时明确拒绝运行；此版不提供隐式覆盖或续写。底层原脚本的输出字段保持原样。

## 评分衔接与一致性边界

```bash
python3 ../02_scoring/score_stress_run.py \
  --run results/original_v1/qwen35-4b \
  --output ../02_scoring/reports/original_v1_qwen35
```

`02_scoring` 才是统一评分入口。它使用保存的答案标签，不重新解释自由文本。T1 必须按 `t3_original_num` 元数据配对，不能仅去掉 `-unknown1` 后缀：六/七选项数据曾有编号前缀转换。

这里保存的是**服务器当前脚本快照**，不声称已经恢复每次历史调用的所有运行时参数。原脚本默认参数不自动等于论文每次历史请求的实际参数；重跑前需依据原运行记录核对模型版本、解码和原生思考配置。GPU/模型服务变化也可能导致新预测不同。论文数值的“完全对应”应通过 `02_scoring` 对同一份已保存原始结果的逐题回归确认，而不能靠强制新推理输出与旧数值相同。

本次整理未运行模型、未调用 API、未修改原始实验数据或结果。缺失输入、输入哈希变化和代码修改都会显式报错，不会生成假的通过结果。
