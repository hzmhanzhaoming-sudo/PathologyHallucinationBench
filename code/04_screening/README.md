# 04_screening：按论文提示词编写的独立筛选入口

本包与 `01_stress_tests`、`02_scoring`、`03_prompting` 分开；不修改它们，也不重新计算旧实验。**固定提示词直接提取自当前补充材料对应的原生 PPT 文本，不根据历史筛选过程反推。**

## 与原文对应

| 入口 stage | 原文提示词 | 论文模型名称 | 来源 |
|---|---|---|---|
| `classify` | Stage 1 Content Classification Prompt | GPT-5.6 Terra | Note 6 第 1 图；PPT 第 16 页 |
| `eligibility` | Stage 1 Eligibility Screening Prompt | GPT-5.6 Terra | Note 6 第 2 图；PPT 第 17 页 |
| `compare` | Stage 2 Semantic Comparison Prompt | GPT-5.6 SOL | Note 6 第 3 图；PPT 第 18 页 |

四份原样文本在 `prompts/`：三个 fixed instruction，加资格筛选的 user prompt template。保持全部字词、标点、空格和原生段落，仅把换行序列统一为 LF。不擅自修正文稿中的空格/标点黏连，也不增加旧执行脚本的隐藏指令。调用时不得 `.strip()`、重写或追加固定提示词。

来源是 `npjdm_for_all/supply_绘图.pptx` 第 16–18 页的原生文字，对应 `Figures/Note_06_01...03` 三张当前补充材料图片。完整来源面板、SHA-256 和逐项说明保存在 `provenance/`。

## 不是历史流程复现

- 分类：仅对提供的文本进行三轴内容归类，不生成答案、不决定资格。
- 资格筛选：按原文五项标准审查文本、选项、答案和来源信息；不提供图像，不推测图像内容。
- 语义比较：输入已有的图像审查结果、原始文本、标签、邻居和队列摘要；**不重新看图**，不直接决定最终入组。
- 不实现旧流程的抽样、配额、排序、邻居检索、去重或 4,500→1,200 的最终选择。不声称单独执行这三段提示即可复现历史队列。
- 原文没有完整列出独立图像审查提示词，本包没有用 Methods 的过程描述补写一个“原文提示词”。

## 原文与实现层的明确边界

1. 原文只写了 `Fixed instruction`，没有指定 API 消息角色。本包默认放在 `system`；也可显式设 `developer`。这是传输设置，不冒充论文原文。
2. 资格筛选用户模板逐字使用 `records={records_json}` 的原文句子，只替换占位符。分类、语义比较没有公开用户模板，本包用户消息仅为调用者输入的 JSON，不加入任何额外自然语言指令。
3. 三张原图均未给出完整 JSON Schema。因此 `--schema` **必填且由调用者显式提供**，没有伪称原文的默认 schema。传输要求 object 根节点及一个结果数组，默认键 `results`，可由 `--result-key` 指定。结果 schema 也属于实现层设置，不是论文原句。运行前应核对外部 schema 没有添加新筛选标准。
4. 输入封装、ID 检查、输出目录、HTTP 调用及结构校验都是本包工程实现；医学判断与筛选规则来自原文提示词。校验通过只代表格式和 ID 对齐，不代表医学判断正确，也不代表最终队列已确定。
5. `--model` 必填：使用对应论文模型在实际提供方的模型 ID，不猜测名称映射、不自动替换模型；请求/返回的模型标识留档。输出上限也是显式运行参数，不声称原文规定了某个 token 数。

### 保留的原文缺口

资格筛选原图有 `not_assessable.Return`、`failure Return` 等标点缺失；原图没有明确写出 `needs_review` 分支，尽管 SI 正文提到该分支。此处**以用户要求的原文提示词为准，保留不改**；不回填历史脚本，也不在后处理阶段自动补分支。详细记录见 `provenance/SOURCE_AUDIT.md`。

## 文件与使用

- `prepare.py`：读取指定批次和外部 schema，生成独立请求目录；完全离线。
- `run_api.py`：默认预览；只有 `--execute` 才发送一次 Responses 兼容请求。无自动重试、无自动修复回答、无模型工具调用。
- `validate_results.py`：离线校验另行取得的结果。保留原响应，不重写分类或筛选决定。
- `verify_prompt_source.py`：核验原文文本；可额外核对原 PPT 和三张图片。
- `test_offline.py`、`test_prompt_source.py`：离线测试；合成记录仅在测试中，不是实验结果。

环境：Python 3.8+。直接第三方依赖只有 `jsonschema`（pip 会解析其依赖）；如果尚未安装，先在独立虚拟环境中运行 `python -m pip install -r requirements.txt`。本次仅在本地隔离测试目录安装校验依赖，没有改动系统/服务器 Python 环境，没有运行筛选或发起 API 调用。

### 输入封装（实现层，不是原文 schema）

`classify` / `eligibility`：UTF-8 JSON 数组，每条是记录对象。默认唯一字符串 ID 为 `unit_id`；资格筛选可用 `--id-key` 指定现有 ID 字段。分类的 `unit_id` 必须与原文一致。其余原始字段由调用者原样提供，不自动修改答案、选项、来源信息。

`compare`：UTF-8 JSON 对象：`targets` 为目标记录数组，`cohort_summaries` 为已有队列统计。每个 target 包含 `review_id`、`original_texts`、`image_review_findings`、`content_labels`、`neighbours`；邻居为带 `review_id` 的记录对象列表，可为空。调用者必须提供实际内容，本包不生成视觉发现、不检索邻居、不读取图片路径。输出的 `related_ids` 只能来自该目标提供的邻居。

按需准备批次；不自动循环全部数据。以下文件路径和模型变量由使用者设置，示例不是实际实验：

```bash
python -B verify_prompt_source.py
python -B prepare.py --stage eligibility --input /path/to/records.json --schema /path/to/response_schema.json --model "$TERRA_API_MODEL" --max-output-tokens 4096 --output /path/to/NEW_prepared_batch
python -B run_api.py --prepared /path/to/NEW_prepared_batch
```

最后一行**仅预览**，不需要密钥，不产生 API 费用。示例 4096 仅为调用者可调整的工程参数，不是原文设定。阶段 2 改为 `--stage compare` 并显式使用 SOL 对应 ID；分类改为 `--stage classify`。

真正运行时需另行在环境变量中提供密钥，并显式授权执行：

```bash
python -B run_api.py --prepared /path/to/NEW_prepared_batch --output /path/to/NEW_result_batch --execute
```

默认密钥变量 `OPENAI_API_KEY`，可用 `--api-key-env` 指定名称。默认地址 `https://api.openai.com/v1`；提供方不同则显式设置 HTTPS `--base-url`。不硬编码、不记录密钥，不允许 URL 携带凭据，不跟随重定向。可用性、模型权限及 schema 支持需在实际运行环境确认；本次未进行在线验证。

所有输出必须新建，已有目录/文件拒绝覆盖。成功时保留请求、外部 schema、原始响应、解析结果和校验报告；失败保留原始响应（若收到）及 `FAILED.json`，不把失败当作筛选结论。超时可能已被提供方处理，故不会自动重试；应先人工核实。

```bash
python -B validate_results.py --prepared /path/to/NEW_prepared_batch --results /path/to/results.json
python -B -m unittest discover -p 'test_*.py' -v
```

API 传输结构参考 [OpenAI 官方 Structured Outputs 文档](https://developers.openai.com/api/docs/guides/structured-outputs)；只用于实现消息与 JSON Schema 接口，不用于修改论文提示词。包级 `package_manifest.json` 单独记录本新增包文件，原三个包的根清单和文件保持不变。
