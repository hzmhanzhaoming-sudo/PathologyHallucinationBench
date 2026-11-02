# 历史提示词与 API 协议说明

本包整理已有运行和评分代码，不重新运行实验，也不修改已完成结果。历史上存在一处模型专用提示词修订，不能把所有已完成实验描述为逐字使用了同一提示词。

2026-10-25 用户确认：今后统一一版，旧版单独保存。因此 `run_local.py` / `run_api.py` 默认且只允许标准 `standard`；下文历史差异仅由 `run_historical_api.py` 复现，并另存结果。

## 逐份核对的范围

只读核对服务器 `/data5/hzm/Benchmark` 下 3 种策略 × 3 个 API 模型 × 7 个输入条件的 **63 份 `.run.json`**。条件为 Base、V2、V3 with/without、T2 present/absent、T3；Base 为 1,200 题，其余每条件 900 题。记录中的 `code_sha256.common.py` 与当前存档代码逐字节匹配；同一模型—策略组的 7 个条件具有一致的提示词代码版本和生成参数。

结构化证据见 `provenance/api_historical_protocols_audit.json`。该证据仅含版本、参数及后处理计数，不含模型原始回答、密钥或其他凭据。`audit_prompt_remote.py` 为生成此核查的只读脚本。

## 九组 API 的历史 profile

| 策略 | Gemini 3.5 Flash Lite | Gemini 3.6 Flash | GPT-5.6 Luna |
|---|---|---|---|
| Zero-shot CoT | `standard` | `standard` | `api_visible_explanation` |
| Few-shot direct | `standard` | `standard` | `standard` |
| Few-shot CoT | `standard` | `standard` | `standard` |

新核心的 `standard` 使用与以上 **8 组历史实验**相同的指令内容；`api_visible_explanation` 对应历史 Luna zero-shot CoT 的 `visible_explanation_v2`。这是历史复现映射，不应把该 Luna 专用配置静默推广至其他模型，也不应把已完成 Luna 结果改标为 `standard`。

标准 CoT 的主指令为：

> Answer the pathology multiple-choice question. Think step by step, then return the final option letter within &lt;answer&gt; and &lt;/answer&gt; tags.

标准 direct 的主指令为：

> Answer the pathology multiple-choice question. Return only the selected option letter. Do not explain.

历史 Luna zero-shot CoT 的主指令为：

> Answer the pathology multiple-choice question. First, write a brief explanation in 2-4 sentences supporting your selected option, using only the information provided. The explanation must be visible in your response. Then, on a new line, return the final option letter within &lt;answer&gt; and &lt;/answer&gt; tags. Do not return the answer letter alone. Keep the entire response concise so that both the explanation and final answer fit within 512 output tokens.

该历史修订在 `luna_explanation_activation.json` 中记为响应“需要可见解释”的要求，而非根据答案正确率选择提示词。最终 7 个条件均确证使用修订后的代码。原始标准 `common.py` SHA-256 为 `fcfdd550d8e0773c802738d6913a4910203c883edd479f61cf13833c8a569b2b`；Luna 修订版为 `be96dc35414df37ef53c3ce514e9256700a9851714048fd2920ba0acc3695bde`。

## 生成预算与原生推理参数

- 全部 63 份元数据记录主请求输出预算 512 tokens，`temperature=None`（未显式发送温度），允许 CoT 在输出预算耗尽且严格解析仍无合法答案时追加一次最多 32 tokens 的 final-answer 请求。不是 512 内再切出 32，也不是答错重试。
- Few-shot direct 不触发 CoT finalization，即使统一参数中保留 `force_answer_on_limit=1`。
- Gemini 3.5 Flash Lite：`thinking_level=default`，不显式传 `thinkingBudget`。
- Gemini 3.6 Flash：`thinking_level=minimal`，三个策略一致；激活记录注明原因是输出不完整，原始任务提示词未改。
- GPT-5.6 Luna：`reasoning_effort=none`，三个策略一致；CoT 解释仍来自可见输出指令，不能把原生推理设置与外显 CoT 文本混为一谈。
- API 请求并发管理与网络恢复不属于评分公式。历史运行器每模型跨策略共享并发上限 3，限流后降至 1；额度、鉴权或未知错误应停止，不得通过恢复逻辑绕过。

## Few-shot 的示例数据不能省略

两种 few-shot 共享同一份 `task_demos.py`（SHA-256 `0286f5e67ca033ecc0480545a46ad60762c6ada786904b711caefbab4f8d74b1`）与 `task_specific_fixed_pairs_v3` 协议。42 份 few-shot API 元数据均对应套件 SHA-256 `fa03da74c6e48d30d2277a8cb2e7833fb2e014aab210846113e62f9f3e6e93fd`。

- Base、V2、T3 各用 3 个固定来源示例；V2/T3 的示例带相应干扰，不能替换为普通 Base 示范。
- V3、T2 使用 **1 对完整示例，共 2 个示范回合**，而不是每条件 3 个示例；同一任务两种输入条件使用完全相同的成对前缀。
- V3 对包含有图和无图示例；T2 对包含正确答案在选项内和不在选项内的示例，T2 两种示例均有图。
- direct 与 CoT 的示例输入一致；direct 示例回答仅字母，CoT 为示例解释与答案标签。
- 必需数据依赖为固定来源示例 JSON、原始解释记录、`task_specific_v3/demo_suite.json` 和对应示例图像。加载时核对套件与源转换、文件哈希及示例/目标图像重叠，不能只复制 Python 而省略示例资料。

## 历史解析与离线后处理顺序

1. 推理时使用严格的 `common.parse_answer`。它决定 CoT 是否需要一次 final-answer 请求；无关分析文字中的 A/B 不作为答案。
2. `normalize_answer_tags.py`：仅对原预测为空的已完成文件解析显式 `(A)` 或 `[A]` 答案标签，政策 `explicit_parenthesized_answer_tag_v1`。最终元数据记录共 10 条修复。
3. `parse_completed_20261015.py`：仅对仍为空的预测采用审定的明确终止字母/单标签规则，政策 `completed_api_explicit_final_letter_20261015_v1`。最终元数据记录共 38 条修复。既有已解析预测和原始输出不变；选择答案时不使用正确标签。
4. `first_listed_fallback_20261015.py`：历史用户明确授权的 **3 条特例**，政策 `user_authorized_first_listed_choice_20261015_v1`。均为 GPT-5.6 Luna 的题目 `400-334`，仅限 zero-shot CoT Base、zero-shot CoT V2、few-shot CoT Base。该脚本从多答案输出按出现顺序取第一个，不依据 gold；这不是普通单答案解析成功，也不是重跑模型。

**第 4 步不得泛化到新实验或其他多答案输出。** 原始多答案回答及 `original_single_answer_valid=False` 等标记必须保留，另做把这 3 条按无效处理的敏感性核查。对当前已存结果重评分时，保留已存 `pred_letter` 和后处理来源，不应不加区分地重新解析整批回答。

更宽松的离线解析不可直接替换 finalization 所依赖的严格解析，否则可能减少/改变原有第二阶段请求，导致新生成协议不再与历史一致。离线脚本应另存，不静默改变核心生成逻辑。

## 代码组织与保留边界

三策略的 `common.py`、`local_backends.py`、`registry.py`、`finalization.py`、`api_execution.py` 和 `api_concurrency.py` 在根 code 目录中逐文件哈希一致，可共用。few-shot 两策略的 `task_demos.py` 也逐字节一致。Luna 分支除上述 zero-shot prompt 外，`api_backends.py` 含正确的 null-error 判定：只有 `data.get('error') is not None` 才视为 API 错误；应保留此修正。

本地模型可保留不同权重加载器和原生 chat-template 适配器，但共用策略指令、示例、严格解析和 finalization。模型 chat-template 与系统提示属于历史适配器的一部分，例如 Patho-R1 的病理专家 system turn；不应为追求字面统一而在整理时删除。

`.gemini_api_batch_20261013` 是历史 API 调度/恢复及离线解析目录，并非单一代理代码。核心来源包括 `run_gemini_six.py`、`run_luna_three.py`、`reliable_api_queue.py` 及上述 3 个解析/特例脚本。`server_proxy.py` 是历史网络辅助，不是科研提示词或评分逻辑；不应复制私有代理配置、密钥、运行态锁或日志到共享代码包。旧代码指向旧结果目录，归档文件不应作为新实验直接执行入口。
