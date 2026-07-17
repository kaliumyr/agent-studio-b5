# B5 记忆文档存储与查找模块 - 个人 README

## 1. 模块概述

### 1.1 模块名称

`B5 - 记忆文档存储与查找模块 (Memory Module)`

### 1.2 模块说明

B5 模块是 Agent 系统的本地记忆管理组件，负责 Agent 的记忆存储、检索和管理。它使 Agent 能够：

1. **记忆检索**：根据用户指定的 memory_id 或关键词查询，从本地记忆库中查找相关文档
2. **记忆存储**：将对话过程中的 messages、trace 和 final_answer 保存为结构化的记忆文档
3. **记忆管理**：维护记忆索引，支持记忆的更新、冲突检测和版本管理
4. **记忆分析**：对比不同记忆的内容，分析错误记忆对 Agent 回答的影响

该模块是 Agent 系统中实现长期记忆和上下文感知能力的核心组件，使 Agent 能够在多次对话中保持知识连续性。

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 |  全部完成 (5/5) |
| 进阶要求 |  全部完成 (5/5) |
| 可独立运行的演示 |  `python3 code/b5_memory.py --config configs/memory.yaml --query "xxx" --search_mode keyword --outdir outputs/B5_advanced` |
| 与团队系统集成情况 |  被 B1 模块调用，提供记忆注入和保存接口 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.10+ |
| 必要依赖 | PyYAML, numpy, sentence-transformers (向量检索可选) |
| 是否需要模型 |  是（向量检索需要 sentence-transformers 模型） |
| 是否需要 GPU | 否 |
| 是否需要外部数据集 | 否（使用项目自带的 memory 目录） |

### 2.2 模型依赖

向量检索功能依赖 sentence-transformers 模型：

| 模型 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace | 自动下载到缓存目录 | 多语言文本向量化（用于语义检索） |

```bash
# 安装向量检索依赖（可选）
pip install numpy sentence-transformers

# 设置镜像源加速下载（国内环境）
export HF_ENDPOINT=https://hf-mirror.com
```

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| memory_index.json | 程序自动生成 | `memory/memory_index.json` | 记忆索引，记录所有记忆的元信息 |
| 全局记忆文档 | 程序生成 | `memory/global/*.md` | 跨任务共享的长期记忆 |
| 对话记忆文档 | 程序生成 | `memory/conversations/*.md` | 单次对话的记忆记录 |
| 样例输入数据 | 项目自带 | `data/memory_inputs/memory_save_input.json` | 记忆保存功能演示 |

### 2.4 安装步骤

```bash
# 创建并激活环境
conda create -n agent python=3.10 -y
conda activate agent

# 安装依赖
cd /root/siton-tmp/assignment_B/agent
pip install -r requirements.txt

# 向量检索可选依赖
pip install numpy sentence-transformers
```

---

## 3. 文件结构与接口边界

### 3.1 文件结构

```text
agent/
├── code/
│   ├── b5_memory.py              # 主模块代码
│   └── common/
│       ├── io_utils.py           # IO工具函数
│       ├── logging_utils.py      # 日志工具
│       ├── path_utils.py         # 路径工具
│       └── text_utils.py         # 文本处理工具（关键词检索、摘要、向量）
├── configs/
│   └── memory.yaml               # 记忆系统配置文件
├── memory/
│   ├── global/                   # 全局记忆存储目录
│   ├── conversations/            # 对话记忆存储目录
│   └── memory_index.json         # 记忆索引文件
├── data/
│   └── memory_inputs/
│       └── memory_save_input.json # 记忆保存样例输入
└── outputs/
    └── B5_advanced/              # 输出目录
        ├── selected_memory.json  # 记忆检索结果
        ├── saved_memory.json     # 记忆保存结果
        ├── updated_memory.json   # 记忆更新结果
        ├── memory_impact_analysis.json # 记忆影响分析结果
        └── memory_log.jsonl      # 操作日志
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| 输入 | B1 模块 / 命令行 | JSON / YAML / 文本 | 接收配置、memory_id、查询词、待保存数据 |
| 输出 | B1 模块 / 文件系统 | JSON / Markdown | 返回记忆文档、保存结果、分析报告 |
| 配置文件 | memory.yaml | YAML | 定义记忆存储路径、索引位置等 |

**核心接口函数**：

| 函数名 | 调用方 | 功能 |
|---|---|---|
| `load_memory()` | B1 / 命令行 | 按 ID 查找记忆 |
| `load_memory_advanced()` | B1 / 命令行 | 增强版查找（支持关键词检索） |
| `load_memory_with_vector()` | B1 / 命令行 | 向量语义检索 |
| `save_memory()` | B1 / 命令行 | 保存对话为记忆文档 |
| `update_memory()` | 命令行 | 更新已有记忆 |
| `analyze_bad_memory_impact()` | 命令行 | 错误记忆影响分析 |

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

B5 模块需要实现以下基础功能：

| # | 功能 | 说明 |
|---|---|---|
| 1 | 读取 memory 配置 | 从 memory.yaml 加载存储路径、索引位置等 |
| 2 | 读取全局记忆文档 | 根据配置加载全局记忆 |
| 3 | 按 ID 读取文档 | 根据用户指定的 memory_id 查找对应文档 |
| 4 | 限制返回长度 | 按 max_memory_chars 截断超长记忆 |
| 5 | 生成记录文件 | 输出 selected_memory.json |
| 6 | 更新 memory_index.json | 记录 memory_id、类型、标题、路径、时间等 |
| 7 | 保存对话记忆 | 将 messages、trace、final_answer 保存为记忆文档 |
| 8 | 两种记忆类型 | 支持 global（全局）和 conversation（对话）两种类型 |

### 4.2 基础功能实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `b5_memory.py::_memory_paths()` | 解析 memory.yaml 配置 |
| `b5_memory.py::_read_index()` | 读取 memory_index.json |
| `b5_memory.py::load_memory()` | 按 ID 查找并返回记忆内容 |
| `b5_memory.py::save_memory()` | 保存对话为记忆文档 |
| `common/text_utils.py::extract_words()` | 文本分词（用于标题生成） |
| `common/text_utils.py::generate_summary()` | 生成记忆摘要 |

**处理流程**：

```text
查找流程:
[memory.yaml] -> [_memory_paths] -> [读取索引] -> [按ID查找] -> [截断超长内容] -> [返回结果]

保存流程:
[messages/trace/answer] -> [生成标题和摘要] -> [写入Markdown] -> [更新索引] -> [返回结果]
```

### 4.3 基础功能输入格式与样例

| 字段 / 输入文件 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `--config` | 文件路径 | ✅ | memory.yaml 配置文件路径 |
| `--select_memory_ids` | 字符串列表 | ❌ | 要查找的 memory_id 列表 |
| `--use_global_memory` | 布尔值 | ❌ | 是否加载全局记忆 |
| `--save_type` | 枚举 | ❌ | conversation / global |
| `--save_input_path` | 文件路径 | ❌ | 记忆保存输入文件 |
| `--outdir` | 目录路径 | ✅ | 输出目录 |

**样例输入 - 查找**：

```bash
--select_memory_ids mem_course_001 mem_conversation_conv_000 --use_global_memory true
```

**样例输入 - 保存** (`data/memory_inputs/memory_save_input.json`)：

```json
{
  "conversation_id": "conv_sample_001",
  "save_type": "conversation",
  "messages_path": "sample_messages.json",
  "trace_path": "sample_trace.json",
  "answer_path": "sample_final_answer.md"
}
```

### 4.4 基础功能演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

# 1. 按ID查找记忆
python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --select_memory_ids mem_course_001 \
  --use_global_memory true \
  --outdir outputs/B5_basic

# 2. 保存对话记忆
python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --save_type conversation \
  --save_input_path data/memory_inputs/memory_save_input.json \
  --outdir outputs/B5_basic
```

**运行后观察**：

- `outputs/B5_basic/selected_memory.json` - 包含查找结果
- `outputs/B5_basic/saved_memory.json` - 包含保存结果
- `memory/conversations/conv_sample_001.md` - 生成的新记忆文档
- `memory/memory_index.json` - 索引已更新

### 4.5 基础功能输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `selected_memory.json` | JSON | 包含 `status`, `selected_memory_docs`, `max_memory_chars`, `truncated` 等字段 |
| `saved_memory.json` | JSON | 包含 `memory_id`, `memory_type`, `path`, `summary` 等字段 |
| 记忆文档 | Markdown | 包含标题、元信息、摘要、最终回答、消息记录、执行轨迹 |
| `memory_log.jsonl` | JSONL | 每次操作的日志记录 |

### 4.6 基础功能结果

**B5 记忆查找结果 `selected_memory.json`**：

![Uploading 20260717130938_302_172.png…]()

```json
{
  "status": "success",
  "query": null,
  "selected_memory_docs": [
    {
      "memory_id": "mem_course_001",
      "memory_type": "global",
      "title": "Agent 基础概念",
      "path": "global/mem_course_001.md",
      "content": "# Agent 基础概念\n\nAgent 系统通常由模型、工具、记忆和执行循环组成。",
      "original_chars": 75,
      "included_chars": 75,
      "truncated": false
    }
  ],
  "max_memory_chars": 2000,
  "total_chars": 75,
  "truncated": false,
  "errors": []
}
```

**保存的记忆文档示例**：

```markdown
# Agent 基础概念

- memory_id: `mem_course_001`
- conversation_id: `course_001`
- created_or_updated_at: `2026-07-06T11:31:44+08:00`

## Summary

Agent 系统通常由模型、工具、记忆和执行循环组成...

## Final Answer

Agent（智能体）是一个能够自主感知环境、规划行动并执行任务的智能系统...

## Messages
...
```

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求 | 是否完成 | 对应文件 / 函数 | 简要说明 |
|---|---|---|---|
| 1. 关键词检索排序 | ✅ | `_search_by_keywords()`, `compute_keyword_score()` | 根据关键词匹配度排序返回 top-k |
| 2. 长度管理/摘要压缩 | ✅ | `generate_summary()`, `save_memory()` | 自动生成摘要，智能截取标题 |
| 3. 记忆更新与冲突管理 | ✅ | `update_memory()`, `compare_text()` | 支持 merge/replace/skip/ask 四种策略 |
| 4. 向量检索 | ✅ | `load_memory_with_vector()`, `_search_by_vector()` | 使用句子向量进行语义检索 |
| 5. 错误记忆影响分析 | ✅ | `analyze_bad_memory_impact()` | 对比错误记忆与正确记忆的影响 |

### 5.2 进阶功能 1：关键词检索排序

#### 功能说明

基础功能只支持按 memory_id 精确查找。关键词检索允许用户输入自然语言查询，系统自动计算每个记忆文档与查询的相关性分数，按分数从高到低排序返回最相关的前 k 个文档。

**解决的问题**：用户无需记住 exact memory_id，只需描述想查找的内容即可找到相关记忆。

**对系统的价值**：使 Agent 能够根据用户意图动态检索相关知识，提升对话的智能化水平。

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `common/text_utils.py::extract_words()` | 中英文分词 |
| `common/text_utils.py::compute_keyword_score()` | 计算文本与查询的匹配分数 |
| `common/text_utils.py::extract_snippet()` | 提取匹配片段 |
| `b5_memory.py::_search_by_keywords()` | 遍历索引，计算分数，排序返回 |

**处理流程**：

```text
[query] -> [分词] -> [遍历所有记忆文档]
                              |
                              v
                   [计算标题匹配分数 * 1.5]
                   [计算内容匹配分数]
                   [取最大值作为最终分数]
                              |
                              v
                   [按分数降序排序]
                   [返回 top_k 结果]
```

#### 输入格式与样例

| 字段 / 输入文件 / 配置项 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `--query` | 字符串 | ✅ | 查询关键词 |
| `--search_mode keyword` | 枚举 | ✅ | 指定使用关键词检索模式 |
| `--top_k` | 整数 | ❌ | 返回结果数量，默认 5 |

**样例输入**：

```bash
--query "Agent 工具调用" --search_mode keyword --top_k 3
```

#### 演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --query "Agent 工具调用" \
  --search_mode keyword \
  --top_k 3 \
  --outdir outputs/B5_advanced
```

#### 输出格式

```json
{
  "status": "success",
  "query": "Agent 工具调用",
  "search_mode": "keyword",
  "top_k": 3,
  "results": [
    {
      "memory_id": "mem_course_001",
      "memory_type": "global",
      "title": "Agent 基础概念",
      "path": "global/mem_course_001.md",
      "score": 0.9612,
      "snippet": "Agent 系统通常由模型、工具、记忆和执行循环组成...",
      "content_preview": "...",
      "matched_chars": 75
    }
  ],
  "total_matched": 3,
  "errors": []
}
```



### 5.3 进阶功能 2：长度管理与摘要压缩

#### 功能说明

在保存记忆时，自动对长文本进行压缩处理：
- 从用户消息中提取关键词生成有意义的标题
- 自动生成 200 字以内的内容摘要
- 在索引中记录摘要信息，便于快速预览

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `common/text_utils.py::generate_summary()` | 基于关键词提取生成文本摘要 |
| `b5_memory.py::save_memory()` | 集成摘要生成和智能标题提取 |

#### 演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --save_type conversation \
  --save_input_path data/memory_inputs/memory_save_input.json \
  --outdir outputs/B5_advanced
```

#### 输出示例

保存的记忆文档自动包含摘要：

```markdown
# Agent 基础概念

- memory_id: `mem_conversation_conv_001`
- created_or_updated_at: `2026-07-06T11:31:44+08:00`

## Summary

Agent（智能体）是一个能够自主感知环境、规划行动并执行任务的智能系统...

## Final Answer
...
```

### 5.4 进阶功能 3：记忆更新与冲突管理

#### 功能说明

支持对已存在的记忆进行更新，并提供四种冲突处理策略：

| 策略 | 说明 |
|---|---|
| `merge` | 合并新旧内容，保留两者信息 |
| `replace` | 用新内容完全替换旧内容 |
| `skip` | 如果检测到冲突则跳过更新 |
| `ask` | 返回冲突信息，人工决策 |

自动检测新旧内容的变更类型：
- **supplement**：大部分内容一致，补充了新信息
- **conflict**：有相当部分内容不同
- **replace**：大部分内容不同，可能是完全不同的主题

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `common/text_utils.py::compare_text()` | 对比新旧文本，识别变更类型 |
| `b5_memory.py::update_memory()` | 执行记忆更新 |
| `b5_memory.py::_merge_memory_content()` | 合并新旧内容 |

#### 演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

# 合并模式
python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --update_memory_id mem_conversation_conv_000 \
  --update_messages_path data/update_inputs/update_messages.json \
  --update_trace_path data/update_inputs/update_trace.json \
  --update_answer_path data/update_inputs/update_answer.md \
  --conflict_strategy merge \
  --outdir outputs/B5_advanced

# 替换模式
python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --update_memory_id mem_conversation_conv_000 \
  --update_messages_path data/update_inputs/update_messages.json \
  --update_trace_path data/update_inputs/update_trace.json \
  --update_answer_path data/update_inputs/update_answer.md \
  --conflict_strategy replace \
  --outdir outputs/B5_advanced
```

#### 输出示例

更新后的记忆文档包含变更记录：

```markdown
# Conversation conv_000 (Updated)

- memory_id: `mem_conversation_conv_000`
- updated_at: `2026-07-06T16:49:16+08:00`
- change_type: `merged`
- previous_overlap: `0.057`

## Summary

Agent（智能体）是一个能够自主感知环境、规划行动并执行任务的智能系统...

## Final Answer
...
```

### 5.5 进阶功能 4：向量检索

#### 功能说明

使用句子向量进行语义检索，能够理解查询的深层含义，而不仅仅是关键词匹配。适合处理以下场景：
- 同义词或近义词匹配（如 "智能体" 和 "Agent"）
- 语义相关但关键词不同（如 "规划" 和 "决策"）
- 模糊查询和自然语言描述

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `common/text_utils.py::get_embedding_model()` | 加载向量化模型 |
| `common/text_utils.py::batch_compute_embeddings()` | 批量计算文本向量 |
| `b5_memory.py::_search_by_vector()` | 计算相似度，排序返回 |
| `b5_memory.py::load_memory_with_vector()` | 向量检索入口函数 |

#### 演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

# 需要先安装依赖
pip install numpy sentence-transformers
export HF_ENDPOINT=https://hf-mirror.com

# 向量检索
python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --query "Agent 系统架构" \
  --search_mode vector \
  --top_k 3 \
  --min_similarity 0.2 \
  --use_global_only true \
  --outdir outputs/B5_advanced
```

#### 输出格式

向量检索结果包含语义相似度分数：

```json
{
  "status": "success",
  "query": "Agent 系统架构",
  "search_mode": "vector",
  "top_k": 3,
  "results": [
    {
      "memory_id": "mem_course_001",
      "title": "Agent 基础概念",
      "similarity": 0.8234,
      "snippet": "Agent 系统通常由模型、工具、记忆和执行循环组成..."
    }
  ],
  "total_matched": 1
}
```

### 5.6 进阶功能 5：错误记忆影响分析

#### 功能说明

对比"错误记忆"和"正确记忆"对 Agent 回答的影响，帮助识别：
- 哪些记忆内容可能误导 Agent
- 错误记忆与正确记忆的内容差异
- 不同记忆对查询的相关性差异

#### 实现路径

| 文件 / 函数 | 作用 |
|---|---|
| `b5_memory.py::analyze_bad_memory_impact()` | 执行影响分析 |
| `common/text_utils.py::compare_text()` | 对比内容差异 |
| `common/text_utils.py::compute_keyword_score()` | 计算相关性分数 |

#### 演示命令

```bash
cd /root/siton-tmp/assignment_B/agent

python3 code/b5_memory.py \
  --config configs/memory.yaml \
  --analyze_bad mem_conversation_conv_000 \
  --analyze_good mem_course_001 \
  --analyze_query "Agent 如何调用工具" \
  --outdir outputs/B5_advanced
```

#### 输出示例

```json
{
  "bad_memory_id": "mem_conversation_conv_000",
  "good_memory_id": "mem_course_001",
  "bad_exists": true,
  "good_exists": true,
  "analysis": {
    "content_overlap": {
      "change_type": "replace",
      "overlap_ratio": 0.013
    },
    "bad_relevance_score": 0.6751,
    "good_relevance_score": 0.6384,
    "recommendation": "Use bad_memory_id may be better"
  }
}
```

---

## 6. 与团队系统的集成说明

### 调用链路

```text
B1 (Agent Runtime) -> B5 (Memory Module) -> 文件系统
```

### 调用方式

| 调用场景 | 调用的函数 | 传入参数 | 返回值 |
|---|---|---|---|
| 任务开始时注入记忆 | `load_memory_advanced()` | memory_ids, global_flag, query | selected_memory_docs |
| 任务完成后保存记忆 | `save_memory()` | conversation_id, messages, trace, answer | memory_id, path |

### 配置依赖

B5 模块需要以下配置文件：
- `configs/memory.yaml`：定义存储路径和索引位置
- `memory/memory_index.json`：记忆索引（自动维护）

### 接口一致性

B5 模块与 B1 的接口已在项目初期约定并保持一致：
- 输入输出均使用 JSON 格式
- memory_id 命名规范：`mem_{type}_{id}`
- 返回结构包含 status、errors 等标准字段

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| 关键词检索对长文档处理较慢 | 每次检索都需要遍历所有文档并计算分数 | 可构建倒排索引加速检索 |
| 记忆更新时冲突检测基于关键词重叠 | 关键词重叠不能完全代表语义相似度 | 可结合向量相似度进行更准确的冲突检测 |
| 记忆摘要基于规则生成 | 未使用 LLM 生成，质量有限 | 可集成 LLM 生成更高质量的摘要 |
