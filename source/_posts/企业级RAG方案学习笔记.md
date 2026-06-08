---
title: 企业级RAG方案学习笔记
date: 2026-06-08 10:30:00
categories: 技术
tags:
  - RAG
  - 大模型
  - 向量数据库
  - AI工程
---

## 前言

RAG（Retrieval-Augmented Generation）是当前企业落地大模型最主流的方案。核心思想：**先检索，再生成**——让大模型基于检索到的真实文档回答问题，而不是凭空编造。

本文从企业级视角，完整梳理 RAG 的离线管线和在线管线。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     RAG 整体架构                         │
├──────────────────────────┬──────────────────────────────┤
│       离线管线            │          在线管线             │
│                          │                              │
│  文档采集 → 分块 → 向量化  │  用户Query → 改写 → 检索      │
│       ↓                  │       ↓                      │
│  向量库写入 → 索引构建     │  向量检索 + 关键词检索         │
│                          │       ↓                      │
│                          │  重排序(Rerank) → 上下文拼接    │
│                          │       ↓                      │
│                          │  LLM 生成 → 后处理返回          │
└──────────────────────────┴──────────────────────────────┘
```

---

## 一、离线管线（Offline Pipeline）

离线管线的核心任务：**把非结构化文档变成可检索的向量索引**。

### 1.1 文档采集与解析

企业数据源多样，需要统一解析：

| 数据源 | 解析工具 | 注意事项 |
|--------|---------|---------|
| PDF | PyMuPDF / pdfplumber / MinerU | 表格、图片、公式需特殊处理 |
| Word/PPT | python-docx / python-pptx | 保留层级结构 |
| 网页/Markdown | BeautifulSoup / markdown-it | 去除导航、广告等噪音 |
| 数据库/API | 自定义 Connector | 结构化数据转自然语言描述 |
| 图片/扫描件 | OCR (PaddleOCR / Tesseract) | 精度依赖清晰度 |

**企业级要点：**
- 统一输出为结构化 JSON，保留标题层级、表格、元数据
- 对敏感数据做脱敏处理（手机号、身份证号等）
- 记录文档来源、版本、更新时间，便于增量更新

### 1.2 文档分块（Chunking）

分块质量直接决定检索效果，是 RAG 最关键的环节之一。

**常见策略：**

```python
# 1. 固定长度分块（最简单，效果最差）
chunk_size = 512
overlap = 64

# 2. 语义分块（基于句子/段落边界）
# 按 \n\n 或句子切分，保证语义完整

# 3. 递归字符分割（LangChain 默认）
RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", "！", "？", "；", " "]
)

# 4. 基于文档结构分块（推荐）
# 按 Markdown 标题 / HTML 标签 / PDF 章节切分
```

**企业级实践：**

| 策略 | 适用场景 | 优缺点 |
|------|---------|--------|
| 固定长度 | 快速原型 | 简单但会切断语义 |
| 语义分块 | 通用场景 | 平衡效果与复杂度 |
| 结构化分块 | 文档有清晰层级 | 效果最好，依赖解析质量 |
| 父子分块 | 需要精确定位+宽上下文 | 小块检索，大块送 LLM |
| Agentic Chunking | 高质量要求 | LLM 辅助分块，成本高 |

**Chunk 参数经验值：**
- 通用文档：chunk_size=512 tokens, overlap=64 tokens
- 法律/合同：chunk_size=1024+, 需保留完整条款
- FAQ/知识库：一问一答为一个 chunk

### 1.3 向量化（Embedding）

将文本 chunk 转为高维向量，用于语义相似度计算。

**主流 Embedding 模型：**

| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3-large | 3072 | 商用标杆，支持降维 |
| OpenAI text-embedding-3-small | 1536 | 性价比高 |
| BGE-M3 (BAAI) | 1024 | 开源最强，支持多语言 |
| GTE-Qwen2 | 768-1536 | 阿里开源，中文优秀 |
| Cohere embed-v4 | 1024 | 多语言，支持 int8 量化 |
| Jina-embeddings-v3 | 1024 | 支持 task-specific LoRA |

**企业级选型建议：**
- 中文场景优先选 BGE-M3 或 GTE-Qwen2
- 需要私有化部署选开源模型 + GPU 推理
- 向量维度与存储/检索成本正相关，不是越大越好

```python
# 使用 Sentence-Transformers 加载本地模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
vectors = model.encode(chunks, normalize_embeddings=True)
```

### 1.4 向量数据库

存储和检索向量的核心组件。

| 数据库 | 类型 | 特点 |
|--------|------|------|
| Milvus | 专用向量库 | 分布式，十亿级，企业首选 |
| pgvector | PG 扩展 | 与现有 PG 生态融合，运维简单 |
| Qdrant | 专用向量库 | Rust 实现，性能优秀 |
| Weaviate | 专用向量库 | 内置混合搜索 |
| Elasticsearch + dense_vector | 搜索引擎扩展 | 已有 ES 集群可复用 |
| Chroma | 轻量级 | 适合原型，不适合生产 |

**索引类型：**

```
FLAT    — 暴力搜索，100% 精确，适合 <10万
IVF_FLAT — 倒排索引 + 暴力，适合百万级
IVF_PQ  — 倒排 + 乘积量化，适合千万级，有精度损失
HNSW    — 图索引，查询快，内存占用大，适合百万级（推荐）
```

### 1.5 增量更新策略

```python
# 文档变更检测
def incremental_update(doc_id, content_hash):
    existing = vector_db.get(doc_id)
    if existing and existing.content_hash == content_hash:
        return  # 无变化，跳过

    # 删除旧向量
    vector_db.delete(filter={"doc_id": doc_id})

    # 重新分块 + 向量化
    chunks = chunk(content)
    vectors = embed(chunks)
    vector_db.upsert(chunks, vectors, metadata={"doc_id": doc_id})
```

---

## 二、在线管线（Online Pipeline）

在线管线的核心任务：**接收用户 Query，经过一系列处理，返回高质量回答**。

### 2.1 Query 改写（Query Rewriting）

用户的原始 query 往往不适合直接检索，需要改写优化。

**核心策略：**

```python
# 1. 查询扩展（Query Expansion）
# 生成多个语义等价的 query，提高召回
prompt = """请将以下用户问题改写为3个不同的搜索查询，保持语义一致：
用户问题：{query}
输出格式：每行一个查询"""

# 2. HyDE（Hypothetical Document Embeddings）
# 让 LLM 先生成一个"假答案"，用假答案的向量去检索
prompt = """请根据以下问题，写一段可能的答案（100字左右）：
问题：{query}"""
# 用这段假答案的 embedding 去做向量检索，往往比原始 query 效果更好

# 3. 多查询融合（Multi-Query Fusion）
# 同时用原始 query + 改写 query 检索，合并去重
```

**Step-Back Prompting：**
```
原始问题："特斯拉2024年Q3在中国的销量是多少？"
回退问题："特斯拉2024年在中国的销售数据"
→ 更宽泛的检索范围，提高召回率
```

### 2.2 检索策略（Retrieval）

**混合检索（Hybrid Search）是企业标配：**

```python
# 向量检索（语义相似）
vector_results = vector_db.search(query_embedding, top_k=20)

# 关键词检索（精确匹配）
bm25_results = bm25_index.search(query_text, top_k=20)

# 融合策略
# 1. RRF (Reciprocal Rank Fusion) — 推荐
def rrf_fusion(results_lists, k=60):
    scores = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 2. 加权融合
final_score = alpha * vector_score + (1 - alpha) * bm25_score
```

**为什么需要混合检索？**
- 纯向量检索：擅长语义匹配，但对专有名词、编号、关键词精确匹配弱
- 纯关键词检索：精确匹配强，但不理解同义词和语义
- 混合检索：两者互补，企业场景必备

### 2.3 重排序（Reranking）

初步检索返回 top_k=20~50，用 Reranker 精排后取 top_n=5~10。

**主流 Reranker：**

| 模型 | 特点 |
|------|------|
| Cohere Rerank v3 | 商用标杆，API 调用 |
| BGE-reranker-v2-m3 | 开源最强，支持多语言 |
| bce-reranker-base_v1 | 网易开源，中文优秀 |
| Jina Reranker v2 | 支持长文档 |

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# 对 query-doc 对打分
pairs = [(query, doc.content) for doc in candidates]
scores = reranker.predict(pairs)

# 取 top_n
top_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

**Reranker vs Embedding 的区别：**
- Embedding：双塔模型，query 和 doc 独立编码，可离线预计算 doc 向量
- Reranker：交叉编码器，query 和 doc 拼接输入，精度更高但速度慢
- 最佳实践：Embedding 粗筛（快）→ Reranker 精排（准）

### 2.4 上下文构建

```python
def build_context(docs, max_tokens=3000):
    context_parts = []
    total_tokens = 0

    for i, doc in enumerate(docs):
        doc_text = f"[{i+1}] 来源：{doc.source}\n{doc.content}\n"
        doc_tokens = count_tokens(doc_text)

        if total_tokens + doc_tokens > max_tokens:
            break

        context_parts.append(doc_text)
        total_tokens += doc_tokens

    return "\n---\n".join(context_parts)
```

**注意事项：**
- 控制上下文总长度，留足空间给 system prompt 和用户问题
- 按相关性分数从高到低排列
- 附带来源信息，方便 LLM 引用和用户溯源

### 2.5 LLM 生成

```python
system_prompt = """你是一个专业的知识助手。请严格基于以下参考文档回答用户问题。
要求：
1. 只使用参考文档中的信息回答，不要编造
2. 如果参考文档中没有相关信息，请明确告知用户
3. 回答时引用来源编号，如 [1] [2]
4. 回答简洁、准确、结构化

参考文档：
{context}
"""

user_prompt = f"用户问题：{query}"
```

---

## 三、企业级进阶方案

### 3.1 多路召回 + 路由

```python
# 根据 query 类型路由到不同的知识库
def route_query(query):
    if is_structured_query(query):
        return sql_agent(query)          # 结构化查询走 Text-to-SQL
    elif is_faq_query(query):
        return faq_search(query)         # FAQ 走精确匹配
    else:
        return rag_pipeline(query)       # 通用问题走 RAG
```

### 3.2 引用溯源与幻觉检测

```python
# 生成后验证：检查回答中的事实是否有文档支撑
def verify_citations(answer, context_docs):
    # 1. 提取回答中的关键断言
    claims = extract_claims(answer)

    # 2. 检查每个断言是否有文档支持
    for claim in claims:
        support = retrieve(claim, context_docs)
        if max_support_score(support) < THRESHOLD:
            mark_as_uncertain(claim)

    # 3. 对不确定的断言添加 disclaimer
    return add_disclaimers(answer)
```

### 3.3 评估体系

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| Recall@K | 前 K 个检索结果中包含正确文档的比例 | 检索质量 |
| MRR | 正确文档排名的倒数 | 排序质量 |
| Faithfulness | 回答是否忠于检索到的文档 | 幻觉率 |
| Answer Relevancy | 回答与问题的相关程度 | 回答质量 |
| Context Precision | 检索到的文档是否与问题相关 | 噪音率 |

**推荐使用 RAGAS 框架做自动化评估：**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
```

### 3.4 性能优化

```
1. 向量缓存：相同 query 缓存检索结果（TTL 5~10 min）
2. Embedding 量化：float32 → int8，减少 75% 内存
3. 预过滤：先按元数据（时间、类别）过滤，再做向量检索
4. 流式输出：LLM 生成使用 SSE 流式返回，减少首 token 延迟
5. 异步管线：Query 改写、向量检索、BM25 检索并行执行
```

### 3.5 生产架构参考

```
                    ┌─────────────┐
                    │   Nginx LB  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐┌────┴─────┐┌─────┴─────┐
        │  RAG API  ││ RAG API  ││ RAG API   │
        │  Node 1   ││ Node 2   ││ Node 3    │
        └─────┬─────┘└────┬─────┘└─────┬─────┘
              │            │            │
    ┌─────────┴────────────┴────────────┴─────────┐
    │              消息队列 (RabbitMQ/Kafka)        │
    │         (异步文档处理 / 增量索引任务)            │
    └──────────────────────┬───────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────┴─────┐   ┌──────┴──────┐   ┌─────┴─────┐
   │ Milvus    │   │ PostgreSQL  │   │ Redis     │
   │ 向量数据库 │   │ 元数据/文档  │   │ 缓存      │
   └───────────┘   └─────────────┘   └───────────┘
```

---

## 四、常见坑与经验

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 检索到但 LLM 没用 | 上下文过长，LLM 忽略中间内容 | 控制上下文长度，重要信息放首位 |
| 检索结果不相关 | 分块太大/太小，Embedding 质量差 | 调整 chunk size，换更好的 Embedding |
| 回答幻觉 | LLM 编造信息 | 加强 prompt 约束 + 幻觉检测 |
| 专有名词搜不到 | 向量检索不擅长精确匹配 | 混合检索，加入 BM25 |
| 增量更新延迟 | 全量重建索引慢 | 增量更新 + 异步处理 |
| 多轮对话丢失上下文 | 每轮独立检索 | 加入对话历史改写 query |

---

## 五、技术选型速查表

| 组件 | 推荐方案 | 备选 |
|------|---------|------|
| Embedding | BGE-M3 / GTE-Qwen2 | OpenAI text-embedding-3 |
| 向量数据库 | Milvus / pgvector | Qdrant |
| Reranker | BGE-reranker-v2-m3 | Cohere Rerank |
| BM25 | Elasticsearch | Whoosh / 自研 |
| LLM | GPT-4o / Claude / DeepSeek | 开源: Qwen2.5 / Llama3 |
| 编排框架 | LangChain / LlamaIndex | 自研 |
| 评估 | RAGAS | DeepEval |

---

## 总结

企业级 RAG 的关键不在于某个单一环节，而在于**全链路的工程化**：

1. **离线管线**：分块质量 > Embedding 模型 > 向量数据库
2. **在线管线**：Query 改写 > 混合检索 > Rerank > Prompt 工程
3. **持续迭代**：建立评估体系，用数据驱动优化

RAG 不是银弹，但它是当前大模型落地最务实的方案。掌握这套体系，足以应对绝大多数企业知识库场景。

---

> 参考资料：
> - [LangChain RAG 文档](https://python.langchain.com/docs/tutorials/rag/)
> - [RAGAS 评估框架](https://docs.ragas.io/)
> - [Milvus 向量数据库](https://milvus.io/docs)
> - [BGE 系列模型](https://huggingface.co/BAAI)
