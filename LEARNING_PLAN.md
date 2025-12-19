# 🎓 LangChain 系统学习方案

---

## 📋 学习总览

本学习方案共分为 **6个阶段，25个主题**，从基础到高级循序渐进。每完成一个主题，请在对应的checkbox中打钩 ✅

**预计学习时间**：35-45小时（根据个人基础调整）

**重要说明**：
- 本课程基于 **LangChain v1** 和 **LangGraph v1**
- LangChain Agents 是基于 LangGraph 构建的（但不需要了解 LangGraph 细节）

---

## 🎯 第一阶段：基础组件（Foundation）

> **目标**：掌握LangChain的基础构建块和核心概念

### [ ] 01. Chat Models（聊天模型与核心方法）
**文件**：`notebook/01_models_chat.ipynb`

**学习内容**：
- Chat Models基础概念
- 初始化ChatOpenAI模型
- 三种消息类型：SystemMessage、HumanMessage、AIMessage
- 三种核心方法：invoke()、stream()、batch()
- 流式输出的原理和应用
- Token使用统计
- 参数配置和最佳实践

**核心要点**：
- Chat Model有三种核心调用方法
- stream()提供实时反馈，提升用户体验
- batch()并行处理多个请求
- 流式输出是现代AI应用的标配

---

### [ ] 02. Prompts & Messages（提示词工程）
**文件**：`notebook/02_prompts.ipynb`

**学习内容**：
- Prompt Templates的概念和作用
- ChatPromptTemplate的使用
- from_template() vs from_messages()
- 变量注入和动态提示词
- Partial Variables（部分变量填充）
- Few-shot Learning（少样本学习）
- 提示词工程最佳实践

**核心要点**：
- Prompt Template让提示词变成可编程组件
- Few-shot learning通过示例引导模型
- 提示词工程是影响AI输出质量的关键
- ChatPromptTemplate是最常用的模板类

---

### [ ] 03. LCEL基础（LangChain Expression Language）
**文件**：`notebook/03_lcel_basics.ipynb`

**学习内容**：
- 什么是LCEL？为什么重要？
- Runnable接口和协议
- Pipe操作符（|）深入讲解
- RunnablePassthrough - 数据传递
- RunnableParallel - 并行执行
- RunnableLambda - 自定义函数
- Chain的组合模式

**核心要点**：
- LCEL是LangChain的核心语法
- 所有组件都实现Runnable接口
- Pipe操作符（|）实现组件链接
- 支持invoke, stream, batch等标准方法
- LCEL让代码简洁且可组合

---

### [ ] 04. Structured Output（结构化输出）
**文件**：`notebook/04_structured_output.ipynb`

**学习内容**：
- 为什么需要结构化输出？
- with_structured_output() 方法
- Pydantic模型定义输出schema
- TypedDict作为简化替代
- JSON Schema方式
- include_raw参数获取原始响应
- 嵌套结构和复杂schema
- 传统Output Parser（简单了解）

**核心要点**：
- with_structured_output()是标准方式
- Pydantic模型提供类型安全和验证
- 结构化输出让AI输出可编程
- 传统Output Parser已被替代

---

### [ ] 05. Tools & Tool Calling（工具与函数调用）
**文件**：`notebook/05_tools.ipynb`

**学习内容**：
- Tools的概念和作用
- 使用@tool装饰器创建工具
- Tool schema和参数定义
- bind_tools绑定工具到模型
- Tool Calling工作流程
- 并行工具调用
- 强制工具选择（tool_choice）
- ToolMessage处理工具结果

**核心要点**：
- Tools让AI能够执行实际操作
- 工具是带有schema的可调用函数
- 模型决定何时以及如何调用工具
- Tool calling是Agent的基础

---

## 🔍 第二阶段：检索增强生成（RAG）

> **目标**：掌握完整的RAG技术栈，从文档处理到智能检索

### [ ] 06. Document Processing（文档处理）
**文件**：`notebook/06_document_processing.ipynb`

**学习内容**：
- Document对象结构
- Document Loaders（文档加载器）
- 加载PDF、TXT、CSV、网页等格式
- Text Splitters（文本分割器）
- CharacterTextSplitter
- RecursiveCharacterTextSplitter（推荐）
- 分割策略和chunk size选择
- chunk overlap的作用

**核心要点**：
- Document Loader统一各种数据源
- Text Splitter将大文档分割成小块
- 合适的chunk size影响检索质量
- RecursiveCharacterTextSplitter适合大多数场景
- 为RAG的Indexing步骤做准备

---

### [ ] 07. Embeddings & Vector Stores（向量化与向量存储）
**文件**：`notebook/07_embeddings_vectorstores.ipynb`

**学习内容**：
- Embeddings的原理和作用
- OpenAIEmbeddings使用
- 向量相似度原理
- Vector Stores概念
- FAISS向量数据库（本地）
- Chroma向量数据库
- 相似度搜索方法
- 向量存储的持久化

**核心要点**：
- Embeddings将文本转换为向量
- 向量存储实现语义搜索
- 相似度搜索是RAG的基础
- 向量数据库选择影响性能和成本
- 完成RAG的Indexing和Retrieval基础

---

### [ ] 08. RAG基础（2-Step RAG）
**文件**：`notebook/08_rag_basic.ipynb`

**学习内容**：
- RAG的概念和原理
- 为什么需要RAG？解决什么问题？
- RAG的三个核心步骤：Indexing、Retrieval、Generation
- Indexing：文档加载、分割、向量化、存储
- Retrieval：使用基础retriever检索相关文档
- Generation：结合检索结果生成答案
- 使用LCEL构建简单RAG Chain
- 使用as_retriever()快速创建检索器
- prompt | retriever | model模式
- 完整RAG应用的实现

**核心要点**：
- RAG = Retrieval + Augmented + Generation
- 2-Step RAG先检索后生成（确定性流程）
- RAG让AI能回答私有数据问题
- 使用|操作符串联各个步骤
- 理解完整的RAG工作流程

---

### [ ] 09. Retrievers（检索器深入）
**文件**：`notebook/09_retrievers.ipynb`

**学习内容**：
- Retriever接口和Runnable协议
- VectorStoreRetriever详解
- 检索参数：search_type和search_kwargs
- similarity（相似度检索）
- MMR（最大边际相关性）
- similarity_score_threshold（阈值过滤）
- MultiQueryRetriever（多查询检索）
- ContextualCompressionRetriever（上下文压缩）
- ParentDocumentRetriever（父文档检索）
- 自定义Retriever
- 检索策略对比和选择

**核心要点**：
- Retriever是Runnable，可以用|组合
- 不同检索策略适用不同场景
- MMR平衡相关性和多样性
- 检索质量直接影响RAG效果
- 优化检索是提升RAG性能的关键

---

### [ ] 10. RAG进阶（Agentic RAG）
**文件**：`notebook/10_rag_advanced.ipynb`

**学习内容**：
- Agentic RAG vs 2-Step RAG的区别
- Agent动态决策何时检索
- 使用@tool创建retriever工具
- RAG Agent的构建（create_agent）
- 多数据源RAG（多个知识库）
- 查询改写和优化技术
- Self-Query（自查询）
- Hybrid RAG模式
- RAG评估指标和方法
- 混合检索策略
- RAG Chain的调试和优化

**核心要点**：
- Agentic RAG让Agent决定何时检索
- 更灵活但需要更多token
- 适合复杂的多步骤推理场景
- Agent可以多次调用检索工具
- 结合2-Step和Agentic的优势

---

## 🤖 第三阶段：智能体（Agents）

> **目标**：构建能自主决策和使用工具的AI Agent

### [ ] 11. Agents基础（create_agent）
**文件**：`notebook/11_agents_basic.ipynb`

**学习内容**：
- Agent的概念和架构
- create_agent() API
- ReAct模式（Reasoning + Acting）
- Agent的执行循环
- Agent的思考过程
- system_prompt参数
- Agent基于LangGraph的架构
- LangChain Agent vs LangGraph的区别

**核心要点**：
- Agent能自主决策和行动
- create_agent是标准API
- Agent自动基于LangGraph构建
- 不需要了解LangGraph细节即可使用
- 适合90%的Agent使用场景

---

### [ ] 12. Agent with Tools
**文件**：`notebook/12_agents_tools.ipynb`

**学习内容**：
- 为Agent配备工具
- 工具的选择和使用逻辑
- 工具调用链
- 错误处理和重试
- Human-in-the-loop集成
- 工具的artifacts
- 实战：搜索+计算Agent
- 实战：SQL查询Agent

**核心要点**：
- 工具扩展Agent的能力
- Agent自动决定使用哪个工具
- Human-in-the-loop增加控制
- 工具越多，Agent越强大

---

### [ ] 13. Structured Output in Agents
**文件**：`notebook/13_agent_structured_output.ipynb`

**学习内容**：
- Agent的结构化输出
- response_format参数
- ToolStrategy - 人工工具调用方式
- ProviderStrategy - 原生结构化输出
- 结构化输出在Agent中的应用
- 错误处理策略
- 实战：数据提取Agent

**核心要点**：
- Agent可以返回结构化数据
- ToolStrategy适用所有支持工具的模型
- ProviderStrategy更可靠但支持有限
- structured_response字段包含解析结果

---

### [ ] 14. Multi-Agent系统
**文件**：`notebook/14_multi_agent.ipynb`

**学习内容**：
- Multi-Agent架构模式
- Supervisor模式（Tool Calling）
- Handoffs模式（转移控制）
- Agent间通信
- 任务协调和编排
- Subgraph作为Agent
- 实战：多Agent协作系统

**核心要点**：
- 多个Agent协作完成复杂任务
- Supervisor模式：中心化控制
- Handoffs模式：去中心化转移
- 可以用LangGraph实现更复杂编排

---

## 💾 第四阶段：持久化与状态管理（Persistence & State）

> **目标**：为Agent添加记忆能力，实现有状态的智能对话

### [ ] 15. Persistence基础（Checkpointing）
**文件**：`notebook/15_persistence_basics.ipynb`

**学习内容**：
- LangGraph的持久化概念
- Checkpointer原理
- MemorySaver / InMemorySaver
- thread_id的使用
- 短期记忆（thread-level persistence）
- 状态检查点
- get_state()和update_state()
- 实战：带记忆的Agent

**核心要点**：
- LangChain Agents基于LangGraph构建
- checkpointer实现持久化
- thread_id标识会话
- 自动保存每个步骤的状态
- 短期记忆让Agent能进行多轮对话

---

### [ ] 16. Cross-Thread Memory（Store）
**文件**：`notebook/16_cross_thread_memory.ipynb`

**学习内容**：
- Store接口
- 跨线程的长期记忆
- InMemoryStore使用
- 用户信息持久化
- namespace的概念
- put()、get()、search()方法
- 实战：记住用户偏好的Agent

**核心要点**：
- checkpointer只能在thread内
- Store实现跨thread的记忆
- 适合存储用户档案、偏好等
- 长期记忆的实现方式

---

### [ ] 17. Memory Management（消息管理）
**文件**：`notebook/17_memory_management.ipynb`

**学习内容**：
- 消息历史管理
- Trim messages（修剪消息）
- Summarize messages（总结消息）
- RemoveMessage删除消息
- 上下文窗口管理
- 传统Memory类简介（补充）
- ConversationBufferMemory等（了解即可）

**核心要点**：
- 长对话会超过上下文窗口
- 修剪和总结是常用策略
- RemoveMessage从状态中删除
- 传统Memory类了解即可

---

## 🚀 第五阶段：生产实践（Production）

> **目标**：将原型系统变成生产就绪的应用

### [ ] 18. 高级Streaming（Advanced Streaming）
**文件**：`notebook/18_advanced_streaming.ipynb`

**学习内容**：
- streamEvents()方法
- 语义事件流
- 流式输出的回调系统
- 中间步骤的流式输出
- Agent流式输出
- 多Agent系统的流式控制
- 流式输出的错误处理
- stream_mode参数

**核心要点**：
- streamEvents()提供更细粒度的控制
- 可以流式输出Agent的思考过程
- 适合复杂的多步骤工作流
- 提升用户体验

---

### [ ] 19. 异步与并发（Async & Concurrency）
**文件**：`notebook/19_async_patterns.ipynb`

**学习内容**：
- 异步编程基础
- ainvoke异步调用
- astream异步流式
- 并发执行多个任务
- 异步批处理
- 性能优化技巧
- 实战：高并发API

**核心要点**：
- 异步提高系统吞吐量
- 适合I/O密集型任务
- 所有方法都有异步版本（a前缀）
- 合理使用避免过度并发

---

### [ ] 20. Error Handling & Retry
**文件**：`notebook/20_error_handling.ipynb`

**学习内容**：
- 常见错误类型
- try-except错误捕获
- 重试策略和中间件
- 降级处理
- 超时控制
- 错误日志记录
- 生产环境最佳实践

**核心要点**：
- 生产环境必须处理各种错误
- 重试机制应对临时故障
- 优雅降级保证服务可用性
- 完善的日志便于排查

---

### [ ] 21. Evaluation & Testing
**文件**：`notebook/21_evaluation.ipynb`

**学习内容**：
- 评估的重要性
- 创建测试数据集
- RAG评估指标（相关性、准确性）
- Agent评估方法
- LangSmith evaluate()方法
- A/B测试
- 自动化测试流程

**核心要点**：
- 评估确保系统质量
- 使用多维度评估指标
- LangSmith提供评估工具
- 持续评估和迭代改进

---

### [ ] 22. LangSmith集成
**文件**：`notebook/22_langsmith_integration.ipynb`

**学习内容**：
- LangSmith简介
- Tracing调用链追踪
- 性能监控
- Prompt管理和版本控制
- 在线评估
- 数据集管理
- 调试技巧

**核心要点**：
- LangSmith是官方观测平台
- Tracing可视化执行流程
- 便于调试和优化
- 团队协作和Prompt管理

---

## 🎨 第六阶段：综合项目（Projects）

> **目标**：整合所学知识，构建完整的实战项目

### [ ] 23. 项目：智能问答系统
**文件**：`notebook/23_project_qa_system.ipynb`

**项目描述**：
构建一个基于企业文档的智能问答系统

**技术栈**：
- Document Loaders + Text Splitters
- Embeddings + Vector Store
- Retriever + RAG Chain/Agent
- Structured Output
- Evaluation

**功能**：
- 上传和索引文档
- 自然语言问答
- 引用来源
- 答案评分
- 流式输出

---

### [ ] 24. 项目：对话机器人
**文件**：`notebook/24_project_chatbot.ipynb`

**项目描述**：
构建一个具有记忆能力的多功能对话机器人

**技术栈**：
- Chat Models + Prompts
- create_agent
- Tools (天气、计算器、搜索)
- Checkpointer持久化
- Streaming

**功能**：
- 多轮对话
- 上下文理解
- 工具调用
- 个性化响应
- 会话管理

---

### [ ] 25. 项目：文档分析系统
**文件**：`notebook/25_project_document_analysis.ipynb`

**项目描述**：
构建一个自动化文档分析和总结系统

**技术栈**：
- Document Processing
- Structured Output
- LCEL Chains
- Batch Processing
- Multi-Agent（可选）

**功能**：
- 批量文档处理
- 自动摘要生成
- 关键信息提取
- 结构化输出
- 报告生成

---

### 重要概念优先级

**必须掌握**：
- ✅ Chat Models的三种方法（invoke/stream/batch）
- ✅ LCEL和Pipe操作符（|）
- ✅ Structured Output（with_structured_output）
- ✅ Tools和Tool Calling
- ✅ RAG的完整流程（Document → Embeddings → Retrieval → Generation）
- ✅ create_agent API
- ✅ Checkpointing持久化

**建议掌握**：
- 📝 Few-shot Learning
- 📝 高级Retriever（MMR、Compression等）
- 📝 Async异步模式
- 📝 Multi-Agent模式
- 📝 Store跨线程记忆

**可选了解**：
- 🔸 传统Memory类
- 🔸 LangGraph底层API
- 🔸 LangSmith高级功能

---

## 📚 参考资源

### 官方文档
- **LangChain Python文档**：https://docs.langchain.com/oss/python/
- **LangChain API参考**：https://python.langchain.com/api_reference/
- **LangGraph文档**：https://docs.langchain.com/oss/python/langgraph/
- **LangSmith文档**：https://docs.langchain.com/langsmith/

### 版本说明
- **LangChain v1**：https://docs.langchain.com/oss/python/releases/langchain-v1
- **LangGraph v1**：https://docs.langchain.com/oss/python/releases/langgraph-v1

### 相关资源
- **Models.dev**：https://models.dev/ - 模型能力数据库
- **LangChain Hub**：Prompt模板分享平台

---
