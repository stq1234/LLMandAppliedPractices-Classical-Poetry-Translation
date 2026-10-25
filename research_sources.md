# 方法依据

## 翻译目的

Katharina Reiss and Hans J. Vermeer. *Towards a General Theory of Translational Action: Skopos Theory Explained*. English translation, 2013; German original, 1984. [出版社资料](https://www.routledge.com/Towards-a-GeneralTheory-of-Translational-Action-Skopos-TheoryExplained/Reiss-Vermeer/p/book/9781905763955)。

本实验将受众、使用目的、注释范围与改写权限写入生成提示，把“忠实译介”和“英语诗歌朗读”作为两个明确的任务条件。该设计借鉴翻译目的与任务约定的思想；实验采用的六组提示、诗词片段与评分细则均为本研究自行制定。

## 多维评价与证据

Markus Freitag et al. 2021. [Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87/). *Transactions of the Association for Computational Linguistics*, 9:1460–1474.

论文采用专业译者、完整上下文和基于 MQM 的明确错误分析。本文借鉴“把质量判断落实到具体错误与文本证据”的评价原则，自行制定适用于诗词的含义、意象、语气、英语表达及用途适合度指标。本文的模型评分不属于专业译者人工评价，也不声称复现该论文的 MQM 研究。

Lianmin Zheng et al. 2023. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685). *NeurIPS Datasets and Benchmarks*.

论文讨论位置、冗长和自偏好等模型评价偏差。本文在评分输入中隐藏生成模型与提示条件，统一评分锚点，混洗译文顺序，并预设第二评分模型与逆序复评子集。上述措施用于观察结果的敏感性，不能替代独立的人工翻译评价。

## 接口参数

Ashish Vaswani et al. 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 本文用自注意力的通用形式说明上下文如何参与词元表示，再用条件自回归概率说明生成任务。该原理说明不把原始编码器—解码器 Transformer 的全部结构或超参数直接归于本实验调用的三个闭源型号。

[OpenAI Chat Completions 官方参数说明](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。本项目对接用户指定的本地兼容服务；模型名称与返回名称逐项核对。生成与评价的实际推理强度、生成长度以及未显式指定的采样参数由实验协议保存。

## 文本来源

各诗词的原文链接、选段、异文与简繁转换说明见 [语料来源](data/corpus_sources.md) 和 [结构化语料](data/poems.json)。不向生成模型提供现成英文译本。
