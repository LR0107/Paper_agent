# LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models

**EMNLP 2023**

**Microsoft**

核心问题：如何在不明显降低大模型性能的前提下，把超长 Prompt 压缩得更短，从而降低推理成本、提升速度。

## 一.研究背景

近年来，大模型能力提升很大，但同时 Prompt 也越来越长：

- Chain-of-Thought（思维链推理）
- In-Context Learning（上下文学习）
- RAG（检索增强生成）
- 长对话历史

都会导致输入长度迅速增加。

Prompt 越长：推理成本更高、推理速度更慢（Transformer 计算量随序列长度增长）、上下文窗口受限

### 现有方法的问题

此前代表工作是：**Selective Context (2023)**

思路：

- 用小语言模型计算每个 Token 的信息量
- 删除信息量低的部分

但作者认为它有两个缺点：

1.忽略 Token 之间的依赖关系

比如：Tom bought 10 apples.He sold 6.

如果删除了第一句：He sold 6.

后面就无法推理。

2.小模型和大模型分布不一致

用于压缩的模型：GPT2

真正执行任务的是：GPT-3.5、Claude

两者关注的信息可能不同。

---

## 二.研究方法

### 1.Budget Controller

Prompt中不同部分的重要性不同：Instruction、Demonstration、Question

问题必须保留、指令必须保留、示例可能存在冗余

没有直接删 Token，而是先删除整段示例。

例如：8个Few-shot示例

变成：只保留2个最重要示例

这样可以保持语义完整。

**如何判断哪个示例重要？**

利用小模型计算：Perplexity(PPL)

PPL高：信息量大；PPL低：信息量小

优先保留 PPL 高的示例。

具体：先用小模型计算每个 Demo 的 PPL，按 PPL 从高到低排序，再根据目标压缩率算出最多允许保留多少 Demo Token，依次保留高 PPL 示例，然后从排序后的列表开始累计求和，超出预算的示例直接丢弃。

### 2.迭代Token压缩

Selective Context的问题

原方法：逐Token计算重要性

例如：The cat sat on the mat.

可能删除：The on the

得到：cat sat mat

虽然短了，但会破坏上下文。

**分段压缩**

先把 Prompt 分成多个 Segment：每次压缩一个 Segment。

**迭代压缩**

压缩第2段时：不仅参考原文，还参考第1段压缩后的结果一起计算

输入：**Compressed Segment1  + Original Segment2**

好处：保留 Token 之间的依赖关系。

例如：

原文：John has 12 apples.He gives away 5.

压缩第二句时：系统仍知道He = John

从而避免关键信息丢失。

## 3.Distribution Alignment

使用**Instruction Tuning**

让小模型（压缩模型）学习 GPT-3.5 （执行模型）的输出风格。

过程：

Prompt
     ↓
GPT-3.5生成答案
     ↓
拿这些数据训练小模型

这样：小模型判断的重要Token ≈ GPT-3.5认为的重要Token

的本质上是知识蒸馏（Distillation）

---

### 实现细节

1.先删示例 ，再删Token

2.数据集Prompt结构都是已知的，不是 LLM 划分。

3.Instruction 和 Question 的压缩率是人工设定；Demonstration 的压缩率通过公式计算出来。

4.根据 PPL 保留 Token不是简单 Top-K 排序：先计算所有 Token 的 PPL，然后根据目标压缩率，求一个阈值 。

---

### 工作流程

**1. Budget Controller（预算分配）**

- 将 Prompt 分为 Instruction、Demonstrations、Question。
- 给不同部分分配不同压缩率：
  - Question 和 Instruction 尽量保留；
  - Demonstrations 压缩更多。

**2. 粗粒度压缩**

- 用小模型计算每个示例（Demo）的重要性（PPL）。
- 优先保留信息量大的示例，删除冗余示例。

**3. 迭代 Token 压缩**

- 将剩余 Prompt 分段。
- 逐段压缩，每压缩完一段，就把压缩结果作为后续段的上下文。
- 根据 Token 的 PPL 保留关键信息，删除不重要 Token。

**4. Distribution Alignment**

- 用 GPT-3.5 等大模型生成数据微调小模型。
- 让小模型学会大模型关注哪些内容，提高压缩质量。

**5. 推理**

- 将压缩后的 Prompt 输入 GPT-3.5、Claude 等大模型。
  
  先分配压缩预算 → 删除冗余示例 → 用小模型分段迭代筛选关键 Token → 对齐大模型偏好

---

## 三.实验设计

### 实验设置

**数据集**

| 数据集           | 任务   |
| ------------- | ---- |
| GSM8K         | 数学推理 |
| BBH           | 复杂推理 |
| ShareGPT      | 对话   |
| Arxiv-March23 | 长文总结 |

**模型**

目标大模型：

- GPT-3.5-Turbo
- Claude-v1.3

压缩模型：

- Alpaca-7B
- GPT2-Alpaca

**baseline**

Sentence Selection：直接删句子

Selective Context：当时SOTA方法

GPT-4 Generation：让GPT-4自己总结Prompt

LLMLingua

### 实验结果

**GSM8K（数学推理）**

| 方法                | EM    |
| ----------------- | ----- |
| Full Prompt       | 78.85 |
| Selective Context | 53.98 |
| LLMLingua         | 79.08 |

压缩率：5×

结果：比原Prompt还高

**更高压缩率**

14×：77.41

20×：77.33

仅比原始下降约：1.5分

**ShareGPT/Arxiv**

9x and 3.3x 更高的BERTScore F1

**消融实验**

| 去掉模块                | EM    |
| ------------------- | ----- |
| 完整模型                | 79.08 |
| 去掉Budget Controller | 73.62 |
| 去掉ITPC              | 72.93 |
| 去掉Alignment         | 78.62 |

Budget Controller、Iterative Compression——这两个模块贡献最大。

EM（Exact Match）：模型输出与标准答案完全一致，则记为 1；否则记为 0。

最后对所有样本求平均。

统计生成长度：包括输入输出token（都下降）

测了端到端耗时：压缩耗时 + LLM推理耗时

| 压缩比 | Speedup |
| --- | ------- |
| 2×  | 1.7×    |
| 5×  | 3.3×    |
| 10× | 5.7×    |

统计了 API 花费

| 方法        | 成本($) |
| --------- | ----- |
| 原始Prompt  | 5.2   |
| LLMLingua | 0.5   |

---

## 四.总结

### 创新点

1.LLMLingua 第一次提出：先粗压缩，再细压缩

Few-shot Prompt 里最大的冗余往往不是 Token，而是整个demo示例。

2.提出迭代 Token 压缩

因为推理链本来就是依赖关系。第二句的重要性必须依赖第一句。

### 不足

1.Alignment部分贡献有限 

消融实验：79.08→78.62 只掉：0.46

这说明：Distribution Alignment并不是主要贡献来源。

2.论文自己承认：25× 30×以后性能快速下降。

说明压缩存在上限，并不是无限压缩。

3.PPL ≠ 真正的重要性

作者假设：高PPL=高信息量=重要

其实PPL 只是 Importance 的近似指标。

---

## Q&A

**1.为什么用 PPL 能判断 Token 是否重要？**

作者的基本假设是：PPL 越高，说明该 Token 越难被语言模型预测，因此携带的信息量越大；PPL 越低，则往往是容易预测的冗余内容。基于信息论视角，保留高 PPL Token 可以在压缩 Prompt 的同时尽可能保留关键语义信息。不过PPL 只是信息量的近似指标，并不完全等同于任务重要性。

**2.为什么要先压缩 Demonstration，再压缩 Token？**

因为 Few-shot Prompt 中最大的冗余通常来自示例本身，而不是单个 Token。如果直接进行 Token 压缩，容易把每个示例压缩得支离破碎，破坏推理过程。LLMLingua 先通过 Budget Controller 删除低价值示例，保留完整的高价值示例，再进行 Token 级压缩，这样能够更好地保留示例中的推理逻辑和语义完整性。

**3.实验为什么选择 GSM8K 和 BBH？**

因为这两个数据集主要考察推理能力和 In-Context Learning 能力。此前很多 Prompt Compression 工作主要评测摘要和对话任务，而作者希望验证 Prompt 压缩后是否还能保持复杂推理能力，因此选择了 GSM8K 和 BBH 作为评测集。

**4.压缩器本身也要计算，会不会得不偿失？**

作者专门分析了这一问题。虽然压缩器需要额外运行一个小模型，但其规模远小于 GPT-3.5 等大模型。实验表明，在计算压缩开销后，系统整体仍然可以获得约 1.7× 到 5.7× 的推理加速，并显著降低 API 成本，因此总体是收益大于开销的。
