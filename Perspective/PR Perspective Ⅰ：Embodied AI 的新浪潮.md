#! https://zhuanlan.zhihu.com/p/260562672
![Image](https://pic4.zhimg.com/80/v2-ca5a91e5f7a1b171d678c70a901017da.jpg)
# PR Perspective Ⅰ：Embodied AI 的新浪潮 —— new generation of AI 

经常看我专栏的朋友们可能会知道，我标题里的 `PR` 代表的是 `Probabilistic Robotics` 系列，也就是与**机器人强化学习**结合的**概率机器人**知识。PR 系列一直是这个专栏的主打内容，目前有 ***Sampling, Structured, Reasoning, Memory*** 分章，所有的一切都是为让深度学习或强化学习知识能够更好地应用到**实体 (real-world) 机器人**中，实现机器人的智能控制与决策推理。

本次的 ***Perspective*** 分章旨在展示 **AI + Robotics** 领域的前沿理念，通过一些大佬的观点来坚定我自己的道路。



本文援引

1.  *Oxford Robotics Institute* 两位 *prof. Nick Hawes & Lars Kunze* 的一篇 Special Issue：[Special Issue on Reintegrating Artificial Intelligence and Robotics](https://link.springer.com/article/10.1007/s13218-019-00625-x) - 2019

2. *Prof. Ingmar Posner from Oxford Robotics Institute* ：[Robots Thinking Fast and Slow: On Dual Process Theory and Metacognition in Embodied AI](https://openreview.net/pdf?id=iFQJmvUect9) - 2020
3. *Prof. Matej Hoffmann and Rolf Pfeifer from University of Zurich*：[The Implications of Embodiment for Behavior and Cognition: Animal and Robotic Case Studies ](https://arxiv.org/ftp/arxiv/papers/1202/1202.0440.pdf) - 2012
4.  [Embodied AI as Science: Models of Embodied Cognition, Embodied Models of Cognition, or Both?](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.8578&rep=rep1&type=pdf) - 2004
5. [The body of knowledge: On the role of the living body in grounding embodied cognition](https://www.sciencedirect.com/science/article/pii/S030326471630168X) - 2016
6. [Embodied Artificial Intelligence: Trends and Challenges](https://link.springer.com/chapter/10.1007/978-3-540-27833-7_1) - 2003
7. [AllenAct: A Framework for Embodied AI Research](https://arxiv.org/abs/2008.12760) - 2020


## 什么是 Embodied AI？

Special Issue "[Frontiers of Embodied Artificial Intelligence: The (r-)evolution of the embodied approach in AI](https://www.mdpi.com/journal/philosophies/special_issues/Artificial_Intelligence)" 中解释到：

> The driving idea of “Embodied AI” is that, in order to successfully explore natural cognitive processes, AI practitioners **have to build and study “complete” or “embodied agents”**: physically realized machines whose structure and functioning are based on biologically informed theses on adaptation and cognition. In other words, not programs, nor virtual agents, but *biological-like* *robots*: “embodied” and “situated” artificial systems that, like biological systems, provide themselves information about their environment by interacting with it, and, in this sense, learn about their environment through their interactive bodies – **something that programs, or computers, cannot do.**

在 [Habitat](https://aihabitat.org/) 模拟器中：

> The study of intelligent systems with a physical or virtual embodiment (robots and egocentric personal assistants). The [embodiment hypothesis](https://dl.acm.org/doi/10.1162/1064546053278973) is the idea that “**intelligence emerges in the interaction of an agent with an environment and as a result of sensorimotor activity**”.

即为成功地探索自然的认知过程，人工智能从业者必须构建和研究“完整的”或“具身的智能体”：物理实现的机器，其结构和功能基于有关适应性和认知的生物学知识。使其能够像生物一样，通过与环境交互来获取有关环境的信息，这是程序或计算机无法做到的。而在“古典”（也称“good old-fashioned AI”，GOFAI）人工智能和认知科学中，重点是通过对世界内部符号表征的计算来解决问题。 

然而，自1990年代提出这个概念以来，embodied AI community 的研究者们始终没有对这个词的具体定义达成共识（引文5）。

![Image](https://pic4.zhimg.com/80/v2-7340c3d5e838a2b692533e7cd775243b.png)

尽管很多文章在讨论 embodied AI 更加深刻的含义（仿生/发育/进化机器人学，普适计算与界面技术，多智能体，人工生命等），本文还是沿用它最简单的定义：**embodied AI = AI + Robotics**


借用欧盟人工智能高级别专家组（AI HLEG）近日发布《人工智能定义：主要能力和科学学科》的图，说明AI与Robotics的关系：

![它的意思大概是Robotics除了传统机器人学的知识外，还需要Machine Learning 大类与 Reasoning 大类的知识](https://pic4.zhimg.com/80/v2-4ce475b09e90f927c7bb471b5cf74119.png)

## Why Embodied AI
为什么学界（也可能只是 embodied AI community）会认为 Embodied AI 与物理世界的**交互**才更能得到智能性？为什么学界认为 embodied AI 是区别于古典主义AI的下一代AI (引文4)？

引文 3 通过对 locomotion, grasping, and visual perception 等机器人任务的分析来展示广义的 embodiment，并总结出如下的 embodied AI 框架。

![Image](https://pic4.zhimg.com/80/v2-912b76a363e1a9bd4874ea874af368a0.png)

从这些案例研究中可以得出结论：**如果我们真的想获得类似于大脑的智力，则大脑（或控制器）必须具有通过形态变化从神经上或机械上迅速切换到各种开发方案的能力。**
> It is the body and the interaction with the environment that are the natural candidates for first primitive representations. We want to point out that cognition is in the service of behavior here. That is, these first representations or models have to bring behavioral advantage. 

## Embodied AI 进展
### A. Reintegrating Artificial Intelligence and Robotics
Nick Hawes 和 Lars Kunze 在引文 1 指出，目前 AI 领域的研究人员都聚焦在现实世界中的抽象问题上，通常这些用于古典AI的简化假设在物理世界和 embodied robot systems 中是不成立的。
AI + 机器人系统的方法面临着各种困难和挑战，包括**弥合感知符号(percept-symbol)鸿沟**，**与人类互动和向人类学习**，integrating task and motion planning以及组合系统以构建完整的机器人解决方案。

### B. Embodied AI community 概览
2020年的引文7通过提出一个符合 Embodied AI 核心要求、模块化且灵活的学习框架 AllenAct，来整合 Embodied AI community 任务环境之间割裂的情况。

> A huge amount of effort is required to do something as simple as taking a model trained in one environment and testing it in another.

它还列举了近年来该领域论文的文章数量与发展趋势，并标记了一些具有影响力的文章。
![出自引文7](https://pic4.zhimg.com/80/v2-4f7fdbcf8ad5f3babf76965240a7b254.png)

### C. Dual Process Theory and metacognition

我们设想机器人能够实时稳定地运行，能够从有限的数据中学习，做出对任务甚至对安全至关重要的决定，甚至越来越显示出解决创造性问题的诀窍。Ingmar Posner 在引文 2 中提出了结合认知科学的新进展：
**Dual Process Theory and metacognition**。

*Dual Process Theory* 概念出自 Daniel Kahneman 的书 ***Thinking Fast and Slow***，他在书中提出**人类的思想是由于两个相互作用的过程而产生**：
- **无意识的、非自愿的、直觉的响应**
- **更加费力的、深思熟虑的推理**

此外，**metacognition（元认知）是指我们评估自己的思维能力的能力**，也起着核心作用。我们人类之所以能够在非平稳环境下做出成功的决策，大部分通常归因于我们的元认知能力：**决策过程，知道我们是否有足够信息来做出决策的能力以及一旦做出决策就可以分析决策结果的能力**。

通过元认知的能力，可以将Dual Process Theory的两个系统衔接起来：*Do not trust your intuition, think about it.*

作者认为，通过这个理论探索专门用于机器人学习的结构时，可能具有比以往更大的优势。




