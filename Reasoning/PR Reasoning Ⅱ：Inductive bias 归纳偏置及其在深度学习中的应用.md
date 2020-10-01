![Image](https://pic4.zhimg.com/80/v2-223c740cc2cd005009ccfc402437f7a2.jpg)
# 【重磅综述】Relational Inductive bias 关系归纳偏置及其在深度/强化学习中的应用

## PR Reasoning Ⅱ：Relational Inductive bias 关系归纳偏置及其在深度学习中的应用

![Image](https://pic4.zhimg.com/80/v2-3f9a753ea8fa0cbf3a556b046017c14b.png)

本文总结自：
1. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
    ![Image](https://pic4.zhimg.com/80/v2-9f8fb14bb733c76705c794e1e079b254.png)
2. [Relational inductive bias for physical construction in humans and machines]()
   ![Image](https://pic4.zhimg.com/80/v2-e963fd7a0cb440de4194a7be09a06b84.png) 
3. [Deep Reinforcement Learning With Relational Inductive Bias]()
   ![Image](https://pic4.zhimg.com/80/v2-2a3a953db6f972e5fa541c24828c354c.png)


## 以 Graph 的形式学习 Relational Inductive bias
从引文1开始，本文讨论了两件事：

1. **combinatorial generalization (组合泛化)** must be a top priority for AI to achieve human-like abilities\
   这可以通过 **biasing learning** 的方式学习。
2. **structured representations and computations (结构化表征和结构化计算)** are key to realizing this objective\
文中指出，**必须摒在”手动设计结构(hand-engineering)“和”端到端（end-to-end）”中二选一的错误做法，应以深度学习 + graph 图的形式学习关系归纳偏置**。

> 问题来了，什么是组合泛化？什么是关系归纳偏置？

### Combinatorial generalization 组合泛化
**combinatorial generalization：从已知的构建基块构造新的推论，预测和行为**

> 一个简单的例子，人可以利用26个字母，组成千奇百怪的句子。\
> 即，**infinite use of finite means，有限的知识创造无限的可能。**

人的组合泛化能力在很大程度上取决于我们**用于表示关系的结构和推理的认知机制**。
- 将复杂系统表示为**多个实体 (entity) 及其之间相互作用的组成**；
- 使用层次结构来抽象化细微的差异，并捕获表示和行为之间的更普遍的共性，例如对象的一部分，场景中的对象，城镇的邻里
以及一个国家/地区的城镇；
- 通过编写熟悉的技能和例程来解决新问题，例如“乘飞机旅行”，“去圣地亚哥”，“吃饭”和“印度餐厅”；
- 通过对齐两个领域之间的关系结构并根据对另一个领域的相应知识得出推论来进行类比。

**世界是组成性的，或者至少我们是用组成性术语理解的。Learning 的时候，我们要么将新知识融入现有的结构化表示中，要么调整结构本身以更好地适应（和利用）新旧知识。**

**过去**数据和计算资源昂贵时候，结构化方法强大的归纳偏置具有改善样本复杂性的能力。combinatorial generalization 在许多结构化方法中至关重要，包括逻辑，语法，经典计划，图形模型，因果推理，贝叶斯非参数和概率编程。整个子领域都专注于以实体和关系为中心的显式学习，例如**关系强化学习**和**统计关系学习**。

**现代**深度学习方法经常遵循**端到端**的设计理念，强调**最小限度的先验表示和计算假设**，并**力求避免明确的结构和“手工设计”**。这种强调与当前的大量廉价数据和廉价计算资源很好地吻合。

然而，DL 在**复杂的语言和场景理解**、**结构化数据的推理**、**将学习迁移到训练条件之外**以及**从少量经验中学习**方面面临的主要挑战。这些挑战需要组合泛化，因此我们需要**深度学习 + 结构化**。与经典方法不同的是，如何学习实体和关系的表示形式，结构以及相应的计算，从而减轻了需要预先指定它们的负担。至关重要的是，这些方法**以特定的体系结构假设的形式带有强烈的关系归纳偏置**，这些偏置**指导**这些方法学习关于实体和关系的方法。

### Inductive biases 归纳偏置
Learning 是通过观察世界并与之互动来吸收有用知识的过程。它涉及搜索解决方案空间，以寻找可以更好地解释数据或获得更高回报的解决方案。归纳偏置允许学习算法将一个解决方案（或解释）优先于另一个解决方案（或解释），而与观察数据无关。
- 在贝叶斯模型中，**归纳偏置通常通过先验分布的选择和参数化来表示**。
- 在其他情况下，归纳偏置可能是**为避免过度拟合而添加的正则化项**（McClelland，1994）
- 或者**可能在算法本身的体系结构中进行了编码**。

归纳偏置通常以灵活性来换取改进的样品复杂性，并且可以从偏置-方差折衷的角度来理解。
理想情况下，归纳偏置既可以在不显着降低性能的情况下改善对解决方案的搜索，又可以帮助找到以理想方式推广的解决方案。
但是，不匹配的归纳偏置也会因引入过强的约束而导致性能欠佳。
归纳偏置可以表达关于数据生成过程或解决方案空间的假设。
例如，当将一维函数拟合到数据时，线性最小二乘遵循近似函数为线性模型的约束，并且在二次惩罚下近似误差应最小。
这反映了这样一种假设，即数据生成过程可以简单地解释为由于加性高斯噪声而损坏的线路过程。
类似地，L2正则化对参数值较小的解决方案进行优先级排序，并且可以针对否则会引起不适的问题引入唯一的解决方案和全局结构。
这可以解释为关于学习过程的一种假设：当解决方案之间的歧义较少时，寻找好的解决方案就容易了。
请注意，这些假设不一定是明确的，它们反映了模型或算法如何与世界交互的解释。 

### Relational Inductive Bias 关系归纳偏置
**我们使用 relational inductive bias 来泛指学习过程中对实体之间的关系和交互施加了约束的归纳偏置，并将其用于 Relational Reasoning。**

> 我们将 structure 定义为组成一组已知构件的乘积。 
“**Structured representations (结构化表征)**”捕获了这种构成（即元素的排列），“**Structured computations (结构化计算)**”对元素及其整体进行了运算。**关系推理 (Relational Reasoning)** 涉及使用 entities （实体）和relations,（关系）的构成rules（规则）来操纵实体和关系的结构化表示。
- **entity** 是具有属性的元素，例如具有大小和质量的物理对象。 
- **relation** 是实体之间的属性。
两个对象之间的关系可能包括相同的大小，比其重的距离和与之的距离。
关系也可以具有属性。
该关系比属性X的X倍多，该属性确定关系为true与false的相对权重阈值。
关系也可能对全球环境敏感。
对于石头和羽毛，该关系以比取决于环境是否在空气中还是在真空中更大的加速度下降。
在这里，我们关注实体之间的成对关系。 
- **rule** 是将实体和关系映射到其他实体和关系的函数（例如非二进制逻辑谓词），例如像X一样大的比例比较？
并且实体X重于实体Y?。
在这里，我们考虑采用一个或两个参数（一元和二进制）并返回一元属性值的规则。

要研究 relational inductive biases，就要定义 entities, relations, and rules 分别是什么？


![经典神经网络模型本身也包含 relational inductive bias](https://pic4.zhimg.com/80/v2-bcb009f0c1331c69bc3b780e45f63464.png)

- 规则函数的自变量（例如，提供哪些实体和关系作为输入）。 
- 如何在计算图上（例如在不同实体和关系之间，在不同时间或处理步骤等之间）重用或共享规则功能。 
- 体系结构如何定义表示形式之间的交互与隔离（例如，通过应用规则得出有关相关实体的结论，而不是分别处理它们）

![Image](https://pic4.zhimg.com/80/v2-978aa2cf78f82b9bfeb97d6c3893bb68.png)

- **Fully connected layers**: 实体是网络中的单位，关系是全部的（第i层中的所有单元都连接到第j层中的所有单元），并且规则由权重和偏置指定。该规则的论据是完整的输入信号，没有重用，也没有信息隔离（图1a）。
因此，在完全连接的层中的隐式关系电感偏置非常弱：所有输入单元可以交互作用来确定任何输出单元的值，独立于输出之间。 
- **Convolutional layers**: 此处的实体仍然是单个单位（或网格元素，例如像素），但关系较稀疏。
全连接层和卷积层之间的差异强加了一些重要的关系归纳偏置：**局部性和平移不变性**
- **Recurrent Layers**: 我们可以将每个处理步骤的输入和隐藏状态视为实体，并将一个步骤的隐藏状态对先前隐藏状态和当前输入的马尔可夫依赖关系视为关系。合并实体的规则将步骤的输入和隐藏状态作为参数来更新隐藏状态。

### 集合 (Set) 和图 (Graph) 的结构化表征与计算
虽然标准的深度学习工具包包含具有各种形式的关系归纳偏置的方法，但是没有“默认”深度学习组件可在任意关系结构上运行。
我们需要具有实体和关系的显式表示的模型，以及需要学习用于计算相互作用的规则以及将其基于数据的方法的学习算法。
重要的是，世界上的实体（例如对象和代理）没有自然顺序。
而是可以通过关系的属性来定义顺序。
例如，一组对象的大小之间的关系可以潜在地用于对它们进行排序，以及它们的质量，年龄，毒性和价格。
顺序的不变性（关系除外）是一种理想的属性，应该由关系推理的深度学习组件来反映。

**集合**是系统的自然表示形式，由顺序未定义或不相关的实体描述

**图**是一种支持任意（成对）关系结构的表示形式，并且对图的计算提供了强大的关系归纳偏置，而卷积层和递归层所不能提供的。

![Image](https://pic4.zhimg.com/80/v2-3d2aa21a887c6ed1d1b2f0df49700702.png)



## Application
图神经网络家族的模型被认为在具有**丰富关系结构的任务**上很有效
- visual scene understanding tasks
- few-shot learning
- learn the dynamics of physical systems
- multi-agent systems
- reason about knowledge graphs
- predict the chemical properties of molecules
- predict traffic on roads
- classify and segment images and videos
- 3D meshes and point clouds
- classify regions in images
- perform semi-supervised text classification
- machine translation
- model-based continuous control
- model-free reinforcement learning
- classical approaches to planning

## Deep Reinforcement Learning With Relational Inductive Bias

TODO
