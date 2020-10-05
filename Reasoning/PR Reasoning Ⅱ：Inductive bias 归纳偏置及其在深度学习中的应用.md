#! https://zhuanlan.zhihu.com/p/261081574
![Image](https://pic4.zhimg.com/80/v2-223c740cc2cd005009ccfc402437f7a2.jpg)
# 【重磅综述】Relational Inductive bias 关系归纳偏置及其在深度/强化学习中的应用

## PR Reasoning Ⅱ：Relational Inductive bias 关系归纳偏置及其在深度学习中的应用

![Image](https://pic4.zhimg.com/80/v2-3f9a753ea8fa0cbf3a556b046017c14b.png)

本文总结自：
1. [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
    ![Image](https://pic4.zhimg.com/80/v2-9f8fb14bb733c76705c794e1e079b254.png)

2. [Deep Reinforcement Learning With Relational Inductive Bias](https://openreview.net/forum?id=HkxaFoC9KQ&utm_campaign=piqcy&utm_medium=email&utm_source=Revue%20newsletter)
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

人类强大的组合泛化能力在很大程度上取决于我们**用于表示关系的表征结构和推理的认知机制**。
- 将复杂系统表示为**多个实体 (entity) 及其之间相互作用的组成**；
- 使用**层次结构 (hierarchies)** 来抽象化细微的差异，并捕获表征和行为之间更普遍的共性。例如物体的一部分，场景中的物体，城镇的邻里
以及一个国家/地区的城镇；
- **通过组合熟知的技能和常识来解决新问题**，例如“乘飞机旅行” ->“去圣地亚哥”，“吃饭” -> “印度餐厅”；
- 通过对齐两个领域之间的关系结构并根据对另一个领域的相应知识得出推论来进行类比。

**世界是组成性的，或者至少我们是用组成性术语理解的。Learning 的时候，我们要么将新知识融入现有的结构化表示中，要么调整结构本身以更好地适应（和利用）新旧知识。**

**过去**数据和计算资源昂贵时候，结构化方法强大的归纳偏置具有改善样本复杂性的能力。combinatorial generalization 在许多结构化方法中至关重要，包括logic, grammars, classic planning, graphical models, causal reasoning, Bayesian nonparametrics, and probabilistic programming。整个子领域都专注于以实体和关系为中心的显式学习，例如**关系强化学习 (稍后展开) **和**统计关系学习**。

**现代**深度学习方法经常遵循**端到端**的设计理念，强调**最小限度的先验表示和计算假设**，并**力求避免明确的结构和“手工设计”**。这种强调与当前的大量廉价数据和廉价计算资源很好地吻合。

然而，DL 在**复杂的语言和场景理解**、**结构化数据的推理**、**将学习迁移到训练条件之外**以及**从少量经验中学习**方面面临的主要挑战。这些挑战需要组合泛化，因此我们需要**深度学习 + 结构化**。与经典方法不同的是，如何学习实体和关系的表示形式，结构以及相应的计算，从而减轻了需要预先指定它们的负担。至关重要的是，这些方法**以特定的体系结构假设的形式带有强烈的关系归纳偏置**，这些偏置**指导**这些方法学习关于实体和关系的方法。

### Inductive biases 归纳偏置
Learning 是通过观察世界并与之互动来吸收有用知识的过程。它涉及搜索解决方案空间，以寻找可以更好地解释数据或获得更高回报的解决方案。归纳偏置允许学习算法将一个解决方案（或解释）优先于另一个解决方案（或解释），而与观察数据无关。

归纳偏置也不是什么新知识了，它出现在机器学习的很多地方：

- 在贝叶斯模型中，**归纳偏置通常通过先验分布的选择和参数化来表示**；
- 在其他情况下，归纳偏置可能是**为避免过度拟合而添加的正则化项**；
- 最大条件独立性、最小交叉验证误差、最大边界、最少特征数、最近邻居等等。

因此，归纳偏置是学习器去预测其未遇到过的输入的结果时，所作的**目标函数的必要假设集合**，是关于“什么样的模型更好”，也是**任务导向的**。

再举个例子，L2正则化使模型对参数值较小的解决方案进行优先排序，并且可以针对其他问题而引入独特的解决方案和全局结构。
这可以解释为关于学习过程的一种假设：当解决方案之间的歧义较少时，寻找好的解决方案就容易了。
注意，这些假设不一定是明确的，它们反映了模型或算法如何与世界交互的解释。 

### Relational Inductive Bias 关系归纳偏置
**我们使用 relational inductive bias 来泛指学习过程中对实体之间的关系和交互施加了约束的归纳偏置，并将其用于 Relational Reasoning。**


> 我们将 **structure** 定义为组合了一组已知构件 (building blocks) 的产物。 \
**Structured representations (结构化表征)** 捕获了这种构成（即元素的排列）。 \
**Structured computations (结构化计算)** 对元素及其整体进行了运算。\
**关系推理 (Relational Reasoning)** 涉及使用实体和关系的构成规则来利用实体和关系的结构化表征。

在刚刚的论述中，已经提到了一些关系归纳偏置的名词，现在我们总结一下并将其定义为**关系归纳偏置的三要素**：

- **entity** 是具有属性的元素，例如具有大小和质量的物理对象。（理解成**图的节点**）
- **relation** 是实体之间的属性。（理解成**图的边**）
两个物体之间的关系可能包括大小相同，重量关系以及距离。
关系也可以具有属性。（图的边也可以有属性）
- **rule** 是将实体和关系映射到其他实体和关系的函数（非二元逻辑谓词），例如一些比较：X更大一些嘛？实体X重于实体Y嘛?。
在这里，我们考虑采用一个或两个参数（一元和二进制）并返回一元属性值的规则。

在经典的深度学习三大模型中，我们就可以找到这样的 relational inductive biases，我们要分别找到它们中 entities, relations, and rules 分别是什么？


![经典神经网络模型本身也包含 relational inductive bias](https://pic4.zhimg.com/80/v2-bcb009f0c1331c69bc3b780e45f63464.png)



![Image](https://pic4.zhimg.com/80/v2-978aa2cf78f82b9bfeb97d6c3893bb68.png)

- **Fully connected layers**: **实体**是网络中的单位，**关系**是全连接的，**规则**由权重和偏置指定。该规则的参数是完整的输入信号，没有复用，也没有信息隔离（图1a）。因此，在全连接的层中的**隐式关系归纳偏置非常弱**：所有输入单元可以相互独立地确定输出单元的值。

- **Convolutional layers**: **实体**仍然是单个单位（或网格元素，例如像素），但关系较稀疏。
全连接层和卷积层之间的差异强加了一些重要的关系归纳偏置：**局部性和平移不变性**。**局部性 (Locality)** 反映了关系规则的参数是那些在输入信号的坐标空间中彼此靠近且与远端实体隔离的实体。**平移不变性 (translational invariance)**反映了输入中各部分重复使用同一规则。这也解释了为什么 CNN 适合处理自然图像，因为局部邻域内存在**较高的协方差**，随着距离的减小，协方差会减小，并且统计信息在整个图像上大致是固定的。
- **Recurrent Layers**: 可以将每个处理步骤的输入和隐状态视为**实体**，并将一个步骤的隐状态对先前隐状态和当前输入的马尔可夫依赖关系视为**关系**。合并实体的**规则**以步骤的输入和隐藏状态作为参数来更新隐藏状态。RNN 适合处理时序信息就得益于，该规则在每个步骤（图1c）中都得到了复用，它反映了关系归纳偏差的 **时间不变性 (temporal invariance)**。

### 集合 (Set) 和图 (Graph) 的结构化表征与计算
虽然标准的深度学习模型包含各种形式的关系归纳偏置，但是并不代表深度学习可在任意关系结构上运行。
因此，**我们需要找到一种关系的结构化表征形式**。

**集合**是系统的自然表征形式，由顺序未定义或不相关的实体描述。它带来了**置换不变性 (permutation invariance)** 的特点,由此产生了 Deep Sets 这类模型（不是重点，感兴趣大家自行google）。

然而很多实际问题会出现，有些实体之间具有关系，而另一些实体对则缺乏关系。这个时候就需要图来表征。

**图**是一种支持任意（成对）关系结构的表征形式，并且对图的计算提供了强大的关系归纳偏置，是卷积层和递归层所不能提供的。

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
看完了FNN、CNN、RNN，该到我们关注的 RL 领域了。本节主要介绍引文 2 的内容，这篇文章尝试在《星际争霸II》中利用迭代的消息传递过程来发现和推理场景中的相关实体和关系，并取得了超越人类大师的成绩。这项工作的主要贡献是介绍了**通过关系归纳偏差来表征和推理无模型的DRL agent 状态的技术**。实验表明，此方法可以具有高效，泛化性和可解释性方面的优势。 

### RL ALGORITHM
![Image](https://pic4.zhimg.com/80/v2-02ed1cb4d8744a1abcbf71ad98d840e7.png)

本文的RL是基于A2C的，根据 embedded 的状态表征，得到策略 $\pi$ 以及 critic 的基准值 $B$。

### RELATIONAL MODULE
有趣的部分来了。

为了执行关系推理的一个步骤，本文架构计算了每个实体与所有其他实体（包括自身）之间的成对交互，定义为 $p_{i,j}$，并通过累积所有关于这个交互的信息来更新每个实体 $\tilde{\mathbf{e}}_{i} \leftarrow\left\{\mathbf{p}_{i, j}\right\}_{j=1: N}$。

简单来讲，本文就是用**self-attention**来表示关系归纳偏置，本文的自注意力是基于多头点积注意力（MHDPA）实现的。MHDPA（图2的扩展框）将实体E分别投影到query, key, and value vectors Q，K和V的矩阵中。Query $q_i$ 和所有key  $k_{j=1:N}$ 之间的相似性由一个点积计算，然后通过softmax函数将其归一化为注意力权重wi，然后用于计算两两交互项 $\mathbf{p}_{i, j}=w_{i, j} \mathbf{v}_{j}$。写作矩阵，即为 $A=\operatorname{softmax}\left(d^{-\frac{1}{2}} Q K^{T}\right) V$，其中 $d$ 
简单来讲，本文就是用**self-attention**来表示关系归纳偏置，本文的自注意力是基于多头点积注意力（MHDPA）实现的。MHDPA（图2的扩展框）将实体E分别投影到query, key, and value vectors Q，K和V的矩阵中。Query $q_i$ 和所有key  $k_{j=1:N}$ 之间的相似性由一个点积计算，然后通过softmax函数将其归一化为注意力权重wi，然后用于计算两两交互项 $\mathbf{p}_{i, j}=w_{i, j} \mathbf{v}_{j}$。写作矩阵，即为 $A=\operatorname{softmax}\left(d^{-\frac{1}{2}} Q K^{T}\right) V$，其中 $d$ query and key vectors 的维度。

很明显，multi-head 被用于假设不同的关系。根据多头累积的交互来计算更新的实体，$\tilde{\mathbf{e}}_{i}=g_{\theta}\left(\mathbf{a}_{i}^{h=1: H}\right)$。

以上只是一步关系更新过程，称之为 'block' （就像 ResNet 那样），它可以使用共享（recurrent）或非共享（deep）参数进行迭代地应用，以计算实体之间的高级交互，类似于在图上传递消息。多个这样的关系 block 的堆叠就构成了 figure 2 的 relational module。

我们就用 BOX-world 展示一下这个推理过程：

![Image](https://pic4.zhimg.com/80/v2-774dfb2a3c8df9dc6eb7e02e4ea091a4.png)

> 钥匙由单个彩色像素表示。agent可以通过在上下左右移动来捡起一个钥匙（即一个不与任何其他彩色像素相邻的钥匙）。盒子由两个相邻的彩色像素表示，右边的像素代表盒子的锁，颜色代表可以用来打开该锁的钥匙；左边的像素表示盒子的内容，盒子被锁定时无法访问。\
> 要收集盒子中的内容，agent必须首先收集打开盒子的钥匙（与锁的颜色匹配的钥匙），然后越过锁，这使锁消失。此时，该框的内容将变得可访问，并且可以由agent拾取。大多数盒子都包含钥匙，如果可以访问，可以用来打开其他盒子。其中一个盒子包含一颗宝石，由一个白色像素表示。agent的目标是通过解锁包含宝石的盒子并在宝石上走来捡起宝石来获得宝石。
agent拥有的密钥在输入观察值中显示为左上角的像素。钥匙只能用一次。

好了，故事设定很长。总之，我们需要 agent 按特定顺序打开盒子。钥匙和盒子出现在房间的随机位置，要求具有根据其抽象关系而不是根据其空间接近性来推理钥匙和盒子的能力。

![Image](https://pic4.zhimg.com/80/v2-0f4cd928e2d0499c67f444dd2f30828e.png)

在每个head上，注意力权重矩阵的一行代表一个物体与其他的关系。图4左上角显示了当实体是解决方案路径上的对象时的分析结果。
对于其中一个关注头来说，每个钥匙都倾向于选择能够用该钥匙解锁的锁。
换句话说，注意力权重反映了一旦收集到钥匙，agent可以使用的选项。

**下一章会通过三篇文章介绍 Deepmind 提出的用于关系归纳偏置的 Graph Network**
