#! https://zhuanlan.zhihu.com/p/261127145
![Image](https://pic4.zhimg.com/80/v2-9c258943ac73f612b17471fea328608b.jpg)

# 【重磅综述】基于图表征的关系推理框架 —— Graph Network

## PR Reasoning Ⅲ：基于图表征的关系推理框架 —— Graph Network

![Image](https://pic4.zhimg.com/80/v2-77b935256c89bf9e113eb8bb64c73b81.png)

本文基于提出 Graph Network 框架的论文

[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
![Image](https://pic4.zhimg.com/80/v2-9f8fb14bb733c76705c794e1e079b254.png)

以及其在**机器人**中的应用论文

[Graph Networks as Learnable Physics Engines for Inference and Control](https://arxiv.org/pdf/1806.01242.pdf)
![Image](https://pic4.zhimg.com/80/v2-865d0b5b8108cfc3fe8da3e05defe578.png)
[Relational inductive bias for physical construction in humans and machines](https://arxiv.org/abs/1806.01203)
![Image](https://pic4.zhimg.com/80/v2-90c8a65a2e6c940bdd06778cae665997.png)


来总结 **Deepmind** 提出的这个**图表征的关系推理框架**。


## Preliminaries
**关系推理 (relational reasoning) 与关系归纳偏置 (relational inductive bias)** 详见

[【重磅综述】Relational Inductive bias 关系归纳偏置及其在深度学习中的应用](https://zhuanlan.zhihu.com/p/261081574 'card')

还可以稍微了解一下 **图神经网络**

[PR StructureⅠ：Graph Neural Network An Introduction](https://zhuanlan.zhihu.com/p/158984343 'card') 

对于结构化建模，也可以了解一下

[PR Structured Ⅱ：Structured Probabilistic Model](https://zhuanlan.zhihu.com/p/161703636 'card')

## Graph Networks
提出了一个 graph networks (GN) framework，该框架定义了一类**基于图结构表征的关系推理的函数**。GN框架概括并扩展了各种图神经网络，MPNN和NLNN方法，并支持从简单的构建模块构建复杂的体系结构。

**注意**：我们避免在 “graph networks” 中使用术语 “neural” 来反映**它可以用除神经网络以外的功能来实现**，尽管这里我们的重点是神经网络的实现。

> 这就仿佛 Bayesian Network 和 Bayesian Neural Network 的区别。所以 Deepmind 这一系列的工作，是旨在推广 Bayesian Network 形式的**结构化计算**到更通用的领域。

还是和 ResNet 一样，对 Graph Network 定义一个基本组件 block：**一个以 graph 为输入，在结构上执行计算，并返回一个 graph 做输出的 graph-to-graph module**。
- entities 表示为图的节点 nodes
- relations 表示为图的边 edges
- system-level properties 表示为图的全局属性

GN框架的块组织强调可定制性，并合成表示期望的归纳偏差的新架构。主要设计原则是：**灵活的表征形式**；
**可配置的块内结构**；
和**可组合的多块体系结构**。

> **引例**：考虑在任意重力场中预测一组橡胶球的运动，这些橡胶球不是独立弹跳，而是具有一个或多个弹簧，这些弹簧将它们与其他（或全部）其他橡胶球连接。

### 定义 graph
GN框架的图稍微复杂点，由三元组 $G=(u,V,E)$ 定义。多出来的 u 是全局属性，以引力为例，代表的就是引力场。边定义为 $E=\left\{\left(\mathbf{e}_{k}, r_{k}, s_{k}\right)\right\}_{k=1: N^{e}}$，其中 $r_k$ 是receiver node的index，$s_k$ 是sender node的index（相当于记录了通信双方信息），可以理解为不同球之间的弹簧及其相应的弹簧常数。
![Image](https://pic4.zhimg.com/80/v2-377ffb768273245a492781061aaf1109.png)


### Internal structure of a GN block GN块的内部结构
GN块包含三个“update”函数 $\phi$ 和三个“aggregation”函数 $\rho$:
$$
\begin{array}{ll}
\mathbf{e}_{k}^{\prime}=\phi^{e}\left(\mathbf{e}_{k}, \mathbf{v}_{r_{k}}, \mathbf{v}_{s_{k}}, \mathbf{u}\right) & \overline{\mathbf{e}}_{i}^{\prime}=\rho^{e \rightarrow v}\left(E_{i}^{\prime}\right) \\
\mathbf{v}_{i}^{\prime}=\phi^{v}\left(\overline{\mathbf{e}}_{i}^{\prime}, \mathbf{v}_{i}, \mathbf{u}\right) & \overline{\mathbf{e}}^{\prime}=\rho^{e \rightarrow u}\left(E^{\prime}\right) \\
\mathbf{u}^{\prime}=\phi^{u}\left(\overline{\mathbf{e}}^{\prime}, \overline{\mathbf{v}}^{\prime}, \mathbf{u}\right) & \overline{\mathbf{v}}^{\prime}=\rho^{v \rightarrow u}\left(V^{\prime}\right)
\end{array}\\
$$

其中 $E_{i}^{\prime}=\left\{\left(\mathbf{e}_{k}^{\prime}, r_{k}, s_{k}\right)\right\}_{r_{k}=i, k=1: N^{e}}, V^{\prime}=\left\{\mathbf{v}_{i}^{\prime}\right\}_{i=1: N^{v}}, \text { and } E^{\prime}=\bigcup_{i} E_{i}^{\prime}=\left\{\left(\mathbf{e}_{k}^{\prime}, r_{k}, s_{k}\right)\right\}_{k=1: N^{e}}$

我们一个一个看，这些变量都是什么：
-  将 $\phi^{e}$ 映射到所有边以计算每条边更新；
-  将 $\phi^{v}$ 映射到所有节点以计算每个节点的更新；
-  $\phi^u$ 是用于全局更新的；
-  $\rho$ 均以集合作为输入，并将其简化为单个元素的汇总信息（意味着这个信息提取/聚合函数需要满足排列不变性）

### Computational steps within a GN block GN块的计算
当将图G作为GN块的输入后，从边开始，到节点再到全局级别进行计算。
图3显示了这些计算中都涉及哪些图元素，图4a显示了具有更新和聚合功能的完整GN块。
算法1显示了以下计算步骤
1. 每条边算一下 $\phi^e$，得到新的边特征。在引例中对应于两个相连球之间的力或势能。
2. $\rho^{e \rightarrow v}$ 聚合节点 $i$ 相连的边的边更新，得到 $\overline{\mathbf{e}}_{i}^{\prime}$，这将被用于接下来的节点更新。引例中，即为某个球受力或势能和。
3. $\phi^v$ 用于每个节点的更新。引例中，即为计算每个球更新后的位置，速度和动能。
4. $\rho^{e \rightarrow u}$ 聚合所有边更新，用于接下来的全局更新。引例中，可以计算总和力（在这种情况下，根据牛顿第三定律应为零）和弹簧的势能。
5. $\rho^{v \rightarrow u}$ 聚合所有的节点，用于接下来的全局更新。引例中，为系统的总动能。
6. $\phi^u$ 用于全局属性的更新。引例中，类似于物理系统的净力和总能量的值。

![Image](https://pic4.zhimg.com/80/v2-0a09b7c81acc891233111ca09bf1f79a.png)

![Image](https://pic4.zhimg.com/80/v2-a64bcfbb9ffd96587b78d3637c83e218.png)

![Image](https://pic4.zhimg.com/80/v2-0cd8dc86a9349f294e99206740d974ad.png)

### Relational inductive biases in graph networks GN 中的关系归纳偏置

1. 图可以表达实体之间的任意关系，这意味着GN的输入确定表示如何交互和隔离，而**不是由固定架构决定的**。
2. 图将实体及其关系表示为集合，具有**排列不变性**。
3. **GN的每条边和每个节点函数分别在所有边缘和节点之间重用**。
这意味着GN会自动支持一种**组合泛化**形式：由于图是由边，节点和全局特征组成的，因此单个GN可以在不同大小（边缘和节点的数量）和形状（边缘连通性）的图形上运行。

### Design principles for graph network architectures 图网络架构的设计原理
#### Flexible representations 灵活表征

GN的灵活表征体现在两方面：
1. 属性表征
2. 图结构本身

#### Configurable within-block structure 可配置的块内结构
![Image](https://pic4.zhimg.com/80/v2-b2ef7abe74a68d72adeffef1ea2e24a9.png)

#### Composable multi-block architectures 可组合的多块架构

![Image](https://pic4.zhimg.com/80/v2-57887c3a2bd35eb7aca3023ec4883987.png)
![Image](https://pic4.zhimg.com/80/v2-bf4201de01e9926f3b0633828cff66f8.png)

## Application
![Image](https://pic4.zhimg.com/80/v2-501089516efd1e645c3c54d623cf3471.png)

### 堆木块
引文 3 认为，一些复杂的任务 DL 之所以不太行，就是因为他们缺乏 relational inductive bias，**缺乏推理对象间关系并选择场景的结构化描述的能力**。所以他们设计了一个堆木块成塔的实验来验证**结构化表征、关系归纳偏置**对提高智能性的作用。
![Image](https://pic4.zhimg.com/80/v2-88aa0b75896433029c78025a8828182f.png)
![Image](https://pic4.zhimg.com/80/v2-b89182699adce753b5ee56bb1e21e224.png)
- 节点：位置 x, 方向 q，由encoder得到 $\mathbf{n}_{i}=\operatorname{enc}_{n}\left(x_{i}, q_{i} ; \theta_{\mathrm{enc}_{n}}\right)$；
- 边：glue $u$，边encoder得到 $\mathbf{e}_{i j}=\operatorname{enc}_{e}\left(u_{i j} ; \theta_{\mathrm{enc}_{e}}\right)$；
- global features：整张图的方程（例如，整张图的稳定性），初始设置为0。

然后进行标准的GN计算，节点计算确定节点之间是否连接、边确定一个木块上的受力、global确定整体的稳定性。因此，GN能够**确定哪些块之间是局部不稳定的**（例如图中最顶部的块），需要胶水。
但是，它没有足够的信息来确定图1中最底部的块也需要胶水，因为它已完全支撑了其上方的块。循环消息传递 Recurrent message-passing 允许将有关其他块的信息传播到最底层的块，从而可以推断出非局部关系。

给定更新的边，节点和全局表征，我们可以将它们解码为特定的**边预测**，例如Q值或未归一化的对数概率。
- 监督学习时，边缘时可能性 $p_{i j} \propto \operatorname{dec}_{e}\left(\mathbf{e}_{i j}^{\prime} ; \theta_{\mathrm{dec}_{e}}\right)$；
- 序列决策时，我们为图中的每个边解码一个动作 $\left(\pi_{i j}=\operatorname{dec}_{e}\left(\mathbf{e}_{i j}^{\prime} ; \theta_{\mathrm{dec}_{e}}\right)\right)$ 加上“stop”操作以结束粘合阶段 $\left(\pi_{\sigma}=\operatorname{dec}_{g}\left(\mathbf{g}^{\prime} ; \theta_{\mathrm{dec}_{g}}\right)\right)$。

### Mujoco classical task
引文 2 这个任务更加展示了 GN 模型结构化表征的强大能力。

插一句，超喜欢这种文章的**情怀**：
> Our framework offers new opportunities for harnessing and exploiting rich knowledge about the world, and takes a key step toward building machines with more human-like representations of the world.



![Image](https://pic4.zhimg.com/80/v2-6e60105da3593d8d3041f67e68d70ca3.png)

![Image](https://pic4.zhimg.com/80/v2-579fd239be89f391eef55177c1f1f669.png)

显然，躯干和关节代表节点和边。

本文区分物理场景中的静态和动态属性，它们在独立的图中表示。
- 静态图 $G_s$ 包含有关系统参数的静态信息，包括全局参数（例如时间步长，粘度，重力等），每个物体/节点的参数（例如质量，惯性张量等）。 
，以及每个关节/边缘参数（例如关节类型和属性，电机类型和属性等）。
- 动态图 $G_d$ 包含有关系统瞬时状态的信息。其中包括每个物体/节点的3D直角坐标位置，4D四元数方向，3D线速度和3D角速度。此外，它还包含应用于相应边缘中不同关节的作用的大小。

**本文是将 GN 模型作为 model-based RL 学习系统model的方法，因此需要有预测和推断的能力。**


**Forward models**

为了进行预测，本文引入了基于GN的 Forward 模型，用于学习从当前状态预测未来状态。它以一个时间步长运行，并包含两个以“deep”排列顺序组成的GN（未共享参数；见图2c）。
1. 第一个GN取一个输入图G，并产生一个隐图$G'$；
2. $G'$与 $G$ 串联（例如，图跳过连接），并作为输入提供给第二个GN，后者返回输出图 $G^*$；
3. 正向模型训练优化GN，从而使 $G^*$ 的${n_i}$特征反映跨时间步长的每个身体状态的预测；

本文使用两个GN的原因是允许所有节点和边缘通过 $g'$ 相互通信。初步测试表明，与单个IN / GN相比，此方法具有较大的性能优势。

当然，forward model 还可以用基于GN的循环前向模型，该模型包含三个RNN子模块，分别应用于所有边缘，节点和全局特征，然后再组成GN块（见图2d）。

**Inference models**
本文实际上就是一种 **“隐式”的系统辨识**，并没有显式估计未观察到的属性的推断，而是以可用于其他机制的**潜在表征来表达**。

本文引入了一个基于GN的递归推断模型，该模型仅观察轨迹的动态状态，并建立未观察到的静态属性的潜在表征（即执行隐式系统辨识）。
1. 在某些控制输入下，它将一系列动态状态图 $G_d$ 作为输入，并在T个时间步长之后返回输出 $G^*(T)$。
2. 通过将 $G^*(T)$ 与输入动态图 $G_d$ 进行图级联，将其传递到一步forward 模型。
3. 循环核将输入 $G_d$ 和隐图 $G_h$ 进行图级联并传递到GN块（见图2e）。 
4. GN块返回的图为graph-split，以形成输出$G^*$ 和最新的隐图$G^*_h$。
   
可以对整个体系结构进行共同训练，学会根据系统的观察特征如何推断系统的未观察特性，并使用它们进行更准确的预测。

有了model-based 的 model，剩下的控制部分就是 RL 和 MPC 的问题了，也就不是我们的关注重点了。

## Conclusion
Graph Network 模型成功地通过**层级更新与聚合**的方式，将**任务的特定结构化先验与深度学习完美地结合在一起**，完善了深度学习在结构化推理问题上的弊端。

最惊艳的莫过于其对于复杂物理系统的完美抽象，**使agent能够在关系归纳偏置的基础上，缩小参数空间，更高效可靠地且可解释地学习到系统的参数**。这种理念必将是 AI 迈向更智能的推理与表征不可或缺的一步。