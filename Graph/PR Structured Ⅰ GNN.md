# PR StructureⅠ：Graph Neural Network An Introduction 

[TOC]

最近，我逐渐将研究重心从MDP转向了结构化建模。无论是我的专栏写作进度，还是科研中的项目要求，都同样指向了结构化建模，这么巧肯定说明了结构化建模是大势所趋。在这个19年都是老研究的AI领域，2020年年中的我还对已经火到巅峰的**图神经网络**一知半解就说不过去了。

本文根据以下三篇综述，旨在入门 **Graph Neural Networks**：

- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/pdf/1901.00596.pdf) | 2019 Jan
- [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/pdf/1812.08434.pdf) | 2018 Dec
- [How Powerful are Graph Neural Networks?]() | 2018 Oct



## 1 Introduction

我们知道，CNNs、RNNs以及 autoencoders 等深度学习方法，可以取代手工的特征提取，有效地捕获**欧氏**数据的隐含特征。但现实生活中，数据更普遍的形式是可以被构建为图的非欧数据。例如，化学分子结构、知识图谱、电子商务等。

由于图可能是不规则的，节点大小、邻居数量不同，从而传统深度学习难以应用于图域。此外，现有机器学习算法的核心假设是实例彼此独立。这种假设不再适用于图数据，因为每个实例（节点）通过各种类型的连接与其他节点相关联。

图神经网络主要解决 **表示对象之间复杂关系的非欧氏域数据处理问题**。传统的 Deep Leanring methods 可以视为图神经网络在欧氏情况下的**子集**。

<img src="./PR Structured Ⅰ GNN.assets/image-20200712132319666.png" alt="image-20200712132319666" style="zoom:50%;" />

现阶段的图神经网络可以分为以下四种：

- **recurrent graph neural networks**
- **convolutional graph neural networks**
- **graph autoen- coders**
- **spatial-temporal graph neural networks**

### 1.1 GNN 与 network embedding

Network embedding 旨在将网络的节点表示为低维的向量。

GNN 可以以端到端的方式抽取 high-level 的表征。

不同在于：GNN是一组针对各种任务而设计的神经网络模型，而网络嵌入涵盖了针对同一任务的各种方法。

### 1.2 GNN 与 graph kernel methods

graph kernel methods 是用于解决图分类问题，使SVM这种基于 kernel 的方法可以用于图数据的监督学习。此外，graph kernel methods 也可以用于 embed graphs or  nodes。区别在于，这种 embedding mapping 是确定性的 func，而非 learnable。GNN直接提取图信息的表征来做分类比 graph kernel methods 更有效。

### 1.3 Definition of GNN

**图 (Graph)** 常被表示为 $G=(V,E)$：

- V 表示**节点 (nodes) 或顶点 (vertices)**，E 表示连接节点的**边 (edges)**；
- $e_{ij}=(v_i,v_j)\in E$ 就代表了连接 i, j 两个节点的边；
- 节点 v 的**邻居**表示为：$N(v)=\{u\in V| (v,u)\in E \}$；
- 图可以写成**邻接矩阵**的形式，$A_{ij}=1 \text{  if  } e_{ij}\in E,A_{ij}=0 \text{  if  } e_{ij}\notin E$；
- 节点属性可以写作节点的特征矩阵 $\boldsymbol X$，边的属性可以写作边的特征矩阵 $\boldsymbol X^e$

**有向图 (Directed Graph)** 指所有边都从一个节点指向另一个节点的图，无向图是邻接矩阵对称的有向图。

**时空图 (Spatial-Temporal Graph)** 是一个属性图，其中节点属性随**时间动态变化**。$G^{(t)}=\left(\mathbf{V}, \mathbf{E}, \mathbf{X}^{(t)}\right) \text { with } \mathbf{X}^{(t)} \in \mathbf{R}^{n \times d}$

## 2 Categorization & Frameworks

### 2.1 Categorization 

#### Recurrent graph neural networks (RecGNNs)

GNN 的 先驱，假设图中的一个节点不断与其邻居交换信息/消息，直到达到稳定的平衡。其消息传递的思想被**空域卷积图神经网络 (spatial- based convolutional graph neural networks)**所继承。

#### Convolutional graph neural networks (ConvGNNs) 

主要分两种：

- Spectral methods
- Spatial methods

将卷积的思想从 grid data 拓展到 graph data。主要思想是通过汇总节点自身的特征 $x_v$ 和邻居的特征 $x_u$ 来生成节点 v 的表征形式，其中 $u\in N(v)$。通过堆叠多个图卷积层可以提取高级节点表征。下图 a 表示节点分类，b 表示图分类过程。


<img src="./PR Structured Ⅰ GNN.assets/image-20200712152008297.png" alt="image-20200712152008297" style="zoom:50%;" />

#### Graph autoencoders (GAEs)

是一种无监督的学习框架，可将节点/图编码到隐向量空间中，并从编码后的信息中重建图数据。用于学习网络嵌入和图生成分布。

- Network Embedding，GAE通过重建图结构信息（例如图邻接矩阵）来学习潜在节点表示。
- Graph Generation，某些方法逐步生成图的节点和边，而其他方法则一次全部输出图。

下图展示了用于网络嵌入的GAE。

<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712152040834.png" alt="image-20200712152040834" style="zoom:50%;" />

#### Spatial-temporal graph neural networks (STGNNs) 

STGNNs 同时考虑空间依赖性和时间依赖性。当前许多方法将图卷积与RNN或CNN集成在一起以捕获空间依赖性，从而对时间依赖性进行建模。旨在从时空图学习隐式特征。下图是用于时空图预测的STGNN。

<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712152050882.png" alt="image-20200712152050882" style="zoom:50%;" />

### 2.2 Frameworks

使用图结构和节点内容信息作为输入，GNN的输出可以通过以下机制之一专注于不同的图分析任务：

- Node-level：输出与节点回归和节点分类任务有关；
- Edge-level：输出与边缘分类和连接预测任务有关；
- Graph-level：输出与图分类任务有关。

训练方法：

- Semi-supervised learning for node-level classification：部分节点有 label，通过堆叠 ConvGNN + softmax 可以实现半监督的多分类；
- Supervised learning for graph-level classification：图级别的监督分类；
- Unsupervised learning for graph embedding：以两种方式利用边的信息
  - 采用自动编码器框架，其中编码器采用图卷积层将图嵌入到潜在表示中，在该表示中使用解码器来重构图结构；
  - 利用负采样方法，该方法将一部分节点对采样为负对，而图中具有链接的现有节点对为正对。然后应用逻辑回归层来区分正对和负对。

### 2.3 近年GNN算法总览

![image-20200712154103209](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712154103209.png)

![image-20200712155738229](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712155738229.png)

---

**下面详细介绍这四种类别 GNN 的理论与架构。**

## 3 Recurrent graph neural networks

**Vanilla GNN** (Scarselli et al.) 基于信息传播机制，根据邻居节点之间信息的循环交换来达到稳定的平衡态。节点的隐状态可表示为：
$$
\mathbf{h}_{v}^{(t)}=\sum_{u \in N(v)} f\left(\mathbf{x}_{v}, \mathbf{x}_{(v, u)}^{\mathbf{e}}, \mathbf{x}_{u}, \mathbf{h}_{u}^{(t-1)}\right)
$$
**Graph Echo State Network (GraphESN)** 提高了 Vanilla GNN 的训练效率。GraphESN 由一个 encoder 和一个 output layer 构成。它实现了contractive状态转移方程来循环更新节点状态，直到全局图状态达到收敛为止。之后，通过将固定的节点状态作为输入来训练输出层。

**Gated Graph Neural Network (GGNN)** 利用GRU作为循环方程，将重复执行减少到固定的步骤数。好处是不再需要约束参数以确保收敛。节点隐状态由其先前的隐状态及其相邻的隐状态更新：
$$
\mathbf{h}_{v}^{(t)}=G R U\left(\mathbf{h}_{v}^{(t-1)}, \sum_{u \in N(v)} \mathbf{W} \mathbf{h}_{u}^{(t-1)}\right)
$$
其使用 BPTT 训练参数，所有节点的中间状态要存储在内存中，可能不易于大型图的计算。

**Stochastic Steady-state Embedding (SSE)** 以随机和异步方式周期性地更新节点隐状态，解决了扩展到大型图的问题。它交替的采样 batch 用于节点的状态更新与节点的梯度计算。为保证稳定性，SSE 取了过去状态与现在状态的加权平均。
$$
\mathbf{h}_{v}^{(t)}=(1-\alpha) \mathbf{h}_{v}^{(t-1)}+\alpha \mathbf{W}_{1} \sigma\left(\mathbf{W}_{2}\left[\mathbf{x}_{v}, \sum_{u \in N(v)}\left[\mathbf{h}_{u}^{(t-1)}, \mathbf{x}_{u}\right]\right]\right)
$$
SSE 并没有收敛的理论保证。



## 4 Convolutional graph neural networks

区别于递归图神经网络，ConvGNN不是使用收缩约束来迭代节点状态，而是使用固定数量的具有不同权重的层在体系结构上解决循环的相互依赖性。

<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712161616515.png" alt="image-20200712161616515" style="zoom:50%;" />

- Spectral- based：通过从图信号处理的角度引入滤波器来定义图卷积，其中图卷积运算被解释为从图信号中去除噪声。
- Spatial-based：继承了RecGNN的思想，以通过信息传播来定义图卷积。自从GCN 弥合了基于频域的方法与基于空间的方法之间的差距以来，基于空间的方法由于其引人注目的效率，灵活性和通用性而迅速发展。

### 4.1 Spectral-based ConvGNNs

#### 4.1.1 基础

卷积操作本身就是从数字信号处理中引申出来的，所以先有了基于频域的方法并不奇怪。该方法假设图是无向的。归一化图拉普拉斯矩阵是无向图的数学表示，定义为：
$$
\mathbf{L}=\mathbf{I}_{\mathbf{n}}-\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}
$$
D 是表示 node degrees 的对角矩阵，$\mathbf{D}_{i i}=\sum_{j}\left(\mathbf{A}_{i, j}\right)$。

归一化图拉普拉斯矩阵具有实对称半正定性质，可被分解成特征矩阵 $\mathbf{L}=\mathbf{U} \mathbf{\Lambda} \mathbf{U}^{T}$。

在**图信号处理**中，一个图信号 $\bold{X}$ 表示图节点的 feature vector，其 **图傅里叶变换** 表示为 $\mathscr{F}(\mathbf{x})=\mathbf{U}^{T} \mathbf{x}$，逆变换为 $\mathscr{F}^{-1}(\hat{\mathbf{x}})=\mathbf{U} \hat{\mathbf{x}}$。图傅立叶变换将输入图信号投影到正交空间，正交基由归一化图拉普拉斯算子的特征向量形成。现在图信号就被映射到频域空间。**基于频域的图卷积**定义为：
$$
\begin{aligned}
\mathbf{x} *_{G} \mathbf{g} &=\mathscr{F}^{-1}(\mathscr{F}(\mathbf{x}) \odot \mathscr{F}(\mathbf{g})) \\
&=\mathbf{U}\left(\mathbf{U}^{T} \mathbf{x} \odot \mathbf{U}^{T} \mathbf{g}\right)
\end{aligned}
$$
可将其中的 $\mathbf{g}_{\theta}=\operatorname{diag}\left(\mathbf{U}^{T} \mathbf{g}\right)$ 定义为**滤波器 (filter)**。定义简化为
$$
\mathbf{x} *_{G} \mathbf{g}_{\theta}=\mathbf{U} \mathbf{g}_{\theta} \mathbf{U}^{T} \mathbf{x}
$$

#### 4.1.2 方法

**Spectral Convolutional Neural Network (Spectral CNN)** 将滤波器视作可学的参数并考虑具有多个通道的图信号。图卷积层定义为：
$$
\mathbf{H}_{:, j}^{(k)}=\sigma\left(\sum_{i=1}^{f_{k-1}} \mathbf{U} \boldsymbol{\Theta}_{i, j}^{(k)} \mathbf{U}^{T} \mathbf{H}_{:, i}^{(k-1)}\right) \quad\left(j=1,2, \cdots, f_{k}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712165321733.png" alt="image-20200712165321733" style="zoom: 50%;" />

由于喇布拉斯矩阵的特征分解，该方法有三个缺点：

1. 对图的任何扰动都会导致本征基的变化；
2. 学习的滤波器是域相关的，这意味着它们不能应用于具有不同结构的图；
3. 特征分解需要O(n^3)计算复杂度。

**Chebyshev Spectral CNN (ChebNet) ** 通过特征值对角矩阵的Chebyshev多项式近似滤波器func。图卷积表示为：
$$
\mathbf{x} *_{G} \mathbf{g}_{\theta}=\mathbf{U}\left(\sum_{i=0}^{K} \theta_{i} T_{i}(\tilde{\boldsymbol{\Lambda}})\right) \mathbf{U}^{T} \mathbf{x}
$$
由ChebNet定义的滤波器在空间中进行了局部定位，这意味着滤波器可以独立于图形大小提取局部特征。 
**CayleyNet** 进一步应用了Cayley多项式，它们是参数有理复函数，可以捕获狭窄的频带。
$$
\mathbf{x} *_{G} \mathbf{g}_{\theta}=c_{0} \mathbf{x}+2 \operatorname{Re}\left\{\sum_{j=1}^{r} c_{j}(h \mathbf{L}-i \mathbf{I})^{j}(h \mathbf{L}+i \mathbf{I})^{-j} \mathbf{x}\right\}
$$
**Graph Convolutional Network (GCN) ** 引入了ChebNet的一阶近似。假设 K=1, $\lambda_{max}=2$，公式7简化为：
$$
\mathbf{x} *_{G} \mathbf{g}_{\theta}=\theta_{0} \mathbf{x}-\theta_{1} \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \mathbf{x}
$$
为限制参数数量并避免过度拟合，GCN假设 $\theta=\theta_{0}=-\theta_{1}$，即为：
$$
\mathbf{x} *_{G} \mathbf{g}_{\theta}=\theta\left(\mathbf{I}_{\mathbf{n}}+\mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}}\right) \mathbf{x}
$$
为了允许输入和输出的多通道，GCN将公式10修改为 a compositional layer，定义为
$$
\mathbf{H}=\mathbf{X} *_{G} \mathbf{g}_{\Theta}=f(\overline{\mathbf{A}} \mathbf{X} \Theta)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712170431243.png" alt="image-20200712170431243" style="zoom:50%;" />

**GCN 除了是基于频域的方法，还融合了基于空域的方法，是二者的集大成者。**

从空域角度看，GCN可以被视为汇总来自节点邻域的特征信息。公式11可以表示为：
$$
\mathbf{h}_{v}=f\left(\boldsymbol{\Theta}^{T}\left(\sum_{u \in\{N(v) \cup v\}} \bar{A}_{v, u} \mathbf{x}_{u}\right)\right) \quad \forall v \in V
$$
GCN 的问题在于使用了特征对称矩阵，下面的更新方法旨在寻找替代的对称阵。

**Adaptive Graph Convolutional Network (AGCN)** 学习图邻接矩阵未指定的隐藏结构关系。它通过一个可学习的距离函数构造一个所谓的**残差图邻接矩阵**，该距离函数将两个节点的特征作为输入。

**Dual Graph Convolutional Network (DGCN)** 引入具有两个并行图卷积层的双图卷积体系结构。这两个层共享参数，它们使用归一化的邻接矩阵A和 positive pointwise mutual information (PPMI) 矩阵，该矩阵通过从图上采样的random walk捕获节点 co-occurrence 信息。 PPMI 矩阵为：
$$
\mathbf{P} \mathbf{P} \mathbf{M} \mathbf{I}_{v_{1}, v_{2}}=\max \left(\log \left(\frac{\operatorname{count}\left(v_{1}, v_{2}\right) \cdot|D|}{\operatorname{count}\left(v_{1}\right) \operatorname{count}\left(v_{2}\right)}\right), 0\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712171214741.png" alt="image-20200712171214741" style="zoom:50%;" />

通过对双图卷积层的输出进行集成，DGCN可以对局部和全局结构信息进行编码，而无需堆叠多个图卷积层。

### 4.2 Spatial-based ConvGNNs

空域图卷积就很好理解了，和图像的卷积一样，根据节点之间的空间关系进行卷积。

<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712132319666.png" style="zoom:50%;" />

**Neural Network for Graphs (NN4G)** 通过a compositional neural architecture在每一层具有独立参数来学习图相互依赖关系。通过直接加和节点的邻域信息来执行图卷积。NN4G 下一层的节点状态为：
$$
\mathbf{h}_{v}^{(k)}=f\left(\mathbf{W}^{(k)^{T}} \mathbf{x}_{v}+\sum_{i=1}^{k-1} \sum_{u \in N(v)} \Theta^{(k)^{T}} \mathbf{h}_{u}^{(k-1)}\right)
$$
NN4G使用未归一化的邻接矩阵，这可能潜在地导致隐藏节点状态具有极为不同的尺度。

**Contextual Graph Markov Model (CGMM)** 是受NN4G启发的概率模型。在保持空间局部性的同时，CGMM具有概率解释性的优势。

**Diffusion Convolutional Neural Network (DCNN)** 将卷积图视为扩散过程。它假定信息以一定的转移概率从一个节点转移到其相邻节点之一，以便信息分配可以在几轮之后达到平衡。 DCNN将扩散图卷积定义为：
$$
\mathbf{H}^{(k)}=f\left(\mathbf{W}^{(k)} \odot \mathbf{P}^{k} \mathbf{X}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712172004136.png" alt="image-20200712172004136" style="zoom:50%;" />

请注意，在DCNN中，隐藏表示矩阵H（k）的维数与输入特征矩阵X相同，并且不是其先前的隐藏表示矩阵H（k-1）的函数。 DCNN将H（1），H（2），...，H（K）串联在一起作为最终模型输出。

由于扩散过程的平稳分布是概率转移矩阵的幂级数的总和，因此 **Diffusion Graph Convolution (DGC)** 会在每个扩散步骤中汇总输出，而不是进行级联。定义为：
$$
\mathbf{H}=\sum_{k=0}^{K} f\left(\mathbf{P}^{k} \mathbf{X} \mathbf{W}^{(k)}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712172227154.png" alt="image-20200712172227154" style="zoom:50%;" />

使用转移概率矩阵的功效意味着遥远的邻居向中央节点贡献的信息很少。

**PGC-DGCNN** 根据最短路径增加远方邻居的贡献。定义了一个最短路矩阵 $S^{(j)}$，如果节点 v 和 u 之间的最短路长度 为 j，那么矩阵上对应元素为 1，否则为 0。并用超参数r控制接收场大小：
$$
\mathbf{H}^{(k)}=\|_{j=0}^{r} f\left(\left(\tilde{\mathbf{D}}^{(j)}\right)^{-1} \mathbf{S}^{(j)} \mathbf{H}^{(k-1)} \mathbf{W}^{(j, k)}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712174617755.png" alt="image-20200712174617755" style="zoom:50%;" />

最短路径邻接矩阵的计算复杂度较高。

**Partition Graph Convolution (PGC)** 根据某些条件（不限于最短路径）将节点的邻居分为Q组。PGC根据每个组定义的邻域构造Q邻接矩阵。然后，PGC将具有不同参数矩阵的GCN 应用于每个邻居组，并对结果求和：
$$
\mathbf{H}^{(k)}=\sum_{j=1}^{Q} \overline{\mathbf{A}}^{(j)} \mathbf{H}^{(k-1)} \mathbf{W}^{(j, k)}
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712175300499.png" alt="image-20200712175300499" style="zoom:50%;" />

**Message Passing Neural Network (MPNN)** 它将图卷积视为消息传递过程，其中信息可以直接从一个节点沿边缘传递到另一个节点。 MPNN运行K步消息传递迭代，以使信息进一步传播。消息传递函数（即空间图卷积）定义为
$$
\mathbf{h}_{v}^{(k)}=U_{k}\left(\mathbf{h}_{v}^{(k-1)}, \sum_{u \in N(v)} M_{k}\left(\mathbf{h}_{v}^{(k-1)}, \mathbf{h}_{u}^{(k-1)}, \mathbf{x}_{v u}^{e}\right)\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712175439303.png" alt="image-20200712175439303" style="zoom:50%;" />

在得出每个节点的隐式表征之后，可以将 $\bold{h}_v^{(K)}$ 传递到输出层以执行节点级预测任务，或者传递给读出功能以执行图形级预测任务。读出功能基于节点隐藏表示生成整个图的表示。
$$
\mathbf{h}_{G}=R\left(\mathbf{h}_{v}^{(K)} \mid v \in G\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712175812940.png" alt="image-20200712175812940" style="zoom:50%;" />

**Graph Isomorphism Network (GIN)** 发现以前的基于MPNN的方法无法基于它们产生的图嵌入来区分不同的图结构。为了修正该缺点，GIN通过可学习的参数Δ（k）来调整中心节点的权重。它通过执行图卷积：
$$
\mathbf{h}_{v}^{(k)}=M L P\left(\left(1+\epsilon^{(k)}\right) \mathbf{h}_{v}^{(k-1)}+\sum_{u \in N(v)} \mathbf{h}_{u}^{(k-1)}\right)
$$
由于节点的邻居数量可能从一千到一千甚至更多不等，因此无法充分利用节点邻居的全部大小。

**GraphSage** 通过采样为每个节点获取固定数量的邻居。
$$
\mathbf{h}_{v}^{(k)}=\sigma\left(\mathbf{W}^{(k)} \cdot f_{k}\left(\mathbf{h}_{v}^{(k-1)},\left\{\mathbf{h}_{u}^{(k-1)}, \forall u \in S_{\mathcal{N}(v)}\right\}\right)\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712180112851.png" alt="image-20200712180112851" style="zoom:50%;" />

**Graph Attention Network (GAT)** 假设相邻节点对中心节点的贡献既不像GraphSage 一样，也不像 GCN 那样预先确定。GAT 用 attention mechanism 得到两个相连节点之间的相对权重。
$$
\mathbf{h}_{v}^{(k)}=\sigma\left(\sum_{u \in \mathcal{N}(v) \cup v} \alpha_{v u}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_{u}^{(k-1)}\right)
$$
$\alpha$ 是 attention score，此处就不介绍了。GAT进一步执行多头关注，以提高模型的表达能力。这显示了在节点分类任务上与GraphSage相比有显着改进。然而GAT假设 attention heads 的贡献是相等的。

**Gated Attention Network (GAAN)** 引入了一种 self-attention mechanism，该机制可以为每个attention head计算一个额外的 attention score。除了在空间上施加图注意力之外，**GeniePath** 进一步提出了一种LSTM式的 gating mechanism 来控制跨图卷积层的信息流。

**Mixture Model Network (MoNet)** 采用不同的方法为节点的邻居分配不同的权重。
它引入节点伪坐标以确定节点与其邻居之间的相对位置。一旦知道两个节点之间的相对位置，权重函数会将相对位置映射到这两个节点之间的相对权重。这样，可以在不同位置之间共享图滤波器的参数。MoNet还提出了一个具有可学习参数的高斯核，以自适应地学习权重函数。

**另一类独特的方法是，根据某些标准对节点的邻居进行排名，并将每个排名与可学习的权重相关联，从而在不同位置实现权重分配。**

**PATCHY-SAN** 根据它们的图标签对每个节点的邻居进行排序，并选择前q个邻居。
图标记本质上是节点分数，可以通过节点度，中心性和Weisfeiler-Lehman颜色得出。由于每个节点现在都有固定数量的有序邻居，因此可以将图结构的数据转换为网格结构的数据。 PATCHY-SAN应用标准的1D卷积过滤器来汇总邻域特征信息，其中过滤器权重的顺序对应于节点邻居的顺序。PATCHY-SAN的排序标准仅考虑图形结构，这需要大量计算才能进行数据处理。

**Large- scale Graph Convolutional Network (LGCN)** 根据节点特征信息对节点的邻居进行排名。LGCN组装一个由其邻域组成的特征矩阵，并沿每一列对该特征矩阵进行排序。排序后的特征矩阵的前q行被用作中心节点的输入数据。

**此外，还有一些提升训练效率的算法**

ConvGNN 训练时通常需要将整个图数据和所有节点的中间状态保存到内存中。ConvGNN的全批次训练算法明显遭受内存溢出问题的困扰，尤其是当一个图包含数百万个节点时。

**Fast Learning with Graph Convolutional Network (Fast-GCN)** 为每个图卷积层采样固定数量的节点，而不是像GraphSage 那样为每个节点采样固定数量的邻居。它把图卷积解释为概率测度下节点嵌入函数的积分变换。采用蒙特卡洛近似和方差减少技术来简化训练过程。由于FastGCN针对每个层独立采样节点，因此层间连接可能稀疏。

**Stochastic Training of Graph Convolutional Networks (StoGCN)** 使用历史节点表示作为控制变量，将图卷积的接收场大小减小到任意小比例。即使每个节点有两个邻居，StoGCN仍可达到可比的性能。但是，StoGCN仍然必须保存所有节点的中间状态，这对于大型图形来说是消耗内存的。

**Cluster-GCN** 使用图聚类算法对子图进行采样，并对所采样的子图中的节点执行图卷积。由于在采样子图中还限制了邻域搜索，因此Cluster-GCN能够处理更大的图，并同时使用更少的时间和更少的内存使用更深的体系结构。 Cluster-GCN特别是可以为现有ConvGNN训练算法提供时间复杂度和内存复杂度的直接比较。

![image-20200712195813501](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712195813501.png)

**总而言之，由于效率、通用性和灵活性问题，空域模型优于频域模型**

1. 频域模型要么需要执行特征向量计算，要么需要同时处理整个图形；
2. 依赖于图傅立叶基础的光谱模型不能很好地推广到新图；
3. 基于频谱的模型仅限于在无向图上运行。

### 4.3 Graph Pooling Modules 

GNN生成节点特征后，我们可以将其用于最终任务。但是直接使用所有这些功能可能在计算上具有挑战性，因此，需要一种 **down-sampling 策略**。
根据目标及其在网络中所扮演的角色，为该策略指定了不同的名称：

- 合并操作旨在通过对节点进行下采样以生成较小的表示，从而减小参数的大小，从而避免过拟合，排列不变性和计算复杂性问题； 
- readout 操作主要用于基于节点表示生成图级表示。它们的机制非常相似。

传统的池化如下：
$$
\mathbf{h}_{G}=\operatorname{mean} / \max / \operatorname{sum}\left(\mathbf{h}_{1}^{(K)}, \mathbf{h}_{2}^{(K)}, \ldots, \mathbf{h}_{n}^{(K)}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712203453039.png" alt="image-20200712203453039" style="zoom:50%;" />

在网络开始时执行简单的最大/均值池化对于降低图域的维数和减轻昂贵的图傅立叶变换操作的成本尤为重要。

最新的图池化方法是 **differentiable pooling (DiffPool)** ，可以生成图形的层次表示。DiffPool不仅可以简单地将图中的节点聚类，还可以在第 k 层学习聚类分配矩阵 S，称为 $\mathbf{S}^{(k)} \in \mathbf{R}^{n_{k} \times n_{k+1}}$，其中 $n_k$ 是第 k 层的节点数。S 矩阵中的概率值是根据节点的特征和拓扑结构生成：
$$
\mathbf{S}^{(k)}=\operatorname{softmax}\left(\operatorname{Conv} G N N_{k}\left(\mathbf{A}^{(k)}, \mathbf{H}^{(k)}\right)\right)
$$
**核心思想**是学习综合的节点分配，该分配同时考虑图形的拓扑和特征信息，因此可以使用任何标准的ConvGNN来实现上述公式。但是，DiffPool的缺点是在合并后会生成密集图，此后计算复杂度变为O(n^2)。

**SAGPool** 同时考虑节点特征和图拓扑并以 self-attention 方式学习池化。
**总的来说，池化是减小图形大小的基本操作。如何提高合并的有效性和计算复杂性是一个尚待研究的问题。**

## 5 Graph Autoencoders

### 5.1 Network Embedding

网络嵌入是节点的低维向量表示，它保留节点的拓扑信息。GAE使用编码器提取网络嵌入，并使用解码器执行网络嵌入以保留图拓扑信息（例如PPMI矩阵和邻接矩阵）来学习网络嵌入。

**Vanilla Graph Autoencoder (GAE)** 利用GCN同时编码节点结构信息和节点特征信息。GAE 的**编码器**由两个图卷积层组成，其形式为
$$
\mathbf{Z}=\operatorname{enc}(\mathbf{X}, \mathbf{A})=\operatorname{Gconv}\left(f\left(G \operatorname{conv}\left(\mathbf{A}, \mathbf{X} ; \mathbf{\Theta}_{\mathbf{1}}\right)\right) ; \mathbf{\Theta}_{\mathbf{2}}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712212800180.png" alt="image-20200712212800180" style="zoom:50%;" />

GAE 的**解码器**旨在通过重建图邻接矩阵，从节点的嵌入中解码出节点关系信息，其定义为
$$
\hat{\mathbf{A}}_{v, u}=\operatorname{dec}\left(\mathbf{z}_{v}, \mathbf{z}_{u}\right)=\sigma\left(\mathbf{z}_{v}^{T} \mathbf{z}_{u}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712212854141.png" alt="image-20200712212854141" style="zoom:50%;" />

由于自动编码器的容量，仅重构图邻接矩阵可能会导致过度拟合。

**Variational Graph Autoencoder (VGAE)** 是GAE的变分版本，可以学习数据的分布。VGAE优化变分下限L：
$$
L=E_{q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})}[\log p(\mathbf{A} \mid \mathbf{Z})]-K L[q(\mathbf{Z} \mid \mathbf{X}, \mathbf{A}) \| p(\mathbf{Z})]
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712213349638.png" alt="image-20200712213349638" style="zoom:50%;" />

<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712213614662.png" alt="image-20200712213614662" style="zoom:50%;" />

**Adversarially Regularized Variational Graph Autoencoder (ARVGA)** 通过采用生成对抗网络（GAN）的训练方案，加强了经验分布对先验分布的拟合。ARVGA努力学习一种编码器，使该编码器产生的经验分布 q(Z|X, A) 与先前的分布 p(Z) 不能区分开。

**以上方法本质上是通过解决链路预测问题来学习网络嵌入的。图的稀疏性导致正节点对的数量远远少于负节点对的数量。**

为了减轻学习网络嵌入中的数据稀疏性问题，另一行著作通过 random permutations or random walks 将图转换为序列。通过这种方式，那些适用于序列的深度学习方法可以直接用于处理图形。

**Deep Recursive Network Embedding (DRNE)** 假设节点的网络嵌入应该近似于其邻居网络嵌入的聚合。它采用长短期记忆（LSTM）网络来聚合节点的邻居。DRNE的重建误差定义为：
$$
L=\sum_{v \in V}\left\|\mathbf{z}_{v}-L S T M\left(\left\{\mathbf{z}_{u} \mid u \in N(v)\right\}\right)\right\|^{2}
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712214932503.png" alt="image-20200712214932503" style="zoom:50%;" />

DRNE通过LSTM网络而不是使用LSTM网络生成网络嵌入来隐式学习网络嵌入。它避免了LSTM网络对于节点序列的排列不是不变的问题。
**Network Representations with Adversarially Regularized Autoencoders (NetRA)** 的网络表示出了一种图编码器-解码器框架，该框架具有一般性的损失函数，定义为：
$$
L=-E_{\mathbf{z} \sim P_{\text {data}}(\mathbf{z})}(\operatorname{dist}(\mathbf{z}, \operatorname{dec}(\operatorname{enc}(\mathbf{z}))))
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712215137652.png" alt="image-20200712215137652" style="zoom:50%;" />

NetRA的编码器和解码器是LSTM  with random walks rooted，每个节点v∈V作为输入。类似于ARVGA，NetRA通过对抗训练将学习到的网络嵌入在事先的分布中进行正规化。尽管NetRA忽略了LSTM网络的节点置换变量问题，但是实验结果验证了NetRA的有效性。 



### 5.2 Graph Generation

- sequential manner：将图线性化为序列，由于存在循环，它们可能会丢失结构信息。
- global manner：一次输出整个图，不可扩展。

#### 5.2.1  Sequential manner 

**Deep Generative Model of Graphs (DeepGMG)**  假设图的概率是所有可能的节点排列之和
$$
p(G)=\sum_{\pi} p(G, \pi)
$$
其中π表示节点顺序。它捕获图中所有节点和边的复杂联合概率。DeepGMG通过做出一系列决策来生成图形，即是否添加节点，要添加哪个节点，是否添加边缘以及要连接到新节点的节点。生成节点和边的决策过程取决于RecGNN更新的成长图的节点状态和图状态。

**GraphRNN** 提出了图级RNN和边级RNN来建模节点和边的生成过程。
每次，图级RNN都会向节点序列添加一个新节点，而边缘级RNN会生成一个二进制序列，该二进制序列指示新节点与该序列中先前生成的节点之间的连接。

#### 5.2.2  Global manner

**Graph Variational Autoencoder (GraphVAE)** 将节点和边的存在建模为独立随机变量。通过假设由编码器定义的后验分布 $q_{\phi}(\mathbf{z} \mid G)$ 和由解码器定义的生成分布 $p_{\theta}(G \mid \mathbf{z})$，GraphVAE优化了变化下限：
$$
L(\phi, \theta ; G)=E_{q_{\phi}(z \mid G)}\left[-\log p_{\theta}(G \mid \mathbf{z})\right]+K L\left[q_{\phi}(\mathbf{z} \mid G) \| p(\mathbf{z})\right]
$$
通过使用ConvGNN作为编码器，并使用简单的多层感知作为解码器，GraphVAE输出生成的图形及其邻接矩阵，节点属性和边缘属性。控制生成图的全局属性（例如图连接性，有效性和节点兼容性）具有挑战性。

**Regularized Graph Variational Autoencoder (RGVAE)** 进一步将有效性约束强加在图变分自动编码器上，以规范解码器的输出分布。

**Molecular Generative Adversarial Network (MolGAN)**  集成convGNN，GAN 和强化学习目标，以生成具有所需属性的图。 MolGAN由生成器和鉴别器组成，它们相互竞争以提高生成器的真实性。在MolGAN中，生成器尝试提出一个伪图及其特征矩阵，而鉴别器的目的是将伪样本与经验数据区分开。此外，根据区分器，引入了与鉴别器并行的奖励网络，以鼓励生成的图具有某些属性。 

**NetGAN** 将LSTM [7]与Wasserstein GAN [116]结合使用，可以从基于随机游走的方法生成图。NetGAN训练生成器通过LSTM网络生成合理的随机游走，并强制执行鉴别器以从真实的随机游走中识别出虚假的随机游走。训练后，一个新的图是由归一化基于生成器产生的随机游走而计算出的节点的共现矩阵。



## 6 Spatial-temporal graph neural networks

在许多实际应用程序中，图在图结构和图输入方面都是动态的。时空图神经网络（STGNN）在捕获图的动态性方面占据重要位置。此类方法旨在模拟动态节点输入，同时假设已连接节点之间的相互依赖性。例如，交通网络由放置在道路上的速度传感器组成，在道路上，边缘权重由传感器对之间的距离确定。由于一条道路的交通状况可能取决于其邻近道路的状况，因此在进行交通速度预测时，必须考虑空间依赖性。作为解决方案，STGNN可同时捕获图的空间和时间依赖性。STGNN的任务可以是预测未来的节点值或标签，或预测时空图标签。 

- 基于RNN的方法
- 基于CNN的方法

### 6.1 RNN-based

假设 simple RNN 是：
$$
\mathbf{H}^{(t)}=\sigma\left(\mathbf{W} \mathbf{X}^{(t)}+\mathbf{U H}^{(t-1)}+\mathbf{b}\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712230809276.png" alt="image-20200712230809276" style="zoom:50%;" />

加入图卷积
$$
\mathbf{H}^{(t)}=\sigma\left(G \operatorname{conv}\left(\mathbf{X}^{(t)}, \mathbf{A} ; \mathbf{W}\right)+G \operatorname{conv}\left(\mathbf{H}^{(t-1)}, \mathbf{A} ; \mathbf{U}\right)+\mathbf{b}\right)
$$
**Graph Convo- lutional Recurrent Network (GCRN) ** 结合了LSTM网络和ChebNet。
**Diffusion Convolutional Recur- rent Neural Network (DCRNN)** 将 diffusion graph convolutional layer 并到GRU网络中。另外，DCRNN采用编解码器框架来预测节点值的未来K步。

**此外，一些工作使用节点级RNN和边缘级RNN来处理时间信息的不同方面。**

**Structural-RNN** 提出了一个递归框架来预测每个时间步的节点标签。它包括两种RNN，即节点RNN和边缘RNN。每个节点和每个边缘的时间信息分别通过节点-RNN和边缘-RNN。为了合并空间信息，节点RNN将边缘RNN的输出作为输入。由于为不同的节点和边缘假设不同的RNN会大大增加模型的复杂性，因此它将节点和边缘分成语义组。同一语义组中的节点或边共享相同的RNN模型，从而节省了计算成本。

**基于RNN的方法存在耗时的迭代传播和梯度爆炸/消失问题**

### 6.2 CNN-based

**CGCN** 将1D卷积层与ChebNet 或GCN 层集成在一起。它通过按顺序堆叠门控的1D卷积层，图卷积层和另一个门控的1D卷积层来构造时空块。
**ST-GCN**使用1D卷积层和PGC层组成时空块。

先前的方法都使用预定义的图结构。他们假设预定义的图结构反映了节点之间真正的依赖关系。但是，利用时空设置中的许多图形数据快照，可以从数据中自动学习潜在的静态图形结构。

**WaveNet** 提出了一种自适应邻接矩阵来执行图卷积。自适应邻接矩阵定义为:
$$
\mathbf{A}_{a d p}=\operatorname{SoftMax}\left(\operatorname{ReLU}\left(\mathbf{E}_{1} \mathbf{E}_{2}^{T}\right)\right)
$$
<img src="D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200712231606429.png" alt="image-20200712231606429" style="zoom:50%;" />

通过将E1乘以E2，可以得到源节点和目标节点之间的依存关系权重。借助基于CNN的复杂时空神经网络，Graph WaveNet无需给出邻接矩阵即可表现良好。 

**GaAN** 利用注意力机制通过基于RNN的方法学习动态空间依赖性。注意力机制用于根据给定的两个节点的当前节点输入来更新它们之间的边缘权重。 
**ASTGCN** 进一步包括空间注意力机制和时间注意力机制，以通过基于CNN的方法学习潜在的动态空间相关性和时间相关性。学习潜在空间相关性的共同缺点是，它需要计算每对节点之间的空间相关性权重，这需要 O(n^2)。

## 7 Applications



## 8 Future Directions