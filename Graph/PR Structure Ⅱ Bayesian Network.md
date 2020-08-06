# PR Structured Ⅱ：Structured Probabilistic Model Ⅰ

本文是结构化概率模型的简明入门指南，主要介绍了结构化模型的优势、有向图、无向图等表示方式以及如何利用结构化模型进行推断等。详细的方法介绍可见扩展阅读。

[TOC]

## 结构化概率模型的初衷

在非结构化建模中，我们已经有了**条件概率公式/贝叶斯公式**来描述概率分布中随机变量之间的相互关系。然而非结构化建模在深度学习领域中受到越来越多的挑战，这些任务需要对输入数据的整个结构有完整的理解：

- 估计密度函数
- 去噪
- 缺失值的填补
- 采样

对一个包含 n 个离散变量并且每个变量都能取 k 个值的 x 的分布建模，那么最简单的表示 P(x) 的方法需要存储一个可以查询的表格。这个表格记录了**每一种可能值的概率**，则需要 $k^n$ 个参数。 这就意味着，对一个只有 32×32 像素的彩色（RGB）图片 来说，存在 23072 种可能的二值图片。这个数量已经超过了 $10^{800}$，比宇宙中的原子总数还要多。 

- 存储参数的开销大；
- 统计效率低：当模型中的参数个数增加时，使用统计估计器估计这些参数所需要的训练数据数量也需要相应地增加；
- 推断的时间开销大； 
- 采样的时间开销大。 

**如何简化？**

以表格形式记录所有可能的变量导致了时间、空间上的挑战，然而我们真的需要记录每一种可能的相互作用吗？**通常，许多变量只是间接地相互作用。**例如：

> A，B，C 按次序接力跑，在已知B完成时间的情况下，A的完成时间对估计C的完成时间并无帮助，即A与C只有**间接地**关系，我们可以忽略这样地间接关系。

**结构化概率模型**为随机变量之间的**直接作用**提供了一个正式的建模框架。这种方式大大减少了模型的参数个数以致于模型只需要更少的数据来进行有效的估计，减小了在模型存储、模型推断以及从模型中采样时的计算开销。

此外，结构化概率模型允许我们明确地将给定的现有知识与知识的学习或者推断分开。我们可以设计、 分析和评估适用于更广范围的图的学习算法和推断算法。同时，我们可以设计能够捕捉到我们认为数据中存在的重要关系的模型。

## Graph (图) 结构

结构化概率模型使用图 $G=(V,E)$ 来表示随机 变量之间的相互作用。

> 一定要区分图论中，图也就是 Graph 与图像 (Image) 的区别，不要混为一谈。

### 有向图结构 / 贝叶斯网络

**有向图模型**（directed graphical model）是一种结构化概率模型，也被称为**信 念网络**（belief network）或者**贝叶斯网络**（Bayesian network）(Pearl, 1985)。 

![image-20200719135706275](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structured Ⅰ GNN.assets\image-20200719135706275.png)

因此，变量 x 的概率分布就可以由一系列**局部条件概率分布**表示：
$$
p(\mathbf{x})=\prod_{i} p\left(\mathrm{x}_{i} \mid \operatorname{Pa}_{\mathcal{G}}\left(\mathrm{x}_{i}\right)\right)
$$
$P_{a\mathcal G}(x_i)$ 表示 x 的所有父节点。

因此上图的概率分布为：
$$
p\left(\mathrm{t}_{0}, \mathrm{t}_{1}, \mathrm{t}_{2}\right)=p\left(\mathrm{t}_{0}\right) p\left(\mathrm{t}_{1} \mid \mathrm{t}_{0}\right) p\left(\mathrm{t}_{2} \mid \mathrm{t}_{1}\right)
$$
有向图的方式极大地简化了我们需要保存的信息，降低了复杂度。只要图中的每个变量都只有少量的父结点（而不是全部其他节点），那么这个分布就可以用较少的参数来表示。

### 无向图结构 / 马尔可夫随机场

**无向模型**（undirected Model），也被称为**马尔可夫随机场**（Markov random ﬁeld, MRF）或者是**马尔可夫网络**（Markov network）(Kindermann, 1980)。无向模型中所有的边都是没有方向的。 

并不是所有情况的相互作用都有一个明确的方向关系。例如：

> 你是否生病，你的同事是否生病以及你的室友是否生病。

我们把对应你健康状况的随机变量记作 $h_y$，对应你的室友健康状况的随机变量记作 $h_r$，你的同事健康的变量记作 $h_c$。下图表示这种关系。 

![](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719141215552.png)

对于图中的每一个**团** C，可用一个**因子**（factor）$\phi(C)$ (也称为**团势能**（clique potential）)，衡量了团中变量每一种可能的联合状态所对应的密切程度。这些因子都被限制为是 非负的。它们一起定义了**未归一化概率函数**
$$
\tilde{p}(\mathbf{x})=\prod_{\mathcal{C} \in \mathcal{G}} \phi(\mathcal{C})
$$
只要所有团中的结点数都不大，那么我们就能够高效地处理这些未归一化概率函数。如下图所示：

![image-20200719141645589](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719141645589.png)

说回刚刚的例子，以团 $(h_y,h_c)$ 为例，1为健康，0为感冒：

![image-20200719142143932](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719142143932.png)

- 两个通常都是健康的，所以对应的状态拥有最高的密切程度；
- 两个人中只有一个人是感冒的密切程度是最低的；
- 两个人都感冒的状态（通过一个人来传染给了另一个人）的密切程度稍高。



### 配分函数 / 归一化常数

无向模型的团构建的非归一化概率函数的概率之和或积分不一定为1，为了得到一个有效的概率分布，我们需要使用对应的归一化的概率分布： 
$$
p(\mathbf{x})=\frac{1}{Z} \tilde{p}(\mathbf{x})
$$
其中，Z 是使得所有的概率之和或者积分为 1 的常数，被称作 **归一化常数** 或 **配分函数**，其定义很明显是：
$$
Z=\int \tilde{p}(\mathbf{x}) d \mathbf{x}
$$
那么问题来了，**如何高效地求出这个配分函数**？直接求积分是很愚蠢地方法，通常我们使用**近似技术**。

[配分函数——深度学习第十八章](https://zhuanlan.zhihu.com/p/48552020)

### 基于能量的模型 EBM

无向模型中许多有趣的理论结果都依赖于 $\forall \boldsymbol{x}, \tilde{p}(\boldsymbol{x})>0$ 这个假设。使这个条件 满足的一种简单方式是使用基于能量的模型（Energy-based model, EBM），其中
$$
\tilde{p}(\mathbf{x})=\exp (-E(\mathbf{x}))
$$
E(x) 被称作是**能量函数**（energy function）。很明显，exp函数的引入保证了上述假设的成立。服从上式形式的任意分布都是**玻尔兹曼分布**（Boltzmann distribution） 的一个实例。因此，我们把许多基于能量的模型称为**玻尔兹曼机** （Boltzmann Machine）。


### Separation & d-separation

图模型中的边告诉我们哪些变量直接相互作用，那么如何从中得知**间接关系**，或得知在**给定其他变量子集的值**时，哪些变量子集彼此**条件独立**。

#### 无向模型中，图中隐含的条件独立性称为分离（separation）

如果图结构显示给定变量集 S 的情况下变量集 A 与变量集 B 无关，那么在给定变量集 S 时，变量集 A 与另一组变量集 B 是分离的。

- 仅涉及未观察到的变量的路径是 ‘‘活跃’’ 的；
- 包括可观察变量的路径称为 ‘‘非活跃’’ 的；
- 如果两个变量之间的路径都是非活跃或者无路径，即为分离；否则，不分离。

![image-20200719160240612](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719160240612.png)

![image-20200719160620909](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719160620909.png)

#### 有向模型中，对应的概念称作 d-分离（d-separation）

“d’’ 代表 “依赖’’ 的意思。d-分离的概念和分离一样，但确定路径是否活跃有些复杂，详见下图。

![image-20200719161117095](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719161117095.png)

![image-20200719161134980](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719161134980.png)



### 有向与无向的转换

有向和无向模型各有优势，具体使用哪一个要根据具体任务分析。

有时，如果我们观察到变量的某个子集，或者如果我们希望执行不同的计算任务，换一种建模方式可能更合适。 例如，有向模型通常提供了一种高效地从模型中抽取样本（在第16.3节中描述）的 直接方法。而无向模型形式通常对于推导近似推断过程（我们将在第十九章中看到， 式(19.56)强调了无向模型的作用）是很有用的。

#### 完全图

一种极端的做法是，利用**“全连接的”完全图**将任何概率分布表示出来：

![image-20200719163653512](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719163653512.png)

#### 道德图：有向 $\rightarrow$ 无向

有向模型能够使用一种无向模型无法完美表示的特定类型的子结构。这个子结构被称为不道德（immorality）。这种结构出现在当两个随机变量 a 和 b 都是第三个随机变量 c 的父结点，并且不存在任一方向上直接连接 a 和 b 的边时。

为了将有向模型图 D 转换为无向模型，我们需要创建一个新图 U。对于每对变量 x 和 y，如果存在连接 D 中的 x 和 y 的有向边（在任一方向上），或者如果 x 和 y 都 是图 D 中另一个变量 z 的父节点，则在 U 中添加连接 x 和 y 的无向边。得到的图 U 被称为是**道德图**（moralized graph）。

![image-20200719164027556](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719164027556.png)

#### 弦图：无向 $\rightarrow$ 有向

如果 U 包含长度大于 3 的**环**（loop），则有向图 D 不能捕获无向模型 U 所包含的所有条件独立性，除非该环还包含**弦**（chord）。

> 环指的是由无向边连接的变量序列，并且满足序列中的最后一个变量连接回序列中的第一个变量。弦是定义环序列中任意两个 非连续变量之间的连接。

通过将弦添加到 U 形成的图被称为**弦图**（chordal graph）或者**三角形化图**（triangulated graph），因为现在可以用更小的、三角的环来描述所有的环。 

要从弦图构建有向图 D，我们还需要为边指定方向。当这样做时，我们不能在 D 中 创建有向循环，否则将无法定义有效的有向概率模型。为 D 中的边分配方向的一种方法是对随机变量排序，然后将每个边从排序较早的节点指向排序稍后的节点。

![image-20200719164332203](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719164332203.png)

### 因子图 factor graph

#### 作用

因子图旨在解决无向模型图表达的模糊性，高效地求各个变量的**边缘分布**。

**模糊性**：无向图中，我们使用团和因子来分解地表示全局概率分布。然而，我们无法确定每一个团是否含有一个作用域包含整个团的因子。用人话来说，就是下图 16.13 的示例，这样一个三节点无向图，可以由三个因子表示，也可以由一个因子表示，这是有歧义的。所以可以将无向图转换为因子图来清晰地实现图的解构，从而高效地求解各变量的边缘分布。 

#### 定义

将一个具有多变量的全局函数因子分解，得到几个局部函数的乘积，以此为基础得到的一个双向图叫做因子图（Factor Graph）。

对于函数  $g\left(X_{1}, \ldots, X_{n}\right)$，有以下式子成立：
$$
\begin{array}{l}
g\left(X_{1}, \ldots, X_{n}\right)=\prod_{j=1}^{m} f_{j}\left(S_{j}\right) \\
\text { 其中, } S_{j} \subseteq\left\{X_{1}, \ldots, X_{n}\right\}_{\circ}
\end{array}
$$
由以上，我们可以将因子图表示为三元组 $G=(X,F,E)$：

- $X=\left\{X_{1}, \ldots, X_{n}\right\}$ 表示变量结点（variable vertices）
- $F=\left\{f_{1}, \ldots, f_{m}\right\}$ 表示因子结点（factor vertices）
- $E$ 为边的集合，如果某一个变量结点 $X_k$ 被因子结点 $f_j$ 的集合 $S_j$ 包含，那么就可以在 $X_k$ 和 $f_j$ 之间加入一条无向边。

#### 示例

- **圆形节点**对应于原来无向图的**随机变量**；
- **方块节点**对应于未归一化概率函数的**因子** $\phi$。

- 当且仅当变量包含在某个因子中时，变量和因子在图中才会存在连接。

- 因子与因子、变量与变量之间没有连接。

因此，因子图是一种因式分解一样的、清晰的图解构方法。

![image-20200719164745855](D:\Github\Reinforcement-Learning-in-Robotics\Graph\PR Structure Ⅱ .assets\image-20200719164745855.png)

## 从图模型中采样

### 有向图的原始采样

有向图模型的一个优点是，可以通过一个简单高效的过程从模型所表示的联合分布中产生样本，这个过程被称为**原始采样**（Ancestral Sampling）。

- 将图中的变量 xi 使用拓扑排序，使得对于所有 i 和 j，如 果 xi 是 xj 的一个父亲结点，则 j 大于 i；
- 首先采 $\mathrm{x}_{1} \sim P\left(\mathrm{x}_{1}\right)$；
- 然后采 $\mathrm{x}_{2} \sim P\left(\mathrm{x}_{2} \mid P a_{\mathcal{G}}\left(\mathrm{x}_{2}\right)\right)$；
- 直到采样 $P\left(\mathrm{x}_{n} \mid P a_{\mathcal{G}}\left(\mathrm{x}_{n}\right)\right)$。



### 无向图的Gibbs采样

原始采样仅适用于有向图，将无向图装化成有向图再抽样的做法会遇到一些棘手的问题：

- 要确定新有 向图的根节点上的边缘分布；
- 者需要引入许多边从而会使得到的有向模型变得难以处理。

直接从无向图采样，需要解决**循环依赖**的问题，采样没有明确的起点。目前的做法是，利用Gibbs采样等方法迭代方位每个变量，在给定其他变量条件下抽样。这个我们在 PR Sampling 分章中有详细的讨论。



## 学习依赖关系 / 图结构

当模型旨在描述直接连接的可见变量之间的依赖关系时，通常不可能连接所有变量，因此设计图模型时需要连接那些紧密相关的变量，并忽略其他变量之间的作用。

### 自适应结构

可以通过**结构学习**技术（可能启发了**神经架构搜索NAS**）找到简洁的图连接方式。

- 提出一种结构，对具有该结构的模型进行训练，然后给出分数；
- 该分数奖励训练集上的高精度并对模型的复杂度进行惩罚；
- 然后提出添加或移除少量边的候选结构作为搜索的下一步；
- 搜索向一个预计会增加分数的新结构发展。 

### 使用潜变量

在深度学习中，最常用于建模这些依赖关系的方法是引入几个**潜变量 h**。可见变量和潜变量之间的固定结构可以用于建模可见单元之间的间接作用。使用简单的参数学习技术，我们可以学习到一个具有固定结构的模型，这个模型在边缘分布 p(v) 上拥有正确的结构。 

推断潜变量的常用方法：
1、隐马尔可夫模型；
2、因子分析；
3、主成分分析；
4、偏最小二乘回归；
5、潜在语义分析和概率潜在语义分析；
6、EM算法。




## 推断和近似推断

解决变量之间如何相互关联的问题是我们使用概率模型的一个主要方式。**推断可以理解为，在给定其他变量的情况下预测一些变量的值或概率分布**。

图结构允许我们用合理数量的参数来表示复杂的高维分布，但是用于深度学习的图经常是 NP-hard 的，从而难以实现高效地推断。 这促使我们使用**近似推断**。在深度学习中，通常涉及变分推断，**通过寻求尽可能接近真实分布的近似分布 q(h|v) 来逼近真实分布 p(h|v)。**



## 与深度学习的结合

**深度学习模型**：

- 通常具有**比可观察变量更多的潜变量**（也就是神经网络的隐层节点）；
- 通过多个潜变量的间接连接来实现变量之间复杂的非线性相互作用；
- 潜变量不具有特定的先验含义；
- 经常是全连接，使得两个组之间的相互作用可以由单个矩阵描述；

**传统的图模型**：

- 大多使用高阶项和结构学习来捕获变量之间复杂的非线性相互作用；
- 有潜变量的话，数量也通常很少，且被赋予一些特定含义；
- 具有非常少的连接，并且每个变量的连接选择可以单独设计；

### 图模型和深度学习的结合

受限玻尔兹曼机（Restricted Boltzmann Machine, RBM）

深度信念网络（deep belief network, DBN）

深度玻尔兹曼机（Deep Boltzmann Machine, DBM）

卷积玻尔兹曼机

用于结构化或序列输出的玻尔兹曼机

变分自编码器（variational auto-encoder, VAE）

生成式对抗网络（generative adversarial network, GAN）



## 扩展阅读

1. [Restricted Boltzmann Machine with python Implementation](http://deeplearning.net/tutorial/rbm.html)
2. [Deep Belief Network with python Implementation](http://deeplearning.net/tutorial/DBN.html)
3. [因子图](https://mqshen.gitbooks.io/prml/Chapter8/inference/factor_graph.html)
4. [因子图的 sum-product algorithm](https://mqshen.gitbooks.io/prml/Chapter8/inference/sum_product_algorithm.html)
5. [Latent Variables & Expectation Maximization Algorithm](https://towardsdatascience.com/latent-variables-expectation-maximization-algorithm-fb15c4e0f32c)
6. [因子图介绍](https://longaspire.github.io/blog/%E5%9B%A0%E5%AD%90%E5%9B%BE%E4%BB%8B%E7%BB%8D/#4-%E5%9B%A0%E5%AD%90%E5%9B%BE%E7%9A%84%E8%BD%AC%E6%8D%A2)
7. [概率图模型总览](https://longaspire.github.io/blog/%E6%A6%82%E7%8E%87%E5%9B%BE%E6%A8%A1%E5%9E%8B%E6%80%BB%E8%A7%88/)
8. [近似推断](https://mqshen.gitbooks.io/prml/Chapter10/)

