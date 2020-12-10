#! https://zhuanlan.zhihu.com/p/276873641
![](https://pic4.zhimg.com/80/v2-2a9449949ebc3864f5d428641083f173.jpg)

# PR Structured Ⅴ：GraphRNN——将图生成问题转化为序列生成
![](https://pic4.zhimg.com/80/v2-4eff79c22960304a1b3f37c0c0ac5bfe.png)
[Paper](https://arxiv.org/pdf/1802.08773.pdf) | [Code](https://github.com/JiaxuanYou/graph-generation)

本文一作实在是太大佬了，让我和小伙伴焦虑了好一阵子。[作者主页](https://cs.stanford.edu/people/jiaxuan/)送你们，将这份焦虑传递下去。

![](https://pic4.zhimg.com/80/v2-46fb6ebf9a94aac54f59d0b4cef870eb.png)
## Introduction
图生成有很多**用处**：
1. 建模physical and social interactions
2. 发现新的化学和分子结构
3. 构建知识图谱

本文摘要直接指出了图生成问题的**难点**：

> 图生成模型需要学习到图的结构分布，然而图具有**非唯一 (non-unique)**，**高维**以及**给定图的边之间存在复杂、非局部的依存关系**。

因此直接对复杂的图分布直接进行建模，并从这些分布中进行有效采样是一项挑战。

目前，图生成面临的**挑战**有：
1. 要让模型在没有图结构假设的情况下，从一组观察到的图中直接学习生成模型；
2. 具备从多个图以及大图中学习生成的能力；
3. Large and variable output spaces：$n$ 个节点的图需要输出$n^2$的值才能完整表示，且每个图的边和节点也不是固定的数值；
4. Non-unique representations：如果我们想学习一个有 $n$ 个节点的图的结构，然而它最多可以表示为 $n!$ 个等效邻接矩阵，这会让训练变得很难。
5. Complex dependencies：很明显，图中的边关系不可能简单地看作相互独立的，它们之间有复杂的依存关系。

## GraphRNN
为解决上述问题，本文提出了 GraphRNN，以 **autoregressive (or recurrent)** 作为一系列新节点和边的添加方式绘制 graph，来捕获图中**所有节点和边的复杂联合概率**。

GraphRNN 可以视作一种级联形式，由两个RNN组成：
1. graph-level RNN：维护图的状态并生成新节点；
2. edge-level RNN：为新生成的节点生成新的边。

### 符号定义
|                             符号                             |                             含义                             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          $G=(V,E)$                           |                            无向图                            |
|                    $V = {v_1, ..., v_n}$                     |                            节点集                            |
| $E=\left\{\left(v_{i}, v_{j}\right) \mid v_{i}, v_{j} \in V\right\}$ |                             边集                             |
|                            $\pi$                             | 节点排序 node ordering<br />$\left(\pi\left(v_{1}\right), \ldots, \pi\left(v_{n}\right)\right)$是节点 $\left(v_{1}, \ldots, v_{n}\right)$ 的排序 |
|                             $Π$                              |                    $n!$ 个可能的排序方式                     |
| $A^{\pi} \in \mathbb{R}^{n \times n}, A_{i, j}^{\pi}=\mathbb{1}\left[\left(\pi\left(v_{i}\right), \pi\left(v_{j}\right)\right) \in E\right]$ |        以 $\pi$ 的排序方式从图 $G$ 转化而来的邻接矩阵        |
|                      $A^Π = {A^π ∈ Π}$                       |                     可能的邻接矩阵的合集                     |
|                        $p_{model}(G)$                        | 这就是图生成要从训练集 $\mathbb{G}=\left\{G_{1}, \ldots, G_{s}\right\}$ 中学习的图的分布 |

![](https://pic4.zhimg.com/80/v2-1bbb85f303ea8784cc53eef466a5c357.png)

## GraphRNN 思路

![](https://pic4.zhimg.com/80/v2-af5874e3e3ebd4ef511c7cbaf6180643.png)

**Key Idea**： **将不同节点顺序下的图表示为序列，并在这些序列上构建一个自回归的生成模型。**

### 将graph建模成序列

定义从graphs到sequences的映射 $f_s$：
$$
S^{\pi}=f_{S}(G, \pi)=\left(S_{1}^{\pi}, \ldots, S_{n}^{\pi}\right)\\
$$
其中，每个元素 $S_{i}^{\pi} \in\{0,1\}^{i-1}, i \in\{1, \ldots, n\}$ 表示节点$π(v_i)$和先前所有节点$π(v_j), j\in\{1,\dots, i-1\}$之间的边的邻接向量 $S_{i}^{\pi}=\left(A_{1, i}^{\pi}, \ldots, A_{i-1, i}^{\pi}\right)^{T}, \forall i \in\{2, \ldots, n\}$（如图一，注意下面的向量依次对应右上方的graph）。

由此，$S^\pi$ 的序列就可以表示整个无向图，这里用一个反向的映射表示
$$
f_{G}\left(S^{\pi}\right)=G\\
$$
在此基础上，对于图分布 $p(G)$ 的学习就可以转化为联合分布 $p(G,S^\pi)$ 的边缘分布：
$$
p(G)=\sum_{S^\pi} p\left(S^{\pi}\right) \mathbb{1}\left[f_{G}\left(S^{\pi}\right)=G\right]\\
$$
这时，我们只需要学习 $p(S^\pi)$ 就可以了。由于它又是个序列模型，所以可以分解为条件分布的乘积：
$$
p\left(S^{\pi}\right)=\prod_{i=1}^{n+1} p\left(S_{i}^{\pi} \mid S_{1}^{\pi}, \ldots, S_{i-1}^{\pi}\right)=\prod_{i=1}^{n+1} p\left(S_{i}^{\pi} \mid S_{<i}^{\pi}\right)\\
$$
定义最后一个元素 $S^\pi_{n+1}$ 为序列终止 EOS。

### GraphRNN 框架

即使 $p(G)$ 被分解成了 $p\left(S_{i}^{\pi} \mid S_{<i}^{\pi}\right)$，这仍然很棘手。因为它需要在之前的节点的连接基础上，得到节点 $\pi(v_i)$ 如何与之前的节点连接。这又是一个复杂的概率关系，本文打算用RNN来建模这种关系，包含**状态转移函数**和**输出函数**：
$$
\begin{aligned}
h_{i} &=f_{\text {trans }}\left(h_{i-1}, S_{i-1}^{\pi}\right) \\
\theta_{i} &=f_{\text {out }}\left(h_{i}\right)
\end{aligned}\\
$$

- $h_i\in R^d$ 是一个编码了到目前为止生成的图的状态的向量；
- $S^\pi_{i-1}$ 是最近一个生成节点的邻接向量；
- $\theta_i$ 指定了下一个节点邻接向量的分布 $S_{i}^{\pi} \sim \mathcal{P}_{\theta_{i}}$。

文中指出，$f_{trans}, f_{out}$ 可以用任意神经网络表示（本文开源代码用了**两个GRU**）。$\mathcal{P}_{\theta_{i}}$ 也可以是任意形态分布。

算法总结为：
![](https://pic4.zhimg.com/80/v2-d2c346a7df204b4afc8e9b84540c4f35.png)

## 利用 BFS 处理变长度的序列
由于RNNs需要固定长度的输入向量，然而$S_i^\pi$的长度是随着 i 变化的，因此本文旨在**利用 BFS （广度优先搜索） 的节点序列，而不是任意节点序列，来学习图的生成**。这样做的好处，据说是不是一般性？

将式1改为
$$
S^{\pi}=f_{S}(G, \operatorname{BFS}(G, \pi))\\
$$
BFS 以一个随机顺序 $\pi$ 为输入，将 $\pi(v_1)$ 作为起点，按照 $\pi$ 中的先后顺序将它的邻居依次添加到 BFS 队列中。

好处如下：
- BFS 是一对多的，一个 BFS 序列可以转化为多个节点排序。因此我们需要训练的数量少了。
- BFS排序通过减少 edge-level RNN 中进行的边缘预测的数量，来使学习变得更容易。因为如果我们新加入一个节点，那么它的连接边只能处在BFS搜索前沿的节点（当搜索完成时，可以想象成树的叶子节点）定义描述就是：

  Proposition 1. Suppose $v_{1}, \ldots, v_{n}$ is a BFS ordering of $n$ nodes in graph $G$, and $\left(v_{i}, v_{j-1}\right) \in E$ but $\left(v_{i}, v_{j}\right) \notin E$ for some $i<j \leq n,$ then $\left(v_{i^{\prime}}, v_{j^{\prime}}\right) \notin E, \forall 1 \leq i^{\prime} \leq i$ and $j \leq j^{\prime}<n$

  这个性质是我们可以**将可变长度的 $S^\pi_i$ 定义为固定长度的 M 维向量**，表示节点 $\pi(v_i)$ 与当前BFS队列中最大大小为M的节点之间的连通性：
  $$
  S_{i}^{\pi}=\left(A_{\max (1, i-M), i}^{\pi}, \ldots, A_{i-1, i}^{\pi}\right)^{T}, i \in\{2, \ldots, n\}\\
  $$
  至于这个 M 怎么去估计，见本文附录吧。

## 扩展到具有节点、边特征的Graph

GraphRNN可以扩展到具有节点和边特征的Graph生成，在节点顺序 $π$ 下，图 $G$ 与它的节点特征矩阵 $X^{\pi} \in \mathbb{R}^{n \times m}$ 和边特征矩阵 $F^{\pi} \in \mathbb{R}^{n \times k}$ 相关联。因此，可以将 $S^\pi$ 的定义扩展为 $S_{i}^{\pi}=\left(X_{i}^{\pi}, F_{i}^{\pi}\right)$。在 $f_{out}$ 模块，用一个 MLP 来生成 $X^\pi_i$，edge-level RNN 来生成 $F^\pi_i$。

作者的开源代码好像并没有这部分，我已经在github上发了issues。