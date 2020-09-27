#! https://zhuanlan.zhihu.com/p/259883878
![Image](https://pic4.zhimg.com/80/v2-d548359bb50d49f2b3421f3fcb044970.jpg)

# PR Structured Ⅳ：General / Graph Conditional Random Field (CRF) 及其 python 实现

![Content](https://pic4.zhimg.com/80/v2-6f40827660d1a54b55a6b634b7dfbeb6.png)

本章承接上一章 

[【归纳综述】：马尔可夫、隐马尔可夫 HMM 、条件随机场 CRF 全解析及其python实现](https://zhuanlan.zhihu.com/p/259660645 'card')

**将其中的 linear CRF 推广到 graph CRF 的形式**

CRF 本身就是概率图模型的一种，只是通常我们处理序列数据的时候遇到的都是观测与隐状态一一对应的情况，即为其简化版 linear CRF。

如果想研究更复杂的 CRF 形式，推荐大家看一下 *Charles Sutton* 的综述：
[An Introduction to Conditional Random Fields](https://arxiv.org/pdf/1011.4088.pdf)

## 1 Generative versus Discriminative Models



我们还是从下图讲起

![白色节点为隐状态 y，灰色节点为观测 x](https://pic4.zhimg.com/80/v2-4c4c4f36b37f887a303cfa70fa9b56bb.png)

上图第一行为生成式模型，第二行为判别式模型。主要区别在于条件分布 $p(y | x)$ 不包括 $p(x)$ 的模型。

对 $p(x)$ 进行建模的困难在于，它通常包含许多难以建模的、高度相关的特征。

> **例**：在 *named-entity recognition (NER) task (China - location; Bush - people)* 中，HMM仅依赖一个特征，即单词的身份。但是，在训练集中不会出现很多单词，尤其是专有名词，因此单词的身份特征并不能提供足够的信息。为了标记未见的单词，我们想利用单词的其他特征，例如大写，相邻单词，前缀和后缀，在预定的人员和位置列表中的成员资格等等。

判别建模的主要优点是**它更适合包含丰富的交叉特征**。通过直接对条件分布建模，我们可以不了解 $p(x)$的形式。 

CRF 在 $y$ 之间进行独立性假设，并且对 $y$ 如何依赖 $x$ 进行假设，但在 $x$ 之间不进行假设。

这一点也可以用 graph 来理解：假设我们有一个关于联合分布 $P(Y,X)$ 的因子图表示。如果我们再为条件分布 $P(Y|X)$ 构造一个图，则仅依赖x的任何因子都会从条件分布的图形结构中消失。它们与条件无关，因为它们相对于 $y$ 是常数。

![Image](https://pic4.zhimg.com/80/v2-29ce59f2d4a6026ff95297045c035575.png)

为了在生成模型中得到更多的信息，一种可能的方法就是，如朴素贝叶斯一样包含大量依赖特征，但在其中的独立性假设，还是会带来问题，损害性能。

例如，尽管朴素贝叶斯分类器在文档分类中表现良好，但与Logistic回归相比，它在一系列应用中的平均表现较差。

## 2 General / Graph Conditional Random Field (CRF)

从 linear-chain CRF 可以直接推广到 general CRF。只是简单地将**线性链因子图**推广到**更通用的因子图**，然后将**前向后向算法**换为**更通用的（近似）推理算法**。

**Definition**

令 $G$ 为 $Y$ 上的因子图。如果对于任何固定的$X$，分布 $P(Y|X)$ 都根据 $G$ 求得，则 $P(Y|X)$ 为其条件随机场。

> 听起来像是啥也没说，可是有的时候定义就这样。

如果$F = {Ψ_a}$是 $G$ 中的一组因子，并且每个因子采用下式指数形式

$$
\Psi_{a}\left(\mathbf{x}_{a}, \mathbf{y}_{a}\right)=\exp \left\{\sum_{k} \theta_{a k} f_{a k}\left(\mathbf{x}_{a}, \mathbf{y}_{a}\right)\right\}\\
$$

则条件分布可以写为

$$
p(\mathbf{Y} \mid \mathbf{X})=\frac{1}{Z(\mathbf{x})} \prod_{\Psi_{A} \in G} \exp \left\{\sum_{k=1}^{K(A)} \theta_{a k} f_{a k}\left(\mathbf{y}_{a}, \mathbf{x}_{a}\right)\right\}\\
$$

## 3 General CRF 概率推断

在 linear-chain CRF 中，我们使用的前向后向算法。而 general CRF 可以使用 inference 方法。

对于 general CRF 是可以用**精确推理算法**的。
最流行的精确算法是**连接树算法 (the junction tree algorithm)**，它会依次对变量进行聚类，直到图成为一棵树。
一旦构建了等效树，就可以使用特定于树的精确推理算法来计算其边际。但是，对于某些复杂的图，必须使用连接树算法生成非常大的聚类，这就是为什么在最坏的情况下该过程仍需要指数时间的原因。

由于这个原因，我们可以使用**近似推理算法**。
最受关注的是两类近似推理算法：**蒙特卡洛算法和变分算法**。
- **蒙特卡洛算法**是随机算法，试图从目标分布中近似产生一个样本。
- **变分算法**是通过尝试找到最接近感兴趣的分布的简单分布，将推理问题转换为优化问题的算法。

通常，蒙特卡洛算法在保证有足够的计算时间的情况下可以保证从感兴趣的分布中进行采样，因此是无偏的，尽管实际上通常不知道何时到达该点。
另一方面，变分算法可以快得多，但是它们往往会产生偏差，这就是说，它们意味着趋向于具有近似所固有的误差源，并且不能通过给它们提供更多的计算来轻松地减少误差。
时间。
尽管如此，变分算法对于CRF还是有用的，因为参数估计需要执行多次推理，因此快速的推理过程对于有效的训练至关重要。

### 3.1 Markov Chain Monte Carlo (MCMC)
用于复杂模型的最流行的蒙特卡洛方法类型是**马尔可夫链蒙特卡洛（MCMC）**。 

这部分内容详见 *PR Sampling* 分章

[PR Sampling Ⅰ：蒙特卡洛采样、重要性采样及python实现](https://zhuanlan.zhihu.com/p/150693309 'card')

[PR Sampling Ⅱ：马尔可夫链蒙特卡洛 MCMC及python实现](https://zhuanlan.zhihu.com/p/150742395 'card')

[PR Sampling Ⅲ：M-H and Gibbs 采样](https://zhuanlan.zhihu.com/p/150946559 'card')

MCMC方法不是尝试直接近似边缘分布 $p(y_s|\mathbf x)$，而是根据联合分布 $p(\mathbf y|x)$ 生成近似样本。 
MCMC方法通过精心构造状态空间与 $Y$ 相同的马尔可夫链来实现，以便在长时间模拟链时，链上状态的分布大约为 $p(y_S|\mathbf x)$。 
假设我们要近似某个函数 $f(x,y)$ 的期望值。
给定MCMC方法中马尔可夫链上的样本 $\mathbf y^1, \mathbf y^2, \dots, \mathbf y^M$，我们可以将该期望近似为：

$$ \sum_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x}) f(\mathbf{x}, \mathbf{y}) \approx \frac{1}{M} \sum_{j=1}^{M} f\left(\mathbf{x}, \mathbf{y}^{j}\right)\\ $$

在CRF中，这些近似期望值可用于近似学习所需的量，特别是梯度。

MCMC方法的一个简单示例是Gibbs采样。在Gibbs采样算法的每次迭代中，每个变量都会单独进行重新采样，同时保持所有其他变量不变。
假设我们已经有来自迭代 $j$ 的样本 $\mathbf{y}^{j}$。
然后生成下一个样本  $\mathbf{y}^{j+1}$

- $\mathbf{y}^{j+1} \leftarrow \mathbf{y}^{j}$
- 对每一个重采样component $s\in V$，从 $p\left(y_{s} \mid \mathbf{y}_{\backslash s}, \mathbf{x}\right)$
- 返回结果 $\mathbf{y}^{j+1}$

该过程定义了一个马尔可夫链，可用于近似上一个公式中的期望。

对于general CRF，，可以将这种条件概率计算为

$$ p\left(y_{s} \mid \mathbf{y}_{\backslash s}, \mathbf{x}\right)=\kappa \prod_{C_{p} \in \mathcal{C}} \prod_{\Psi_{c} \in C_{p}} \Psi_{c}\left(\mathbf{x}_{c}, \mathbf{y}_{c} ; \theta_{p}\right)\\ $$

其中 $κ$ 是归一化常数。

这比联合概率 $p(y|x)$ 更容易计算，因为计算 $κ$ 仅需要对 $y_s$ 的所有可能值求和，而不是对整个向量 $\mathbf{y}$ 进行赋值。

Gibbs采样**优点**：易于实现。诸如BUGS之类的软件包可以将图形模型作为输入，并自动编译适当的Gibbs样本。 

Gibbs采样**缺点**：如果 $p(y|x)$ 具有很强的依赖性，则它可能无法正常工作，这在顺序数据中通常是这种情况。马尔可夫链上的样本分布接近所需分布 $p(y|x)$ 之前可能需要进行多次迭代。

### 3.2 Belief Propagation (BP) 信念传播

假设 $G$ 是一棵树，我们希望计算变量 $s$ 的边际分布。 
BP的直觉是 $s$ 的每个相邻因子都对 $s$ 的边缘产生了乘法贡献，称为**消息 *message***。由于该图是一棵树，因此可以分别计算每一个消息，即对于每个因子 $a\in N(s)$，将 $V_a$ 称为 $a$ 的**上游变量集**，即 $a$ 介于 $s$ 和 $v$ 之间的变量 $v$ 集合。 
$F_a$ 是位于 $a$ 上游的一组因子，包括 $a$ 自身。
但是现在由于G是一棵树，集合 $\left\{V_{a}\right\} \cup\{s\}$ 形成了 $G$ 中变量的分区。这意味着我们可以将边际所需的求和分解为独立子问题的乘积，如 (3.2.1)：

$$ \begin{aligned}
p\left(y_{s}\right) & \propto \sum_{\mathbf{y} \backslash y_{s}} \prod_{a} \Psi_{a}\left(\mathbf{y}_{a}\right) \\
&=\prod_{a \in N(s)} \sum_{\mathbf{y}_{V_{a}}} \prod_{\Psi_{b} \in F_{a}} \Psi_{b}\left(\mathbf{y}_{b}\right)
\end{aligned}\\ $$

用 $m_{as}$ 表示上述方程式中的每个因子，即 (3.2.2)

$$ m_{a s}\left(x_{s}\right)=\sum_{\mathbf{y}_{V_{a}}} \prod_{\Psi_{b} \in F_{a}} \Psi_{b}\left(\mathbf{y}_{b}\right)\\ $$

可以认为是从因子a到变量s的消息，它概括了a上游网络对s信念的影响。
以类似的方式，我们可以将变量到因子的消息定义为 (3.2.3)

$$ m_{s A}\left(x_{s}\right)=\sum_{\mathbf{y}_{V_{s}}} \prod_{\Psi_{b} \in F_{s}} \Psi_{b}\left(\mathbf{y}_{b}\right)\\ $$

然后，从上上上（吐槽一下知乎的公式编辑器）式中，我们得到边际 $p(y_s)$ 与变量s所有传入消息的乘积成正比。
同样，因子边际可以计算为 (3.2.4)

$$ p\left(\mathbf{y}_{a}\right) \propto \Psi_{a}\left(\mathbf{y}_{a}\right) \prod_{s \in a} m_{s a}\left(\mathbf{y}_{a}\right)\\ $$

在这里，我们将变量a视为集合，将其表示为因子 $Ψ_a$ 的范围。
另外，有时我们会使用反符号 $c \ni s$ 表示包含变量s的所有因子c的集合。

根据（3.2.2）计算消息是不切实际的，因为如我们所定义的消息需要对图中的许多变量求和。
幸运的是，我们可以使用**局部求和的递归**来改写消息。递归为 (3.2.5)

$$ \begin{array}{rl}
m_{a s}\left(x_{s}\right)=\sum_{\mathbf{y}_{a} \backslash y_{s}} \Psi_{a}\left(\mathbf{y}_{a}\right) \prod_{t \in a \backslash s} & m_{t a}\left(x_{t}\right) \\
m_{s a}\left(x_{s}\right)=\prod_{b \in N(s) \backslash a} m_{b s}\left(x_{s}\right)
\end{array}\\ $$

通过重复替换可以看出该递归与 $m$ 的定义相匹配，并且可以通过归纳证明。
在树中，可以安排这些递归，以使先行消息始终在其依赖项之前发送，方法是先从根节点发送消息，依此类推。这就是称为信念传播的算法。

除了计算单变量边际，我们还将希望针对给定的分配 $\mathbf y$ 计算因子边际 $p(\mathbf y_a)$ 和联合概率 $p(\mathbf y)$ 。 

![Image](https://pic4.zhimg.com/80/v2-e8cce99018f6d1e3a014da1765d292e3.png)

首先，要计算因子（或实际上是任何相连的变量集）的边际，我们可以使用与单变量情况相同的边际分解，得到

$$ p\left(\mathbf{y}_{a}\right)=\kappa \Psi_{a}\left(\mathbf{y}_{a}\right) \prod_{s \in a} m_{s a}\left(y_{s}\right)\\ $$

其中 $κ$ 是归一化常数。
实际上，类似的想法适用于任何连接的变量集-不仅仅是碰巧是某个因素的域的变量-尽管如果变量集太大，则计算κ是不切实际的。

BP也可以用于计算归一化常数 $Z(x)$ 。
这可以直接从传播算法中完成，类​​似于 linear CRF 的前向后向算法。

## 4 General CRF 参数学习

## 5 Python 实现
对于 graph CRF，我能找到的 python 库就是 `PyStruct` 了，这个库在CRF族算法中已经算是很全面的了。 [文档链接](https://pystruct.github.io/)

PyStruct 是将模型和求解器分开定义，与之前的 `sklearn-crfsuite` 库不太一样。

### 5.1 General CRF 参数学习

```python
from pystruct.models import GraphCRF
from pystruct.learners import FrankWolfeSSVM
model = GraphCRF(directed=False, inference_method="max-product")
ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
ssvm.fit(X_train, y_train) 
# Output: FrankWolfeSSVM(C=0.1, batch_mode=False, check_dual_every=10,
#            do_averaging=True, line_search=True, logger=None, max_iter=10,
#            model=GraphCRF(n_states: 26, inference_method: max-product),
#            n_jobs=1, random_state=None, sample_method='perm',
#            show_loss_every=0, tol=0.001, verbose=0)
```

### 5.2 General CRF 概率推断

`models.GraphCRF` 类中的 `inference_method` 属性有以下几种选项：

- default=None
- `'max-product'` for **max-product belief propagation**.
Recommended for chains an trees. Loopy belief propagatin in case of a general graph.
- `'lp'` for **Linear Programming relaxation** using cvxopt.
- `'ad3'` for **AD3 dual decomposition**.
- `'qpbo'` for **QPBO + alpha expansion**.
- `'ogm'` for **OpenGM inference algorithms**.

在初始化 `models.GraphCRF` 的时候，设定 `inference_method`，在训练完模型参数之后，就可以调用 `inference` 方法进行推断

> `inference(x, w, relaxed=False, return_energy=False)`
>
> Inference for x using parameters w.

```python
observed_xseq = ...
w = ssvm.get_params()
model.inference(observed_xseq, w) 
# Output: y_pred (ndarray or tuple) 
```

