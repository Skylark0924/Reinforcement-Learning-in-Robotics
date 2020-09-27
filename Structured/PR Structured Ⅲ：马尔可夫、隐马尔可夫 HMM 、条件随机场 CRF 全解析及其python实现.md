#! https://zhuanlan.zhihu.com/p/259660645
![Image](https://pic4.zhimg.com/80/v2-c0a76aeb48e699c638bf0215fbc8395e.jpg)
# 【归纳综述】马尔可夫、隐马尔可夫 HMM 、条件随机场 CRF 全解析及其python实现

## PR Structured Ⅲ：马尔可夫、隐马尔可夫 HMM 、条件随机场 CRF 全解析及其python实现

![Content](https://pic4.zhimg.com/80/v2-b83a33f624f7ddcbf097d7e66445e994.png)


**归纳性长文，不断更新中...欢迎关注收藏**

本章承接概率图知识

[PR Structured Ⅱ：Structured Probabilistic Model An Introduction](https://zhuanlan.zhihu.com/p/161703636 'card')

马尔可夫不仅是强化学习在时序决策上的理论基础，也是语音、NLP等领域处理时序数据并进行预测的基础。

> **在使用这一族方法的时候，我们的目的是什么？**
> 
> 与普通的回归和分类不同，时序数据相邻数据间是有关系的，而非相互独立的。因此我们可以用上下文（主要是上文）所提供的信息，更好地对这类数据做分类或回归。



本章分以下三个递进的环节：

1. 将之前所有的观测作为未来预测的依据是**不现实的**，因为其复杂度会随着观测数量的增加而无限制地增长。因此，就有了马尔科夫模型，即假定**未来的预测仅与最近的观测有关，而独立于其他所有的观测**；
2. 通过引入**隐变量**，解决Markov Model需要强独立性的问题，即隐马尔可夫模型  HMM；
3. **HMM等为生成式模型，计算联合概率分布 $P(X, Z)$；CRF则是判别式模型，计算条件概率 $P(Y|X)$。由于 CRF 利用最大熵模型的思路建立条件概率模型，对于观测序列并没有做马尔科夫假设，可以得到全局最优，而HMM则是在马尔科夫假设下建立的联合分布，会出现局部最优的情况**。（此处 $Y,Z$ 均代表隐变量，$X$ 为观测）

## 1 马尔可夫 Markov

> **引例**：假设我们观测⼀个⼆值变量，这个⼆值变量表⽰某⼀天是否下⾬。给定这个变量的⼀系列观测，我们希望预测下⼀天是否会下⾬。
>
> - 如果我们将所有的数据都看成独⽴同分布的， 那么我们能够从数据中得到的唯⼀的信息就是⾬天的相对频率；
> - 然而，我们知道天⽓经常会呈现出持续若⼲天的趋势。因此，观测到今天是否下⾬对于预测明天是否下⾬会 有极⼤的帮助。

我们可以使用概率的乘积规则来表示观测序列的联合概率分布，形式为
$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right)=p\left(\boldsymbol{x}_{1}\right) \prod_{n=2}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n-1}\right)\\
$$
利用马尔科夫性，可以将上式变为**⼀阶马尔科夫链（first-order Markov chain）**：
$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right)=p\left(\boldsymbol{x}_{1}\right) \sum_{n=2}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-1}\right)\\
$$

![Image](https://pic4.zhimg.com/80/v2-a9c6c3d3163c8942637bf02c956a80df.png)

根据我们在概率图模型中讲的 **d-分离** 性质，观测 $x_n$ 的条件概率分布为
$$
p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n-1}\right)=p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-1}\right)\\
$$
当然了，还可以有**⼆阶马尔科夫链**，其中特定的观测$x_n$依赖于前两次观测$x_{n−1}$和$x_{n−2}$的值：

![Image](https://pic4.zhimg.com/80/v2-79166f7fdd4e1bbfd6c42ecc75c7ec17.png)

$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}\right)=p\left(\boldsymbol{x}_{1}\right) p\left(\boldsymbol{x}_{2} \mid \boldsymbol{x}_{1}\right) \prod_{n=3}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-1}, \boldsymbol{x}_{n-2}\right)\\
$$
还有**M阶马尔可夫链**，当然计算代价也很大：

>  如果有K个状态的离散变量，
>
> - ⼀阶马尔可夫链 $p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-1}\right)$ 需要 $K(K-1)$ 个参数；
> - 而M阶马尔可夫 $p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-M}, \ldots, \boldsymbol{x}_{n-1}\right)$ 需要 $K^M(K-1)$ 个参数。

### 1.1 连续变量

#### **1.1.1 Autoregressive (AR)**

使⽤线性⾼斯条件概率分布， 每个结点都是⼀个⾼斯概率分布，均值是⽗结点的⼀个线性函数。

#### **1.1.2 Tapped delay line**

使用神经网络等参数化模型拟合 $p\left(\boldsymbol{x}_{n} \mid \boldsymbol{x}_{n-M}, \ldots, \boldsymbol{x}_{n-1}\right)$ ，因为它对应于存储（延迟）观测变量的前⾯M个值来预测下⼀个值。


## 2 状态空间模型 state space model

问题来了，我们既想构造任意阶数的、不受马尔可夫假设限制的序列模型，同时能够使⽤较少数量的⾃由参数确定。怎么做呢？**引入隐变量**。

对于每 个观测$x_n$，我们引⼊⼀个对应的潜在变量$z_n$（类型或维度可能与观测变量不同）。我们现在假设潜在变量构成了马尔科夫链，得到的图结构被称为状态空间模型（state space model）。

![Image](https://pic4.zhimg.com/80/v2-778f85bd8051dc60fdc0a0b8133a31af.png)

它满⾜下⾯的关键的条件独⽴性质，即给定 $z_n$ 的条件下，$z_{n-1}$ 和 $z_{n+1}$ 是独⽴的，从而
$$
\boldsymbol{z}_{n+1} \perp \boldsymbol{z}_{n-1} \mid \boldsymbol{z}_{n}\\
$$
联合概率分布为
$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}, \boldsymbol{z}_{1}, \ldots, \boldsymbol{z}_{N}\right)=p\left(\boldsymbol{z}_{1}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}\right)\right] \prod_{n=1}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{z}_{n}\right)\\
$$
HMM 这类引入隐变量的模型好处在于：**根据d-分离准则，总存在⼀个通过隐变量连接任意两个观测变量 $x_n$ 和 $x_m$ 的路径，且这个路径永远不会被阻隔**。这也就间接地解决了马尔可夫模型中需要直接连接多个之前观测的问题，**相当于将 $x_n$ 之前的观测信息全部包含在对应的隐变量 $z_n$ 中。**

### 2.1 隐马尔可夫 HMM

> 隐变量：离散
> 
> 观测变量：离散或连续

> **引例**：以句子和词性对应为例，观测变量 $X=\{x_1,x_2,\dots, x_n\}$ 为显式的句子，隐变量 $Z=\{z_1, z_2,\dots,z_n\}$ 为每一个单词对应的词性。

 观测变量和隐变量上的联合概率分布为：
$$
p(\boldsymbol{X}, \boldsymbol{Z} \mid \lambda)=p\left(\boldsymbol{z}_{1} \mid \boldsymbol{\pi}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}, \boldsymbol{A}\right)\right] \prod_{m=1}^{N} p\left(\boldsymbol{x}_{m} \mid \boldsymbol{z}_{m}, \boldsymbol{\phi}\right)\\
$$

**首先，定义HMM中用到的变量**（2.1节适用）

给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X=\{x_1,x_2,\dots, x_n\}$，求观测序列 $X$ 在模型$λ$下出现的条件概率 $P(X|λ)$：

- $A = [a_{ij}]$是**隐状态转移概率矩阵**，其中 $a_{ij}=\dfrac{A_{ij}}{\sum_{s=1}^N A_{is}}$，$A_{ij}$ 为从隐状态 $z_i$ 转移到 $z_j$ 的频率计数；
- $B = [b_j(k)]$是**观测状态生成概率矩阵**，其中 $b_j(k)=\dfrac{B_{jk}}{\sum_{s=1}^MB_{js}}$，$B_{jk}$ 是从隐状态 $z_j$ 对应观测 $x_K$ 的频率计数；
- $Π=\pi_i=\dfrac{C_i}{\sum_{s=1}^NC_s}$是**隐状态的初始概率分布**,$C_i$ 是初始隐状态为 $z_i$ 的频率计数。


**HMM可以解决的三个问题：**

1. **评估观察序列概率**。

   给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X=\{x_1,x_2,\dots, x_n\}$，计算在该模型下，观测序列 $X$ 出现的概率 **（因此HMM是生成式算法）**。用**前向后向算法**求解。

2. **模型参数学习**。
   
   给定观测序列 $X=\{x_1,x_2,\dots, x_n\}$，估计模型参数 $\lambda=(A,B,\Pi)$ ，使该模型下观测序列的条件概率$P(X|λ)$最大。用到**基于EM算法的鲍姆-韦尔奇算法**求解。

3. **预测问题/解码问题**。

   给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X$，求给定观测序列条件下，最可能出现的对应的隐状态序列 $Z=\{z_1, z_2,\dots,z_n\}$ 。用到**基于动态规划的维特比算法**求解。

以下对上述三种问题及三种解法展开叙述。



#### **2.1.1 观测序列概率**
##### **A. 暴力解法**：

1. 任意一个隐状态序列 $Z=\{z_1, z_2,\dots,z_n\}$ 出现的概率是：

$$
P(Z \mid \lambda)=\pi_{z_{1}} a_{z_{1} z_{2}} a_{z_{2} z_{3}} \ldots a_{z_{n-1}z_{z_{n}}}\\
$$

2. 对于固定的状态序列 $Z=\{z_1, z_2,\dots,z_n\}$，我们要求的观察序列 $X=\{x_1,x_2,\dots, x_n\}$ 出现的概率是
   $$
   P(X \mid Z, \lambda)=b_{z_{1}}\left(x_{1}\right) b_{z_{2}}\left(x_{2}\right) \dots b_{z_{n}}\left(x_{n}\right)\\
   $$

3. 则 $X$ 和 $Z$ 联合出现的概率是：
   $$
   P(X, Z \mid \lambda)=P(Z \mid \lambda) P(O \mid Z, \lambda)=\pi_{z_{1}} b_{z_{1}}\left(x_{1}\right) a_{z_{1} z_{2}} b_{z_{2}}\left(x_{2}\right) \ldots a_{z_{n-1}} z_{n} b_{z_{n}}\left(x_{n}\right)\\
   $$
   
4. 边缘概率分布，即可得到观测序列 $X$ 在模型 $\lambda$ 下出现的条件概率 $P(X|λ)$：
   $$
   P(X \mid \lambda)=\sum_{Z} P(X, Z \mid \lambda)=\sum_{z_{1}, z_{2}, \ldots, z_{T}} \pi_{z_{1}} b_{z_{1}}(x_{1}) a_{z_{1} z_{2}} b_{z_{2}}\left(x_{2}\right) \ldots a_{z_{n-1}} z_{n} b_{z_{n}}\left(x_{n}\right)\\
   $$

   

##### **B. 前向后向算法**：
前向后向算法是 `前向 + 后向` 的形式，即分别从 start 和 stop 位置开始递推，这样才能承接上下文。用**动态规划**的思想从子问题的最优解得到整个问题的最优解。

1. **定义前向概率**：定义位置 $t$ 时对应的隐状态为 $q_i$，从 $1\sim t$ 观测为 $x_1,x_2,\dots,x_t$，记作 $\alpha_t(i)=P(x_1,x_2,\dots,x_t,z_t=q_i|\lambda)$
2. **计算位置 1 的各个隐藏状态前向概率**：
   $$
   \alpha_{1}(i)=\pi_{i} b_{i}\left(x_{1}\right), i=1,2, \ldots N\\
   $$
3. **递推 $2,3,...T$ 位置的前向概率**：
   $$
   \alpha_{t+1}(i)=\left[\sum_{j=1}^{N} \alpha_{t}(j) a_{j i}\right] b_{i}\left(x_{t+1}\right), i=1,2, \ldots N\\
   $$
4. **最终结果**：$P(X \mid \lambda)=\sum_{i=1}^{N} \alpha_{T}(i)$

后向算法同理

1. **定义后向概率**：定义位置 $t$ 时对应的隐状态为 $q_i$，从 $t+1 \sim T$ 观测为 $x_{t+1},x_{t+2},\dots,x_{T}$，记作 $\beta_{t}(i)=P\left(x_{t+1},x_{t+2},\dots,x_{T},  z_{t}=q_{i}\mid \lambda\right)$
2. **初始化位置 T 的各个隐藏状态后向概率**：
   $$
   \beta_T(i)=1, i=1,2,\dots, N\\
   $$
3. **递推 $T-1,T-2,...1$ 位置的后向概率**：
   $$
   \beta_{t}(i)=\sum_{j=1}^{N} a_{i j} b_{j}\left(x_{t+1}\right) \beta_{t+1}(j), i=1,2, \ldots N\\
   $$
4. **最终结果**：$P(X \mid \lambda)=\sum_{i=1}^{N} \pi_{i} b_{i}\left(x_{1}\right) \beta_{1}(i)$

统一形式
$$
   P(\mathbf{X} | \lambda)=\sum_{i=1}^{Q} \sum_{j=1}^{Q} \alpha_{t}(i) a_{i, j} b_{j}\left(x_{t+1}\right) \beta_{t+1}(j), \quad t=1,2, \cdots, T-1\\
$$

#### **2.1.2 观测序列概率 Python 实现**
本章使用 `hmmlearn` 库实现，[文档地址](https://hmmlearn.readthedocs.io/en/latest/)

```python
import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

seen = np.array([[0,1,0]]).T
print model.score(seen)
```
**注意**：score函数返回的是**以自然对数为底的对数概率值**，因此会是负数。

#### **2.1.3 HMM 参数学习**

##### **A. 鲍姆-韦尔奇算法原理**
HMM 的参数学习一般采用 **鲍姆-韦尔奇算法**，其思想就是 **EM算法**。

- **E步**：求出联合分布$P(X,Z|λ)$基于条件概率$P(Z|X,\bar \lambda)$的期望，其中 $\bar \lambda$ 为模型当前的参数：
  $$
   L(\lambda, \bar{\lambda})=\sum_{Z} P(Z|X,\bar \lambda) \log P(X,Z|λ)\\
  $$
- **M步**：最大化上式的期望，得到更新的模型参数λ。
  $$
   \bar{\lambda}=\arg \max _{\lambda} \sum_{Z} P(Z|X,\bar \lambda) \log P(X,Z|λ)\\
  $$
- 不断进行EM迭代，直到模型参数的值收敛为止。

##### **B. 鲍姆-韦尔奇算法推导**
TODO

##### **C. 鲍姆-韦尔奇算法流程**

**输入**： $D$ 个观测序列样本 $\{(X_1), (X_2), \dots, (X_D)\}$

**输出**：HMM模型参数
1. 随机初始化三个矩阵参数 $\pi_i, a_{ij}, b_j(k)$
2. 对于每个样本 $d=1,2,...D$，用前向后向算法计算 $\gamma_{t}^{(d)}(i), \quad \xi_{t}^{(d)}(i, j), t=1,2 \ldots T$
3. 更新模型参数：
   $$
   \begin{aligned}
   &\pi_{i}=\frac{\sum_{d=1}^{D} \gamma_{1}^{(d)}(i)}{D}\\
   &a_{i j}=\frac{\sum_{d=1}^{D} \sum_{t=1}^{T-1} \xi_{t}^{(d)}(i, j)}{\sum_{d=1}^{D} \sum_{t=1}^{T-1} \gamma_{t}^{(d)}(i)}\\
   &b_{j}(k)=\frac{\sum_{d=1}^{D} \sum_{t=1, x_{t}^{(d)}=z_{k}}^{T} \gamma_{t}^{(d)}(j)}{\sum_{d=1}^{D} \sum_{t=1}^{T} \gamma_{t}^{(d)}(j)}
   \end{aligned}\\
   $$
4. 重复 2-3 直至三种参数收敛。



#### **2.1.4 HMM 参数学习 Python 实现**

```python
import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0],[1],[0],[1],[0],[0],[0],[1],[1],[0],[1],[1]])
model2.fit(X2,lengths=[4,4,4])
print model2.startprob_
print model2.transmat_
print model2.emissionprob_
print model2.score(X2)
model2.fit(X2)
print model2.startprob_
print model2.transmat_
print model2.emissionprob_
print model2.score(X2)
model2.fit(X2)
print model2.startprob_
print model2.transmat_
print model2.emissionprob_
print model2.score(X2)
```

> 有现成的库真的是简单，仿佛上一节是在讲另一件事。



#### **2.1.5 HMM 解码问题**

##### **A. HMM 解码问题概述**
HMM 的解码问题是**给定模型，求给定观测序列条件下，最可能出现的对应的隐藏状态序列**。

**解码问题公式定义**：

已知参数 $\lambda$，观测序列 $X = \{x_1, x_2, \dots, x_T\}$，找到隐状态序列 $Z^* = \{z_1^*, z_2^*, \dots, z_T^*\}$，使得 $P(Z^*|X)$ 最大。

根据 2.2.1 B 节的定义，在给定模型和观测的情况下，可以通过HMM的前向后向算法计算位置 $t$ 隐状态为 $q_i$ 的概率 $\gamma_t(i)$。则有

$$
z_{t}^{*}=\arg \max _{1 \leq i \leq N}\left[\gamma_{t}(i)\right], t=1,2, \ldots T\\
$$

然而这样计算并不能保证隐状态序列整体是最优的，这就像多智能体合作的平均和博弈一样，不能只在一个智能体上求最优而致使其余智能体效果很差。反映在HMM中，就是相邻隐状态可能存在转移概率为0的情况。

在博弈问题中，我们需要找到一个纳什均衡点。在多目标优化问题中，就叫做帕累托最优点。而HMM的解法就要依靠维特比算法。

##### **B. 维特比算法概述**
TODO

##### **C. 维特比算法流程**
**输入**：HMM模型 $\lambda = (A,B, \Pi)$，观测序列 $X=\{x_1, x_2, \dots, x_T\}$；

**输出**：最有可能的隐状态序列 $Z^* = \{z_1^*, z_2^*, \dots, z_T^*\}$ 

1. 初始化局部状态：
   
   $$
   \begin{array}{c}
   \delta_{1}(i)=\pi_{i} b_{i}\left(x_{1}\right), i=1,2 \ldots N \\
   \Psi_{1}(i)=0, i=1,2 \ldots N
   \end{array}\\
   $$
2. 动态规划递推位置 $t=2, 3, \dots, T$时刻的局部状态：
   
   $$
   \begin{array}{l}
   \delta_{t}(i)=\max _{1 \leq j \leq N}\left[\delta_{t-1}(j) a_{j i}\right] b_{i}\left(x_{t}\right), i=1,2 \ldots N \\
   \Psi_{t}(i)=\arg \max _{1 \leq j \leq N}\left[\delta_{t-1}(j) a_{j i}\right], i=1,2 \ldots N
   \end{array}\\
   $$
3. 位置 $T$ 最大的 $\mathfrak{g} \delta_{T}(i)$ 就是最优隐状态序列出现的概率。最大的 $\Psi_{T}(i)$ 就是$T$ 位置最可能的隐状态。

   $$
   \begin{array}{c}
   P *=\max _{1 \leq j \leq N} \delta_{T}(i) \\
   i_{T}^{*}=\arg \max _{1 \leq j \leq N}\left[\delta_{T}(i)\right]
   \end{array}\\
   $$

4. 利用局部状态 $\Psi(i)$ 开始回溯。对于$t=T−1,T−2,...,1$：
   
   $$
   i_{t}^{*}=\Psi_{t+1}\left(i_{t+1}^{*}\right)\\
   $$

5. 最终得到最有可能的隐状态序列 $Z^* = \{z_1^*, z_2^*, \dots, z_T^*\}$ 

#### **2.1.6 HMM 解码 Python 实现**
承接 2.1.2

```python
seen = np.array([[0,1,0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
print("The hidden box", ", ".join(map(lambda x: states[x], box)))
```





### 2.2 线性动态系统 LDS

> 隐变量和观测变量都是连续的⾼斯变量

> **引例**：假设我们希望使⽤⼀个有噪声的传感器测量⼀个未知量z的值，传感器返回⼀个观测值x， 表⽰z的值加上⼀个零均值的⾼斯噪声。给定⼀个单次的测量，我们关于z的最好的猜测是假 设z = x。
>
> - 一个简单的思路是，通过取多次测量然后求平均的⽅法提⾼我们对z的估计效果，因为随机噪声项倾向于彼此抵消。
>
> 那么我们将情况变得更复杂一些。假设我们希望测量⼀个随着时间变化的量z。
>
> - 简单思路：如果依然**简单地对测量求平均**，由随机噪声产⽣的误差确实会被消去，然而z的时间变化也被平均掉了，从⽽引⼊了⼀种新的误差。
> - 好一点的思路：为了估计$z_N$的值，我们只**取最近的⼏次测量** $\boldsymbol{x}_{N-L}, \ldots, \boldsymbol{x}_{N}$ **求平均**。
>   - z变化很慢，且随机噪声的⽔平很⾼，选择⼀个相对长的窗求平均；
>   - z变化很快，且噪声⽔平相对较⼩，直接使⽤$x_N$来估计$z_N$会更合适。
>   - 如果我们求**加权平均**，即最近的测量⽐之前的测量的贡献更⼤，那么或许效果会更好。

**那么该如何定义这个加权平均？总不能handcraft吧**

定义⼀个概率模型，它描述了时间的演化和测量过程。



由于模型由树结构的有向图表⽰，因此推断问题可以使⽤加和-乘积算法⾼效地求解。

- **前向递归方程**，类似于 HMM 的$α$信息，被称为**Kalman滤波 (Kalman filter) 方程**（Kalman, 1960; Zarchan and Musoff, 2005），
- **后向递归⽅程**，类似于$β$信息，被称为**Kalman平滑 (Kalman smoother) 方程**，或者**Rauch-Tung-Striebel (RTS) 方程**（Rauch et al., 1965）。



**单独地概率最⼤的潜在变量值组成的序列与概率 最⼤的潜在变量序列相同**

转移分布：
$$
p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}\right)=\mathcal{N}\left(\boldsymbol{z}_{n} \mid \boldsymbol{A} \boldsymbol{z}_{n-1}, \boldsymbol{\Gamma}\right)\\
$$
发射分布：
$$
p\left(\boldsymbol{x}_{n} \mid \boldsymbol{z}_{n}\right)=\mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{C} \boldsymbol{z}_{n}, \boldsymbol{\Sigma}\right)\\
$$
初始隐变量也服从高斯分布：
$$
p\left(\boldsymbol{z}_{1}\right)=\mathcal{N}\left(\boldsymbol{z}_{1} \mid \boldsymbol{\mu}_{0}, \boldsymbol{P}_{0}\right)\\
$$
以上三个式子写作矩阵形式：
$$
\begin{array}{c}
\boldsymbol{z}_{n}=\boldsymbol{A} \boldsymbol{z}_{n-1}+\boldsymbol{w}_{n} \\
\boldsymbol{x}_{n}=\boldsymbol{C} \boldsymbol{z}_{n}+\boldsymbol{v}_{n} \\
\boldsymbol{z}_{1}=\boldsymbol{\mu}_{0}+\boldsymbol{u}
\end{array}\\
$$
其中噪声项为
$$
\begin{aligned}
\boldsymbol{w} \sim \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{0}, \boldsymbol{\Gamma}) \\
\boldsymbol{v} \sim \mathcal{N}(\boldsymbol{v} \mid \boldsymbol{0}, \boldsymbol{\Sigma}) \\
\boldsymbol{u} \sim \mathcal{N}\left(\boldsymbol{u} \mid \boldsymbol{0}, \boldsymbol{P}_{0}\right)
\end{aligned}\\
$$
由此，模型参数记作 $\boldsymbol{\theta}=\left\{\boldsymbol{A}, \boldsymbol{\Gamma}, \boldsymbol{C}, \boldsymbol{\Sigma}, \boldsymbol{\mu}_{0}, \boldsymbol{P}_{0}\right\}$

#### **2.2.1 LDS推断**

LDS的推断要解决**以观测序列为条件的隐变量的边缘概率分布**。我们也希望以观测数据        $x_{1}, \ldots, x_{n-1}$ 为条件，对于下⼀个潜在状态 $z_n$ 以及下 ⼀个观测 $x_n$ 做出预测。

线性动态系统与隐马尔可夫模型具有相同的分解⽅式。
$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}, \boldsymbol{z}_{1}, \ldots, \boldsymbol{z}_{N}\right)=p\left(\boldsymbol{z}_{1}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}\right)\right] \prod_{n=1}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{z}_{n}\right)\\
$$


#### **2.2.2 LDS学习**
TODO





## 3 马尔可夫随机场 MRF
> **什么叫随机场？**
> 
> 随机场是由若干个位置组成的整体，当给每一个位置中按照某种分布随机赋予一个值之后，其全体就叫做随机场。
> 
> **例**：假如我们有一个十个词形成的句子需要做词性标注。这十个词每个词的词性可以在我们已知的词性集合（名词，动词...)中去选择。当我们为每个词选择完词性后，这就形成了一个随机场。

马尔可夫随机场是随机场的一个具有马尔可夫性得特例，它假设随机场中某一个位置的赋值仅仅与和它相邻的位置的赋值有关，和与其不相邻的位置的赋值无关。

**例**：如果我们假设所有词的词性只和它相邻的词的词性有关时，这个随机场就特化成一个马尔科夫随机场。比如第三个词的词性除了与自己本身的位置有关外，只与第二个词和第四个词的词性有关。　

## 4 条件随机场 CRF

条件随机场 Conditional Random Field 是 **马尔可夫随机场 + 隐状态**的特例。

区别于生成式的隐马尔可夫模型，CRF是**判别式**的。CRF 试图对多个随机变量（代表状态序列）在给定观测序列的值之后的条件概率进行建模：

**给定观测序列 $X=\{x_1,x_2,\dots, x_n\}$，以及隐状态序列 $Y=\{y_1, y_2,\dots,y_n\}$ 的情况下，构建条件概率模型 $P(Y|X)$。若随机变量Y构成的是一个马尔科夫随机场，则 $P(Y|X)$ 为CRF。**



与HMM类似，CRF也关心如下三种问题：

1. **概率计算问题**。
   已知条件随机场  $P(Y|X)$，给定观测序列 $\tilde{\mathbf{X}}=\left\{\tilde{\mathbf{x}}_{1}, \tilde{\mathbf{x}}_{2}, \cdots, \tilde{\mathbf{x}}_{n}\right\}$、标记序列 $\tilde{\mathbf{Y}}=\left\{\tilde{\mathbf{y}}_{1}, \tilde{\mathbf{y}}_{2}, \cdots, \tilde{\mathbf{y}}_{n}\right\}$ ，求

   - 条件概率： $P\left(Y_{i}=\tilde{\mathbf{y}}_{i} \mid \tilde{\mathbf{X}}\right)$。
   - 条件概率： $P\left(Y_{i}=\tilde{\mathbf{y}}_{i}, Y_{i+1}=\tilde{\mathbf{y}}_{i+1} \mid \tilde{\mathbf{X}}\right)$ 。

   用 **前向-后向算法** 求解。

2. **参数学习**。
   CRF 实际上是定义在时序数据上的对数线性模型，其学习方法包括：极大似然估计、正则化的极大似然估计。
   用**改进的迭代尺度法`Improved Iterative Scaling:IIS`、 梯度下降法、拟牛顿法** 求解。

3. **预测、解码问题**。

   给定条件随机场  $P(Y|X)$，给定观测序列 $\tilde{\mathbf{X}}=\left\{\tilde{\mathbf{x}}_{1}, \tilde{\mathbf{x}}_{2}, \cdots, \tilde{\mathbf{x}}_{n}\right\}$，求条件概率最大的输出序列（标记序列）$\mathbf{Y}^{*}=\left\{\tilde{\mathbf{y}}_{1}^{*}, \tilde{\mathbf{y}}_{2}^{*}, \cdots, \tilde{\mathbf{y}}_{n}^{*}\right\}$，其中 $\tilde{\mathbf{y}}_{i}^{*} \in \mathcal{Y}=\left\{\mathbf{y}_{1}, \mathbf{y}_{2}, \cdots, \mathbf{y}_{m}\right\}$。
   用**维特比算法**求解。

   

### 4.1 线性链式条件随机场 linear-CRF / chain-structured CRF

> X和Y有相同的结构的CRF

![四、条件随机场 CRF - 图17](https://static.bookstack.cn/projects/huaxiaozhuan-ai/3960d6d4fa9731cb57ec01b05e1c0ca3.jpeg)

给定观测变量序列 $X=\{x_1,x_2,\dots, x_n\}$, 隐状态序列 $Y=\{y_1, y_2,\dots,y_n\}$，链式条件随机场主要包含两种关于标记变量的团：

- 单个标记变量与 $X$ 构成的团： $\{y_i, X\}, i=1,2,\dots, n$ 。
- 相邻标记变量与 $X$ 构成的团： $\{y_{i-1},y_i,X\}, , i=1,2,\dots, n$ 。

对于单个变量的团是满足马尔可夫性的：

$$
P(y_i|X, y_1, \dots, y_n)=P(y_i| X,y_{i-1}, y_{i+1}) \\
$$

> **那么 CRF 的参数学习是在学什么？**
> 
> 特征函数和其权重系数。
> 
> **注意**：特征函数听着玄学，实际上就是个bool值，只能是0或者1。


与马尔可夫随机场定义联合概率的方式类似，条件随机场使用势函数和团来定义条件概率 $P(Y|X)$:
$$
P(\mathbf{Y} \mid \mathbf{X})=\frac{1}{Z} \exp \left(\sum_{k=1}^{K_{1}} \sum_{i=1}^{n-1} \lambda_{k} t_{k}\left(y_{i}, y_{i+1}, \mathbf{X}, i\right)+\sum_{l=1}^{K_{2}} \sum_{i=1}^{n} \mu_{l} s_{l}\left(y_{i}, \mathbf{X}, i\right)\right)\\
$$
其中：

- $t_{k}\left(y_{i}, y_{i+1}, \mathbf{X}, i\right)$ ：在已知观测序列情况下，两个相邻标记位置上的**转移特征函数**`transition feature function`。

  - 它刻画了相邻标记变量之间的相关关系，以及观察序列 $X$ 对它们的影响。
  - 位置变量 $i$ 也对势函数有影响。比如：已知观测序列情况下，相邻标记取值`(代词，动词)`出现在序列头部可能性较高，而`(动词，代词)`出现在序列头部的可能性较低。

- $s_{l}\left(y_{i}, \mathbf{X}, i\right)$ ：在已知观察序列情况下，标记位置 $i$ 上的**状态特征函数** `status feature function`。

  - 它刻画了观测序列 $X$ 对于标记变量的影响。
  - 位置变量 $i$ 也对势函数有影响。比如：已知观测序列情况下，标记取值 `名词`出现在序列头部可能性较高，而 `动词` 出现在序列头部的可能性较低。

- $\lambda_{k}, \mu_{l}$ 为参数，$Z$ 为规范化因子（它用于确保上式满足概率的定义）。

  $K_1$ 为转移特征函数的个数，$K_2$ 为状态特征函数的个数。

> **特征函数和权重系数的实际意义是什么？**
> 
> 每个特征函数定义了一个linear-CRF的规则，则其系数定义了这个规则的可信度。所有的规则和其可信度一起构成了我们的linear-CRF的最终的条件概率分布。

### 4.2 linear-CRF 的矩阵形式
**简化特征函数**

假设总共有$K=K_1+K_2$个特征函数。我们用一个特征函数 $f_k(y_i, y_i, X, i)$ 来表示：

$$
f_{k}\left(y_{i-1}, y_{i}, X, i\right)=\left\{\begin{array}{ll}
t_{k}\left(y_{i-1}, y_{i}, X, i\right) & k=1,2, \ldots K_{1} \\
s_{l}\left(y_{i}, X, i\right) & k=K_{1}+l, l=1,2 \ldots, K_{2}
\end{array}\right.\\
$$

对 $f_k(y_i, y_i, X, i)$ 在各个序列位置求和得到：

$$
f_{k}(Y, X)=\sum_{i=1}^{n} f_{k}\left(y_{i-1}, y_{i}, X, i\right)\\
$$

**简化权重系数**

$$
w_{k}=\left\{\begin{array}{ll}
\lambda_{k} & k=1,2, \ldots K_{1} \\
\mu_{l} & k=K_{1}+l, l=1,2 \ldots, K_{2}
\end{array}\right.\\
$$

**得到参数化后的linear-CRF**

$$
P(Y\mid X)=\frac{1}{Z(X)} \exp \sum_{k=1}^{K} w_{k} f_{k}(Y, X)\\
$$

其中，归一化因子为

$$
Z(X)=\sum_{Y} \exp \sum_{k=1}^{K} w_{k} f_{k}(Y, X)\\
$$

**参数表示为向量**

$$w=\left(w_{1}, w_{2}, \ldots w_{K}\right)^{T} \quad F(Y,X)=\left(f_{1}(Y,X), f_{2}(Y,X), \ldots f_{K}(Y,X)\right)^{T}\\$$

**linear-CRF的内积形式**

$$P_{w}(Y \mid X)=\frac{\exp (w \bullet F(Y,X))}{Z_{w}(X)}=\frac{\exp (w \bullet F(Y,X))}{\sum_{Y} \exp (w \bullet F(Y,X))}\\$$

**linear-CRF的矩阵形式**

定义一个$m×m$的矩阵$M$，$m$为$Y$所有可能的状态的取值个数

$$M_{i}(X)=\left[M_{i}\left(y_{i-1}, y_{i} \mid X\right)\right]=\left[\exp \left(W_{i}\left(y_{i-1}, y_{i} \mid X\right)\right)\right]=\left[\exp \left(\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i-1}, y_{i}, X, i\right)\right)\right]\\$$

$$P_{w}(Y \mid X)=\frac{1}{Z_{w}(x)} \prod_{i=1}^{n+1} M_{i}\left(y_{i-1}, y_{i} \mid X\right)\\$$

### 



### 4.3 概率计算问题

#### **4.3.1 定义前向概率 $\alpha_i(y_i|X)$**

表示在位置 $i$ 的标记是 $y_i$ ，并且到位置 $i$ 的前半部分标记序列的非规范化概率（即暂不考虑 $Z(X)$）。

根据 

$$M_{i}\left(y_{i-1}, y_{i} \mid X\right)=\exp \left(\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i-1}, y_{i}, X, i\right)\right)\\$$

可得

$$\alpha_{i+1}\left(y_{i+1} \mid X\right)=\alpha_{i}\left(y_{i} \mid X\right)\left[M_{i+1}\left(y_{i+1}, y_{i} \mid X\right)\right] \quad i=1,2, \ldots, n+1\\$$

在起点处，我们定义：
$$\alpha_{0}\left(y_{0} \mid x\right)=\left\{\begin{array}{ll}
1 & y_{0}=\text {start} \\
0 & \text {else}
\end{array}\right.\\$$

写成**前向向量**形式：

$$\alpha_{i}(x)=\left(\alpha_{i}\left(y_{i}=1 \mid x\right), \alpha_{i}\left(y_{i}=2 \mid x\right), \ldots \alpha_{i}\left(y_{i}=m \mid x\right)\right)^{T}\\$$

同时 $M_{i}(x)=\left[M_{i}\left(y_{i-1}, y_{i} \mid x\right)\right]$，故这样的递推公式用矩阵乘法表示为：

$$\alpha_{i+1}^{T}(x)=\alpha_{i}^{T}(x) M_{i+1}(x)\\$$

#### **4.3.2 定义后向概率 $\beta_i(y_i|X)$**

同理，后向递推的矩阵乘法表示为

$$\beta_{i}(x)=M_{i+1}(x) \beta_{i+1}(x)\\$$

在终点处

$$\beta_{n+1}\left(y_{n+1} \mid x\right)=\left\{\begin{array}{ll}
1 & y_{n+1}=s t o p \\
0 & \text { else }
\end{array}\right.\\$$

#### **4.3.3 linear-CRF的前向后向概率计算**
由此可得到 序列位置$i$的标记是$y_i$时的条件概率$P(y_i|x)$:
$$P\left(y_{i} \mid x\right)=\frac{\alpha_{i}^{T}\left(y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{Z(x)}=\frac{\alpha_{i}^{T}\left(y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{\alpha_{n}^{T}(x) \bullet \mathbf{1}}\\$$

也容易计算序列位置i的标记是$y_i$，位置i−1的标记是$y_i−1$ 时的条件概率$P(y_i−1,y_i|x)$:
$$
P\left(y_{i-1}, y_{i} \mid x\right)=\frac{\alpha_{i-1}^{T}\left(y_{i-1} \mid x\right) M_{i}\left(y_{i-1}, y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{Z(x)}=\frac{\alpha_{i-1}^{T}\left(y_{i-1} \mid x\right) M_{i}\left(y_{i-1}, y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{\alpha_{n}^{T}(x) \bullet 1}\\
$$
其中 
$$
Z(x)=\sum_{c=1}^{m} \alpha_{n}\left(y_{c} \mid x\right)=\sum_{c=1}^{m} \beta_{1}\left(y_{c} \mid x\right)=\alpha_{n}^{T}(x) \bullet \mathbf{1}=\mathbf{1}^{T} \bullet \beta_{1}(x)\\
$$

#### **4.3.4 linear-CRF的前向后向期望计算**

有了上一节计算的条件概率，我们也可以很方便的计算联合分布$P(x,y)$与条件分布$P(y|x)$的期望。

特征函数$f_k(x,y)$关于条件分布$P(y|x)$的期望表达式是：
$$
\begin{aligned}
E_{P(y \mid x)}\left[f_{k}\right] &=E_{P(y \mid x)}\left[f_{k}(y, x)\right] \\
&=\sum_{i=1}^{n+1} \sum_{y_{i-1} y_{i}} P\left(y_{i-1}, y_{i} \mid x\right) f_{k}\left(y_{i-1}, y_{i}, x, i\right) \\
&=\sum_{i=1}^{n+1} \sum_{y_{i-1} y_{i}} f_{k}\left(y_{i-1}, y_{i}, x, i\right) \frac{\alpha_{i-1}^{T}\left(y_{i-1} \mid x\right) M_{i}\left(y_{i-1}, y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{\alpha_{n}^{T}(x) \cdot 1}
\end{aligned}\\
$$
同时，联合分布的期望为：
$$
\begin{aligned}
E_{P(x, y)}\left[f_{k}\right] &=\sum_{x, y} P(x, y) \sum_{i=1}^{n+1} f_{k}\left(y_{i-1}, y_{i}, x, i\right) \\
&=\sum_{x} \bar{P}(x) \sum_{y} P(y \mid x) \sum_{i=1}^{n+1} f_{k}\left(y_{i-1}, y_{i}, x, i\right) \\
&=\sum_{x} \bar{P}(x) \sum_{i=1}^{n+1} \sum_{y_{i-1} y_{i}} f_{k}\left(y_{i-1}, y_{i}, x, i\right) \frac{\alpha_{i-1}^{T}\left(y_{i-1} \mid x\right) M_{i}\left(y_{i-1}, y_{i} \mid x\right) \beta_{i}\left(y_{i} \mid x\right)}{\alpha_{n}^{T}(x) \cdot 1}
\end{aligned}\\
$$

#### **4.3.5 Python 实现**

借助 `sklearn-crfsuite` 库实现，和 sklearn 格式一样，[文档链接](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html)

```python
import sklearn_crfsuite

# 这里的 crf 在 4.4.2 节定义
# observed_xseq: new observations
observed_xseq = ...  # type -> list of dicts
# y_probs: predicted probabilities for each label at each position
y_probs = crf.predict_marginals_single(observed_xseq)  # type -> list of dicts
```



### 4.4 CRF参数学习算法

#### **4.4.1 改进的迭代尺度法、拟牛顿法**

详见 [链接](https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.4.a1c8cb11a2e246b2.md#4b9wvq)

#### **4.4.2 Python 实现**

借助 `sklearn_crfsuite` 库实现，和 sklearn 格式一样，[文档链接](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)

```python
import sklearn_crfsuite

X_train = ...
y_train = ...

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)
# CPU times: user 32 s, sys: 108 ms, total: 32.1 s
# Wall time: 32.3 s
y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
# 0.76980231377134023
```



此外，还可以尝试一下 `PyStruct` 库，[文档链接](http://pystruct.github.io/index.html)



### 4.5 预测问题

#### **4.5.1 linear-CRF 维特比算法解码**

给定条件随机场  $P(Y|X)$，给定观测序列 $\tilde{\mathbf{X}}=\left\{\tilde{\mathbf{x}}_{1}, \tilde{\mathbf{x}}_{2}, \cdots, \tilde{\mathbf{x}}_{n}\right\}$，求条件概率最大的输出序列（标记序列）$Y^*$。

维特比算法本身是一个动态规划算法，利用了两个局部状态和对应的递推公式，从局部递推到整体，进而得解。对于具体不同的问题，仅仅是这两个局部状态的定义和对应的递推公式不同而已。由于在之前已详述维特比算法，这里就是做一个简略的流程描述。

对于我们linear-CRF中的维特比算法，我们的第一个局部状态定义为$δ_i(l)$,表示在位置 $i$ 标记 $l$ 各个可能取值 $(1,2...m)$ 对应的非规范化概率的最大值。

根据$δ_i(l)$的定义，我们递推在位置$i+1$标记 $l$ 的表达式为：
$$
\delta_{i+1}(l)=\max _{1 \leq j \leq m}\left\{\delta_{i}(j)+\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i}=j, y_{i+1}=l, x, i\right)\right\}, l=1,2, \ldots m\\
$$
和HMM的维特比算法类似，我们需要用另一个局部状态 $Ψ_{i+1}(l)$ 来记录使 $δ_{i+1}(l)$ 达到最大的位置 $i$ 的标记取值,这个值用来最终回溯最优解， $Ψ_{i+1}(l)$ 的递推表达式为：
$$
\Psi_{i+1}(l)=\arg \max _{1 \leq j \leq m}\left\{\delta_{i}(j)+\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i}=j, y_{i+1}=l, x, i\right)\right\}, l=1,2, \ldots m\\
$$

#### **4.5.2 linear-CRF 维特比算法流程**

输入：模型的KK个特征函数，和对应的K个权重。观测序列x=(x1,x2,...xn)x=(x1,x2,...xn),可能的标记个数mm

输出：最优标记序列y∗=(y∗1,y∗2,...y∗n)

1. 初始化：
   $$
   \begin{array}{c}
   \left.\delta_{1}(l)=\sum_{k=1}^{K} w_{k} f_{k}\left(y_{0}=\operatorname{start}, y_{1}=l, x, i\right)\right\}, l=1,2, \ldots m \\
   \Psi_{1}(l)=\text {start }, l=1,2, \ldots m
   \end{array}\\
   $$
   
2. 对于$i=1,2...n−1$,进行递推：
   $$
   \begin{array}{c}
   \delta_{i+1}(l)=\max _{1 \leq j \leq m}\left\{\delta_{i}(j)+\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i}=j, y_{i+1}=l, x, i\right)\right\}, l=1,2, \ldots m \\
   \Psi_{i+1}(l)=\arg \max _{1 \leq j \leq m}\left\{\delta_{i}(j)+\sum_{k=1}^{K} w_{k} f_{k}\left(y_{i}=j, y_{i+1}=l, x, i\right)\right\}, l=1,2, \ldots m
   \end{array}\\
   $$

3. 终止：
   $$
   y_{n}^{*}=\arg \max _{1 \leq j \leq m} \delta_{n}(j)\\
   $$

4. 回溯：
   $$
   y_{i}^{*}=\Psi_{i+1}\left(y_{i+1}^{*}\right), i=n-1, n-2, \ldots 1\\
   $$

最后得到最优标记序列 $Y^{*}=\left(y_{1}^{*}, y_{2}^{*}, \ldots y_{n}^{*}\right)$



#### **4.5.3 Python 实现**

借助 `sklearn-crfsuite` 库实现，和 sklearn 格式一样，[文档链接](https://sklearn-crfsuite.readthedocs.io/en/latest/api.html)

```python
import sklearn_crfsuite

# observed_xseq: new observations
observed_xseq = ...  # type -> list of dicts
# y: predicted labels
y = crf.predict_single(observed_xseq)  # type -> list of strings
```



## 5 Reference
1. [条件随机场CRF(一)从随机场到线性链条件随机场](https://www.cnblogs.com/pinard/p/7048333.html)
2. [条件随机场CRF(二) 前向后向算法评估标记序列概率](https://www.cnblogs.com/pinard/p/7055072.html)
3. [条件随机场CRF(三) 模型学习与维特比算法解码](https://www.cnblogs.com/pinard/p/7068574.html)
4. [隐马尔可夫模型 - AI算法工程师手册](https://www.bookstack.cn/read/huaxiaozhuan-ai/1948f43fc33e8d4c.md)
5. [马尔可夫随机场 - AI算法工程师手册](https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.3.a1c8cb11a2e246b2.md)
6. [条件随机场 CRF - AI算法工程师手册](https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.4.a1c8cb11a2e246b2.md)
7. PRML
8. Deep Learning - Ian Goodfellow

