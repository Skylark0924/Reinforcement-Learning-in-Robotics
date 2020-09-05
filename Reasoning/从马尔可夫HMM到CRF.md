#! https://zhuanlan.zhihu.com/p/213271759

![Image](https://pic4.zhimg.com/80/v2-c0a76aeb48e699c638bf0215fbc8395e.jpg)
# PR Reasoning Ⅰ：从马尔可夫、隐马尔可夫 HMM 到条件随机场 CRF

![Image](https://pic4.zhimg.com/80/v2-d73dde1f353a6c95720dd68e4b73a1b4.jpg)
[TOC]

**未完，暂存**

马尔可夫不仅是强化学习在时序决策上的理论基础，也是语音、NLP等领域处理时序数据并进行预测的基础。本章分以下三部分：

1. 对将之前所有的观测作为未来预测的依据是**不现实的**，因为其复杂度会随着观测数量的增加而无限制地增长。因此，使用马尔科夫模型，假定**未来的预测仅与最近的观测有关，而独立于其他所有的观测**；
2. 通过引入**隐变量**，解决Markov Model需要强独立性的问题，即隐马尔可夫模型  HMM；
3. **HMM等为生成式模型，计算联合概率分布，只能得到局部最优；CRF则是判别式模型，计算条件概率，可以得到全局最优**。

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

#### 1.1.1 Autoregressive (AR)

使⽤线性⾼斯条件概率分布， 每个结点都是⼀个⾼斯概率分布，均值是⽗结点的⼀个线性函数。

#### 1.1.2 Tapped delay line

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
p(\boldsymbol{X}, \boldsymbol{Z} \mid \boldsymbol{\theta})=p\left(\boldsymbol{z}_{1} \mid \boldsymbol{\pi}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}, \boldsymbol{A}\right)\right] \prod_{m=1}^{N} p\left(\boldsymbol{x}_{m} \mid \boldsymbol{z}_{m}, \boldsymbol{\phi}\right)\\
$$

**HMM可以解决的三个问题：**

1. **评估观察序列概率**。

   给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X=\{x_1,x_2,\dots, x_n\}$，计算在该模型下，观测序列 $X$ 出现的概率。用到**前向后向算法**求解。

2. **模型参数学习**。
   给定观测序列 $X=\{x_1,x_2,\dots, x_n\}$，估计模型参数 $\lambda=(A,B,\Pi)$ ，使该模型下观测序列的条件概率$P(X|λ)$最大。用到**基于EM算法的鲍姆-韦尔奇算法**求解。

3. **预测问题/解码问题**。

   给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X$，求给定观测序列条件下，最可能出现的对应的隐状态序列 $Z=\{z_1, z_2,\dots,z_n\}$ 。用到**基于动态规划的维特比算法**求解。

以下对上述三种问题及三种解法展开叙述。



#### 2.1.1  前向后向算法

给定模型 $\lambda=(A,B,\Pi)$ 和观测序列 $X=\{x_1,x_2,\dots, x_n\}$，求观测序列 $X$ 在模型$λ$下出现的条件概率 $P(X|λ)$：

- $A$是**隐状态转移概率矩阵**，
- $B$是**观测状态生成概率矩阵**， 
- $Π$是**隐状态的初始概率分布**。

**暴力解法**：

1. 任意一个隐状态序列 $Z=\{z_1, z_2,\dots,z_n\}$ 出现的概率是：

$$
P(Z \mid \lambda)=\pi_{z_{1}} a_{z_{1} z_{2}} a_{z_{2} z_{3}} \ldots a_{z_{n-1}z_{z_{n}}} 
$$

2. 对于固定的状态序列 $Z=\{z_1, z_2,\dots,z_n\}$，我们要求的观察序列 $X=\{x_1,x_2,\dots, x_n\}$ 出现的概率是
   $$
   P(X \mid Z, \lambda)=b_{z_{1}}\left(x_{1}\right) b_{z_{2}}\left(x_{2}\right) \dots b_{z_{n}}\left(x_{n}\right)
   $$

3. 则 $X$ 和 $Z$ 联合出现的概率是：
   $$
   P(X, Z \mid \lambda)=P(Z \mid \lambda) P(O \mid Z, \lambda)=\pi_{z_{1}} b_{z_{1}}\left(x_{1}\right) a_{z_{1} z_{2}} b_{z_{2}}\left(x_{2}\right) \ldots a_{z_{n-1}} z_{n} b_{z_{n}}\left(x_{n}\right)
   $$
   
4. 边缘概率分布，即可得到观测序列 $X$ 在模型 $\lambda$ 下出现的条件概率 $P(X|λ)$：
   $$
   P(X \mid \lambda)=\sum_{Z} P(X, Z \mid \lambda)=\sum_{z_{1}, z_{2}, \ldots, z_{T}} \pi_{z_{1}} b_{z_{1}}(x_{1}) a_{z_{1} z_{2}} b_{z_{2}}\left(x_{2}\right) \ldots a_{z_{n-1}} z_{n} b_{z_{n}}\left(x_{n}\right)
   $$

**前向后向算法**：



#### 2.1.2 HMM 极大似然估计学习





#### 2.1.3 用于HMM的加和-乘积算法



#### 2.1.4 维特比算法





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

- **前向递归方程**，类似于 HMM 的α信息，被称为**Kalman滤波 (Kalman filter) 方程**（Kalman, 1960; Zarchan and Musoff, 2005），
- **后向递归⽅程**，类似于β信息，被称为**Kalman平滑 (Kalman smoother) 方程**，或者**Rauch-Tung-Striebel (RTS) 方程**（Rauch et al., 1965）。



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

#### 2.2.1 LDS推断

LDS的推断要解决**以观测序列为条件的隐变量的边缘概率分布**。我们也希望以观测数据        $x_{1}, \ldots, x_{n-1}$ 为条件，对于下⼀个潜在状态 $z_n$ 以及下 ⼀个观测 $x_n$ 做出预测。

线性动态系统与隐马尔可夫模型具有相同的分解⽅式。
$$
p\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{N}, \boldsymbol{z}_{1}, \ldots, \boldsymbol{z}_{N}\right)=p\left(\boldsymbol{z}_{1}\right)\left[\prod_{n=2}^{N} p\left(\boldsymbol{z}_{n} \mid \boldsymbol{z}_{n-1}\right)\right] \prod_{n=1}^{N} p\left(\boldsymbol{x}_{n} \mid \boldsymbol{z}_{n}\right)\\
$$


#### 2.2.2 LDS学习









## 3 条件随机场 CRF

条件随机场 Conditional Random Field 区别于生成式的隐马尔可夫模型和马尔可夫随机场，是**判别式**的。CRF 试图对多个随机变量（代表状态序列）在给定观测序列的值之后的条件概率进行建模：

**给定观测序列 $X=\{x_1,x_2,\dots, x_n\}$，以及隐状态序列 $Z=\{z_1, z_2,\dots,z_n\}$ 的情况下，构建条件概率模型 $P(Y|X)$。**

与HMM类似，CRF也关心如下三种问题：

1. **概率计算问题**。
   已知条件随机场  $P(Y|X)$，给定观测序列 $\tilde{\mathbf{X}}=\left\{\tilde{\mathbf{x}}_{1}, \tilde{\mathbf{x}}_{2}, \cdots, \tilde{\mathbf{x}}_{n}\right\}$、标记序列 $\tilde{\mathbf{Y}}=\left\{\tilde{\mathbf{y}}_{1}, \tilde{\mathbf{y}}_{2}, \cdots, \tilde{\mathbf{y}}_{n}\right\}$ ，求

   - 条件概率： ![四、条件随机场 CRF - 图129](https://static.bookstack.cn/projects/huaxiaozhuan-ai/bc99bd8e6c2de29dfc5412884fefe70c.svg) 。
   - 条件概率： ![四、条件随机场 CRF - 图130](https://static.bookstack.cn/projects/huaxiaozhuan-ai/21fe71b8656b0fc518b78c0f564901cf.svg) 。

   用 **前向-后向算法** 求解。

2. **参数学习**。
   CRF 实际上是定义在时序数据上的对数线性模型，其学习方法包括：极大似然估计、正则化的极大似然估计。
   用**改进的迭代尺度法`Improved Iterative Scaling:IIS`、 梯度下降法、拟牛顿法** 求解。

3. **预测问题**。

   给定条件随机场  $P(Y|X)$，给定观测序列 $\tilde{\mathbf{X}}=\left\{\tilde{\mathbf{x}}_{1}, \tilde{\mathbf{x}}_{2}, \cdots, \tilde{\mathbf{x}}_{n}\right\}$，求条件概率最大的输出序列（标记序列）$\mathbf{Y}^{*}=\left\{\tilde{\mathbf{y}}_{1}^{*}, \tilde{\mathbf{y}}_{2}^{*}, \cdots, \tilde{\mathbf{y}}_{n}^{*}\right\}$，其中 $\tilde{\mathbf{y}}_{i}^{*} \in \mathcal{Y}=\left\{\mathbf{y}_{1}, \mathbf{y}_{2}, \cdots, \mathbf{y}_{m}\right\}$。
   用**维特比算法**求解。

   

### 3.1 链式条件随机场 chain-structured CRF

![四、条件随机场 CRF - 图17](https://static.bookstack.cn/projects/huaxiaozhuan-ai/3960d6d4fa9731cb57ec01b05e1c0ca3.jpeg)

给定观测变量序列 $X=\{x_1,x_2,\dots, x_n\}$ ，链式条件随机场主要包含两种关于标记变量的团：

- 单个标记变量与 $X$ 构成的团： $\{Y_i, X\}, i=1,2,\dots, n$ 。
- 相邻标记变量与 $X$ 构成的团： $\{Y_{i-1},Y_i,X\}, , i=1,2,\dots, n$ 。

与马尔可夫随机场定义联合概率的方式类似，条件随机场使用势函数和团来定义条件概率 $P(Y|X)$:
$$
P(\mathbf{Y} \mid \mathbf{X})=\frac{1}{Z} \exp \left(\sum_{j=1}^{K_{1}} \sum_{i=1}^{n-1} \lambda_{j} t_{j}\left(Y_{i}, Y_{i+1}, \mathbf{X}, i\right)+\sum_{k=1}^{K_{2}} \sum_{i=1}^{n} \mu_{k} s_{k}\left(Y_{i}, \mathbf{X}, i\right)\right)
$$
其中：

- $t_{j}\left(Y_{i}, Y_{i+1}, \mathbf{X}, i\right)$ ：在已知观测序列情况下，两个相邻标记位置上的转移特征函数`transition feature function`。

  - 它刻画了相邻标记变量之间的相关关系，以及观察序列 $X$ 对它们的影响。
  - 位置变量 $i$ 也对势函数有影响。比如：已知观测序列情况下，相邻标记取值`(代词，动词)`出现在序列头部可能性较高，而`(动词，代词)`出现在序列头部的可能性较低。

- $s_{k}\left(Y_{i}, \mathbf{X}, i\right)$ ：在已知观察序列情况下，标记位置 ![四、条件随机场 CRF - 图29](https://static.bookstack.cn/projects/huaxiaozhuan-ai/452a92dfa1a2581be3de59af9a412b14.svg) 上的状态特征函数 `status feature function`。

  - 它刻画了观测序列 $X$ 对于标记变量的影响。
  - 位置变量 ![四、条件随机场 CRF - 图31](https://static.bookstack.cn/projects/huaxiaozhuan-ai/452a92dfa1a2581be3de59af9a412b14.svg) 也对势函数有影响。比如：已知观测序列情况下，标记取值 `名词`出现在序列头部可能性较高，而 `动词` 出现在序列头部的可能性较低。

- $\lambda_{j}, \mu_{k}$ 为参数，$Z$ 为规范化因子（它用于确保上式满足概率的定义）。

  $K_1$ 为转移特征函数的个数，$K_2$ 为状态特征函数的个数。

### 3.2 概率计算问题



### 3.3 CRF参数学习算法



### 3.4 预测问题



