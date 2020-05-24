## PR Ⅴ: 高斯混合模型 Gaussian Mixture Model (GMM)

在概率的神经网络 Bayesian Neural Network之后，我们还可以用概率的方法做**聚类**（无监督学习），这就是今天要介绍的 ———— 高斯混合模型 Gaussian Mixture Model，缩写为GMM。

### 一、高斯分布

高斯分布的基础已经在前面讲过了

[](https://zhuanlan.zhihu.com/p/139478368)

从一元高斯分布到多元高斯分布都给出了图像以及概率密度公式。快速浏览一下再回来。

### 二、高斯混合模型

不论是一元还是多元高斯都是指单个高斯分布，只是特征变量数量的区别。而GMM是多个高斯分布的**混合(线性组合)**，用于解决同一集合下的数据包含多个不同的分布的情况。

![图1](./PR Ⅴ GMM.assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwMzAyMTc1NDQyMjcy.jfif)

![图2](./PR Ⅴ GMM.assets/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwMzAyMTc1NTQ5ODc3.jfif)

概率密度函数：
$$
p(x)=\sum_{j=1}^{k} \alpha_{j} \phi\left(x | \mu_{j}, \Sigma_{j}\right)
$$

- k 是单高斯分布的个数；
- $\alpha_j$是每个分布的权重，$\sum_{j=1}^k\alpha_j=1$；
- $\phi\left(x | \mu_{j}, \Sigma_{j}\right)$就是参数为 $(\mu_{j}, \Sigma_{j})$ 的高斯分布的概率密度，如果是一元高斯，那么就写作：

$$
\phi(x|\mu_{j}, \Sigma_{j})=\frac{1}{\sigma \sqrt{2 \pi}} \exp \left(-\frac{(x-\mu_{j})^{2}}{2 \Sigma_{j}}\right)
$$

高斯混合模型的参数估计就是对 $\theta=(\alpha_1,\dots, \alpha_k,; \mu_1,\dots, \mu_k;\Sigma)$ 的估计。如果用之前的极大似然估计试试，在样本集 $\equiv\left\{x^{(1)}, \cdots, x^{(m)}\right\}$下 ，对数似然如下：
$$
L(\alpha, \mu, \Sigma)=\sum_{i=1}^{m} \log p\left(x^{(i)} ; \alpha, \mu, \Sigma\right)=\sum_{i=1}^{m} \log \sum_{z^{(i)}=1}^{k} p\left(x^{(i)} | Z^{(i)} ; \mu, \Sigma\right) p\left(z^{(i)} ; \alpha\right)
$$
很明显系数 $\alpha$ 我们无从得知，这个极大似然估计也没法进行下去。

### 三、隐变量

**极大似然估计MLE、极大后验概率估计MAP和贝叶斯推断BI**能够直接应用都有个前提：概率模型的变量均为**观测变量**。这样就可以直接将给定数据带入来估计参数。

然而，有的时候变量中还包括**隐变量(hidden/latent variable)**，上述方法就不能直接使用了。

**隐变量解释如下：**

相对于观测变量，隐变量指不可观测的随机变量，或理论上可行但实际并没有给出具体数据的变量（数据不完整），亦或是抽象的变量，例如心理状态、行为等。

> 

### 四、EM 算法

**Expectation Maximization algorithm 期望极大算法** 用于对**含有隐变量**的概率模型参数进行极大似然估计MLE/极大后验概率估计MAP。

每次迭代分两步：

- E步：求期望
- M步：求极大