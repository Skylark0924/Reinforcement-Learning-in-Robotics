## 【综述】：使用Policy Search算法在少量尝试下学习机器人控制

#### A Survey on Policy Search Algorithms for Learning Robot Controllers in a Handful of Trials

2019 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8944013&tag=1) | IEEE Transactions on RobotiTrialscs

*Konstantinos Chatzilygeroudis , Vassilis Vassiliades , Freek Stulp , Sylvain Calinon ,  Jean-Baptiste Mouret*

> 这是审稿人给我推荐的一篇综述，后悔没有早点看到，真是醍醐灌顶。让我更加确信**在机器人强化学习中恰当地融合先验**是**提高学习效率、数据利用率**的正确路径。

#### 由于文章内容过于详实，建议大家 **点赞+关注+收藏** 再看哦！！

### 1 Introduction

机器人在现实中的问题：

- 难以GPU加速
- 难以在大型集群上并行 (即使Google花了大价钱试了并行机械臂 [Learning hand-eye coordination for robotic grasping with deep learning and large- scale data collection](https://journals.sagepub.com/doi/pdf/10.1177/0278364917710318)，这种方法也并不具有泛化性)

我们十分期望RL所带来的自适应的优势，可以让机器人在面对新事物时，只需要几次尝试，或者人形机器人在一条腿损坏的情况下，能学着如何跛行。

相较于 data-efficient RL 的概念，本文提出了 **micro-data RL (MDRL)**。区别在于 data-efficiency 表示数据量与任务复杂度之间的比率，或与上一代算法相比用的步数少了，但仍可能是millions of time-steps。所以**micro-data RL** 的定义更加直接，旨在使机器人能够在很少的尝试次数下就能学习到控制策略。由于机器人大都使用policy search的方法，因此也称为**micro-data policy search (MDPS)**。

现有的MDPS一般有两种做法：

1. **利用先验知识：**类似 Imitation Learning & RLfD
2. **学习替代模型(surrogate models)：** Model-based RL的做法。

二者也可兼而有之。

本综述对于RL中**先验知识的引入方式**分为以下三种：

![image-20200528152451409](./A_survey_on_PS_for_Learning_Robot_controllers_in_a_Handful_of_trials.assets\image-20200528152451409.png)

> 引文对应文章名见底部。



### 2 Using Priors on the policy parameters/representation

策略参数/表示代表着我们如何从state映射到action。我们可以利用先验来帮助我们设计初始化参数或构建简单的策略表示方法。当我们设计**策略参数/策略表示**的时候，要兼顾以下两点：

- 策略要具有表达力(Expressive)，可以理解为泛化性强、拟合能力强；
- 可高效搜索的参数空间 dim($\theta$)，即策略的参数空间要尽可能小。

最优的期望回报与参数 $\theta$ 下的期望回报的差：
$$
J_{\zeta}\left(\pi_{\zeta}^{*}\right)-\max _{\theta} J_{\zeta}(\theta)<\delta
$$

**荐读：**

[Robot programming by demonstration](https://infoscience.epfl.ch/record/114050) 

[An Algorithmic Perspective on Imitation Learning](https://arxiv.org/abs/1811.06711)

#### 2.1 Hand-Designed policies

**思想：**为了降低参数维度，可以用先验知识**手动给出一个压缩维度的策略表示**，再用PS算法在这个小参数空间里找到最优参数。

**例子：** [Learning Ball Acquisition on a Physical Robot](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ISRA2004-chinpinch.pdf)

**缺点：**

1. 限制参数维度会导致高 $\delta$。
2. 不具有泛化性。



#### 2.2 Policies as Function Approximators

**思想：**先验知识确定 linear policies, RBF, and neural networks (NN) 等拟合函数的基函数。即使用expert demonstrations对网络做与训练得到初始的参数，相当于模仿学习的pre-train思想。函数逼近器可用于生成单个估计（与统计中的一阶矩相对应），但也可以扩展到更高阶矩。通常，将其扩展到二阶矩可以使系统获得有关我们可以用来完成任务的变化的信息，以及以协方差形式显示的不同策略参数之间的协同作用。这通常要花更大的代价去学习，但是学习到的representation通常可以更具表现力，有助于适应和推广。

**例子：** [Learning parametric dynamic movement primitives from multiple demonstrations](https://www.sciencedirect.com/science/article/pii/S0893608011000566)

**缺点：**基函数数量少可以加速学习，但是也会增大 $\delta$



#### 2.3 Trajectory-Based Policies

**思想：**两种思想

1. 一种对轨迹进行编码的方法是将策略定义为一系列航路点。

**例子：**35-48



#### 2.4 Learning the Controller

**思想：**policy生成了参考轨迹，控制器要按照这个目标轨迹来控制电机。我们可以把控制器（PID或linear quadratic tracking）的参数也包含在要学习的参数 $\theta$ 中，可以提供一种协调电机命令对扰动做出反应的方法。

**例子：**49-52



#### 2.5 Learning the Policy Representation

**思想：**同时学习策略表示及其参数

**例子：**53-55

**缺点：**不适用于MDRL，需要大量的rollouts



#### 2.6 Hierarchical and Symbolic Policy Representations

**思想：**为了泛化，使用由门控网络和多个子策略组成的分层策略，并引入基于熵的约束，以确保agent找到具有不同子策略的不同解决方案。在期望最大化过程中，这些子策略被视为潜在变量，从而允许在子策略之间分配更新信息。层次结构的较高层可以用符号表示代替。

**例子：**

[Rl-tops: An architecture for mod- ularity and re-use in reinforcement learning](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.37.491&rep=rep1&type=pdf) 

[Exploration in relational do- mains for model-based reinforcement learning](http://www.jmlr.org/papers/v13/lang12a.html)

[PEORL: Integrating Symbolic Planning and Hierarchical Reinforcement Learning for Robust Decision-Making](https://arxiv.org/abs/1804.07779)



#### 2.7 Initialization With Demonstrations/Imitation Learning

**思想：**通过模仿学习初始化策略参数，使其离最优参数较近，确保其能够收敛到全局最优。

例子：

[Robot programming by demonstration](https://infoscience.epfl.ch/record/114050) 

[An Algorithmic Perspective on Imitation Learning](https://arxiv.org/abs/1811.06711)



### 3 Learning Models of the Expected Return

上一章主要讲了如何初始化参数，那么如何在初始化参数上进行参数更新？本章思路旨在**学习一个期望回报关于策略参数的模型** $J(\theta)$ ，并从中选取最优的策略参数迭代方向。

**荐读：** 

[Robots that can adapt like animals](https://www.nature.com/articles/nature14422?rel=mas)

[Automatic gait optimization with Gaussian process regression](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-152.pdf)

#### 3.1 BO: Active Learning of Policy Parameters

**思路：**BO 贝叶斯优化是最具代表性的算法，包含两部分：

- 期望回报的model；
- 一个获取函数 (acquisition function)，用于利用model确定参数空间中每一点的效果。

![image-20200528201113694](./A_survey_on_PS_for_Learning_Robot_controllers_in_a_Handful_of_trials.assets\image-20200528201113694.png)

**建模方式：**高斯过程回归 GP regression 是最常用于构建 surrogate model 的算法。

[高斯过程](https://zhuanlan.zhihu.com/p/139478368)

**获取函数：**

- Probability of Improvement (PI)：定义了参数空间上一个新点期望回报优于当前最优参数的可能性。结合GPs的surrogate model，可以得到解析解。**缺点：**只是纯粹的exploitation，改进版可以引入折衷系数。
- Expected Improvement (EI)：PI的扩展，计算期望回报与当前参数对应回报的差。结合GPs可算。
- Upper Confidence Bound (UCB)：利用参数的期望方差进行简单的线性计算，$\mathrm{UCB}(\boldsymbol{\theta})=\mu(\boldsymbol{\theta})+\alpha \sigma(\boldsymbol{\theta})$。结合GPs可算。**缺点：**超参 $\alpha$ 与核函数不易确定，GP-UCB算法可以自动调整超参，并为算法的regret限提供了一些理论上的保证。
- Entropy Search (ES)：为了最大地较少每一步最优参数位置的不确定性，通过最大位置上分布的熵来量化这种不确定性。

以上各有千秋。

由于BO没有被用来学习动力学模型，而是期望回报模型，因此可用于复杂的、高维的机器人系统。例如，100步内学到四足机器人[Automatic gait optimization with Gaussian process regression](https://www.aaai.org/Papers/IJCAI/2007/IJCAI07-152.pdf)策略，30步内学到一个 pocket-sized, vibrating soft tensegrity robot [Adaptive and resilient soft tensegrity robots](https://www.liebertpub.com/doi/full/10.1089/soro.2017.0066) 的策略。

**缺点：**不易扩展到在高维策略空间，随着维数的增加，对期望回报的建模变得越来越困难。所以一般这么做都需要先用第二章知识，设计一个低维的策略空间。



#### 3.2 BO With Priors: Using Non-Zero Mean Functions as a Starting Point for the Search Process

**思路：**BO的优势在于，可以利用先验。**Intelligent trial and error (IT&E)** [Robots that can adapt like animals](https://www.nature.com/articles/nature14422?rel=mas) 算法，使用MAP-Elites，渐进照明（也称为质量多样性）算法来创建约15000个高性能策略的库，并将它们存储在低维度中
地图（例如6维，而策略空间为36维）。当机器人需要适应未知环境时，BO算法会在低维地图中搜索最佳策略，并将地图中存储的奖励用作GP的均值函数。

该算法允许六足步行机器人在不到2分钟的时间内（少于12次试验）适应多种损伤情况（例如，腿部缺失或腿部缩短），而它**使用完整机器人的模拟器来生成先验**。

**使用仿真器先验的GP：**
$$
\mu\left(\boldsymbol{\theta}_{*}\right)=R_{m}\left(\boldsymbol{\theta}_{*}\right)+\boldsymbol{k}^{T} K^{-1}\left(D_{1: t}-R_{m}\left(\boldsymbol{\theta}_{1: t}\right)\right)
$$
$R_{m}\left(\boldsymbol{\theta}_{*}\right)$是从模拟器得到的先验，$R_{m}\left(\boldsymbol{\theta}_{1: t}\right)$是观测数据。这个公式允许先验和观测平滑地结合在一起。如果可观测，那么就修正先验地预测；在远离real-world数据的区域，采用先验。

此外，还可以使用模拟器来学习GP的核函数。

**多信息来源：**

[Virtual vs. real: Trading off simulations and physical experiments in reinforcement learning with Bayesian optimization](https://ieeexplore.ieee.org/abstract/document/7989186) 提出了multifidelity entropy search (**MF-ES**)算法，他们的BO可以有多个信息源（real-world和仿真器），并用上一节的ES熵方法来度量来自real-world和仿真器的内容。因此，该算法可以自动决定是评估廉价但不准确的仿真，还是执行昂贵且精确的真实实验。

[Bayesian optimiza- tion with automatic prior selection for data-efficient direct policy search](https://ieeexplore.ieee.org/abstract/document/8463197/) 将BO与多个信息源（或先验信息）结合起来。他们为BO定义了一个新的获取函数 most likely expected improvement (**MLEI**)。 MLEI尝试在先验的可能性与高性能解决方案的潜力之间取得适当的平衡。即，根据不准确的模型获得的良好EI应该被忽略。相反，具有低EI的可能模型可能过于悲观（“无用”），无济于事。一个“可能足够大”的模型，可以让我们期望一些好的改进可能是找到目标函数最大值的最有帮助的方法。 MLEI定义如下：
$$
\begin{array}{l}
\operatorname{EIP}(\boldsymbol{\theta}, \mathcal{P})=\operatorname{EI}(\boldsymbol{\theta}) \times p\left(\hat{J}\left(\boldsymbol{\theta}_{1 . . t}\right) | \boldsymbol{\theta}_{1 . . t}, \mathcal{P}\left(\boldsymbol{\theta}_{1 . . t}\right)\right) \\
\operatorname{MLEI}\left(\boldsymbol{\theta}, \mathcal{P}_{1}, \ldots, \mathcal{P}_{m}\right)=\max _{\boldsymbol{p} \in \operatorname{P}_{1}, \ldots, \mathcal{P}_{m}} \operatorname{EIP}(\boldsymbol{\theta}, \boldsymbol{p})
\end{array}
$$
**安全意识方法 Safety-Aware Approaches：** 

利用先验告知机器人那些区域会导致自身受损，[Bayesian optimization with inequality constraints](http://proceedings.mlr.press/v32/gardner14.pdf)

[Safe controller op- timization for quadrotors with gaussian processes](https://ieeexplore.ieee.org/abstract/document/7487170) 引入了SafeOpt，这是一个BO过程，它仅在搜索空间的安全区域内通过在勘探与开发之间进行权衡来自动调整控制器参数。



### 4 Learning Models of the Dynamics

利用轨迹数据对机器人动力学进行建模，也就是通常意义的model-based RL算法。学习一个 $\hat{\boldsymbol{x}}_{t+1}=\hat{f}\left(\boldsymbol{x}_{t}, \boldsymbol{u}_{t}\right)$model，并可利用它来估计期望回报 $\hat{J}\left(\boldsymbol{\theta} | \boldsymbol{\tau}_{1}, \ldots, \boldsymbol{\tau}_{N}\right)$

**荐读：**

[Using parameterized black-box priors to scale up model-based policy search for robotics](https://ieeexplore.ieee.org/abstract/document/8461083) 

[Gaussian Processes for Data-Efficient Learning in Robotics and Control](https://ieeexplore.ieee.org/abstract/document/6654139) 

[Black-box data-efficient policy search for robotics](https://ieeexplore.ieee.org/abstract/document/8202137)

#### 4.1 Model-Based PS: Alternating Between Updating the Model and Learning a Policy in the Model

![image-20200528213920267](./A_survey_on_PS_for_Learning_Robot_controllers_in_a_Handful_of_trials.assets\image-20200528213920267.png)

**4.1.1 如何学习模型：**

对模型的建模包括确定性（NN）和概率性（GPs）两种。

概率性建模通常比确定性高效，因其提供了不确定性信息，故可以找到更鲁棒的控制策略。PILCO和MBPO等就是这么做的。

[Model-Based RL Ⅲ: 从源码读懂PILCO](https://zhuanlan.zhihu.com/p/138337983)

[Model-Based RL Ⅱ: MBPO原理解读](https://zhuanlan.zhihu.com/p/105645139)

1. 使用 **least-square conditional density estimation (最小均方条件密度估计)** 来估计转移概率 $p\left(\boldsymbol{x}_{t+1} | \boldsymbol{x}_{t}, \boldsymbol{u}_{t}\right)$，来取代学习一个模型，可以避免GPs的一些缺点。[Least-squares conditional density estimation](https://search.ieice.org/bin/summary.php?id=e93-d_3_583)

2. 使用 **local linear models (局部线性模型)** 学习，仅在一个策略可以驱动系统的区域中进行训练的模型。Guided PS (GPS) 可以在动力学未知的情况下高效地学习 2-D walking以及peg-in-the-hole tasks。

3. **Ensemble + Bayesian NN (BNN)** 取代 GPs，(MBPO的做法)，因为BNN在样本量增加的情况下可扩展性更好。 [“Deep reinforcement learning in a handful of trials using probabilistic dynamics models]()

   [Learning to control a 6-degree-of-freedom walking robot](https://ieeexplore.ieee.org/abstract/document/4400335) 混合了MPC并在half-cheetah上获得了benchmark。

**4.1.2 如何使用model进行Long-Term 预测：**

model-based PS 算法通常被用来预测当前决策的长期影响，分为两类：

- 随机长期预测：通过抽样的均值；
- 确定性长期预测：通过确定性推断技术。

**随机长期预测：**

如果是概率化建模，我们就可以从model里得到 $\hat{p}\left(\boldsymbol{x}_{t+1} | \boldsymbol{x}_{t}, \boldsymbol{u}_{t}\right), \hat{p}\left(\hat{r}_{t+1} | \boldsymbol{x}_{t}, \boldsymbol{u}_{t}, \boldsymbol{x}_{t+1}\right)$ ，我们可以从当前状态rollout一个trajectory：$\left\{\boldsymbol{\tau}=\left(\boldsymbol{x}_{0}, \boldsymbol{u}_{0}, \boldsymbol{x}_{1}, \boldsymbol{u}_{1}, \ldots, \boldsymbol{x}_{T}\right), \boldsymbol{r}=\left(\hat{r}_{1}, \hat{r}_{2}, \ldots, \hat{r}_{T}\right)\right\}$。很明显，这样一条轨迹是从状态分布中**采样**得到的。即使同样的策略，再做一次rollout得到的结果肯定是不一样的。毕竟，策略、模型、初始状态都是概率的、随机的。那么我们如何利用model rollout来评估策略的性能呢？

1. **Monte-Carlo & PEGASUS policy evaluation：** 一个很直接的想法就是同一个策略生成多条轨迹，求平均的期望回报$\tilde{\hat{J}}(\boldsymbol{\theta})=\frac{1}{m} \sum_{i=1}^{m} \hat{R}_{i}\left(\boldsymbol{\tau}^{i}\right)$（蒙特卡洛法）。更高效的方法是[PEGASUS](https://arxiv.org/abs/1301.3878)采样，固定每一步的random seed，使其采样生成的轨迹相同。这样做降低了采样方差，且优化模型的semi-stochastic版本相当于优化实际模型。**多篇文章证明sampling-based策略评估方法虽然存在采样方差大的问题，但是也明显优于确定性推断方法，且可以利用GPU并行采样。**
2. **Probabilistic Inference for Particle-Based PS (PIPPS):** 结合 Reparameterization gradients (RP) and the Likelihood ratio gradients (LR)，[文章](https://arxiv.org/abs/1902.01240)证明 LR 及其混合版本不受梯度爆炸的影响，但却需要大量的rollouts来精确估计梯度。

**确定性长期预测：**

用lineariza- tion, sigma-point methods, or moment matching等推断技术，将概率模型近似成高斯分布。

<img src="./A_survey_on_PS_for_Learning_Robot_controllers_in_a_Handful_of_trials.assets\image-20200529151707215.png" alt="image-20200529151707215" style="zoom: 67%;" />

上式只在model是线性时可计算，即 $\hat{p}\left(\boldsymbol{x}_{t+1}\right)$ 为高斯分布。PILCO 就是使用moment matching，这是预测分布的最佳单模态近似值，因为它最大限度地减少了真实预测分布和单模态近似之间的 KL 散度。

使用确定性推断的好处在于，在预测中表现出低方差。允许分析梯度计算，因此我们可以利用基于梯度的优化。

**Policy Evaluation as a Noisy Observation：**

第三种思路 [Black-DROPS](https://ieeexplore.ieee.org/abstract/document/8202137) 算法，将rollouts作为一种带噪声的观测，来度量 $G(\theta)=\hat{J}(\theta)+N(\theta)$。这个式子可以看作受到噪声干扰的期望回报公式。 当噪声是常量时，对G求期望相当于对 J 求期望。Black-DROPS 算法利用 CMA-ES 的最新变体（即优化噪声和黑盒函数最成功的算法之一），将随机扰动与重新评估相结合，实现不确定处理，以及重新启动策略，以便更好地探索。

Black-DROPS 和 PILCO 一样是十分经典的 data-efficiency 算法，推荐阅读。



#### 4.2 Using Priors on the Dynamics Reducing

重头戏来了，先验结合model-based，大都是利用先验来初始化model参数。

**4.2.1 GPs With Priors for Dynamical Models**

![image-20200529154918336](./A_survey_on_PS_for_Learning_Robot_controllers_in_a_Handful_of_trials.assets\image-20200529154918336.png)

[PILCO with priors](https://ieeexplore.ieee.org/abstract/document/7139550)使用模拟器创建一个GP先验，然后再用PS改进。

[PI-REM](https://ieeexplore.ieee.org/abstract/document/8206343) 利用动力学先验的分析方程，并尝试使用稍微修改的 PILCO PS 程序主动使实际试验尽可能接近模拟试验（即参考轨迹）

[Black-DROPS with priors](https://ieeexplore.ieee.org/abstract/document/8461083) 提出了一种新的GP学习方案，将模型识别和非参数模型学习（称为GP-MI）相结合，然后使用Black-DROPS执行PS。 [Black-DROPS with GP-MI](https://ieeexplore.ieee.org/abstract/document/478951)  胜过 Black-DROPS, PILCO, PILCO with priors, Black-DROPSwith fixed priors (i.e., this should be similar to PI-REM) , and IT&E.



### 5 Other Approaches 

**荐读：**

[Closing the sim-to-real loop: Adapting simulation randomization with real world experience](https://ieeexplore.ieee.org/abstract/document/8793789) 
[Learning to Adapt inDynamic, Real-World Environments through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347) 
[Meta Reinforce- ment Learning with Latent Variable Gaussian Processes](https://arxiv.org/abs/1803.07551)

#### 5.1 Guided Policy Search GPS

具有未知动力学的GPS [Learning neural network policies with guided policy search under unknown dynamics](http://papers.nips.cc/paper/5444-learning-neural-network-policies-with-guided-policy-search-under-unknown-dynamics) [End-to-end training of deep visuomotor policies](https://dl.acm.org/doi/abs/10.5555/2946645.2946684) 是一种有点混合的方法，它结合了局部轨迹优化（直接发生在实际系统上），学习动力学model（参见V-A1节），以及间接PS，它试图用一个大的NN策略（使用监督学习）来近似本地控制器。它由两个循环构成：

- 外循环：在real-world上执行本地线性高斯策略，收集数据，训练动力学模型
- 内循环：在优化本地线性高斯策略（使用轨迹优化和拟合动力学模型）和优化全局策略以匹配所有本地策略之间交替（通过监督学习，不使用学习模型）

**优点：**可处理高维

**缺点：**不是最data-efficiency的

#### 5.2 Transferability Approaches

可迁移性方法 [The transferability approach: Crossing the reality gap in evolutionary robotics](https://ieeexplore.ieee.org/abstract/document/6151107) 的主要假设是，物理模拟器对于某些策略（例如静态步态）是准确的，对于其他一些策略（例如，高度动态步态）是不准确的。因此，如果搜索被限制在精确模拟的策略，则可以在模拟中学习。由于目前没有模拟器提供其准确性的估计值，因此可转移性方法的关键思想是学习**可迁移函数的模型**，该模型预测给定策略参数或仿真轨迹的模拟器的准确性。此函数通常比期望回报更容易学习，因为这本质上是一个分类问题（而不是回归）。此外，模型中的小错误通常没有什么后果，因为搜索主要是由模拟中的预期回报（而不是由可转移性优化）驱动的。

**缺点：** 它只能找到在模拟和现实中执行类似的策略



#### 5.3 Simulation-to-Reality and Meta-Learning Approaches 

Meta-learning：找到对任务（或环境）分布的鲁棒性策略

[Meta-Learning: An Introduction Ⅰ](https://zhuanlan.zhihu.com/p/99730942)

Sim-2-Real：利用参数化模拟器来学习一个策略，可以有效地迁移到real-world系统上，分为两类：

- domain randomization：引入视觉差异，灯光、场景等；
- different dynamics properties：引入不同的动力学属性。



### 6 Challenges and Frontier

1. 维度上的可扩展性
2. 先验
3. 泛化性及鲁棒性
4. Planning、MPC 和 PS 之间的相互作用
5. 计算时长



### 7 Conclusion

1. **Low-DOF Robots: ** 建议使用PILCO、Black-DROPS系列
2. **High-DOF Robots:**  BO方法可以平衡计算时时长与收敛性。如果有先验或模拟器，推荐 IT&E 和 MF-ES
3. **Complex Robots: ** Black-DROPS with GP-MI  and VGMI
4. **Raw Observations:** 如果是视觉信息输入， 建议使用 SimTo- Real methods combined with online adaptation (e.g., SimOpt) 

### Reference for Fig. 2

> [15] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, “Robots that can adapt like animals,” Nature, vol. 521, no. 7553, pp. 503–507, 2015.
> [16] R. Pautrat, K. Chatzilygeroudis, and J.-B. Mouret, “Bayesian optimization with automatic prior selection for data-efficient direct policy search,” in Proc. IEEE Int. Conf. Robot. Autom., 2018, pp. 7571–7578.
> [17] K. Chatzilygeroudis and J.-B. Mouret, “Using parameterized black-box priors to scale up model-based policy search for robotics,” in Proc. IEEE Int. Conf. Robot. Autom., 2018, pp. 5121–5128.
> [19] P. Abbeel, M. Quigley, and A. Y. Ng, “Using inaccurate models in reinforcement learning,” in Proc. Int. Conf. Mach. Learn., 2006, pp. 1–8
> [32] F. Guenter, M. Hersch, S. Calinon, and A. Billard, “Reinforcement learning for imitating constrained reaching movements,” Adv. Robot., vol. 21, pp. 1521–1544, 2007.
> [35] S. M. Khansari-Zadeh and A. Billard, “Learning stable nonlinear dynam- ical systems with gaussian mixture models,” IEEETrans. Robot., vol. 27, no. 5, pp. 943–957, Oct. 2011
> [43] F. Stulp and O. Sigaud, “Policy improvement: Between black-box opti- mization and episodic reinforcement learning,” in Proc. Journées Fran- cophones Planification, Décision, et Apprentissage pour la conduite de systèmes, 2013.
> [44] F. Stulp, E. Theodorou, and S. Schaal, “Reinforcement learning with sequences of motion primitives for robust manipulation,” IEEE Trans. Robot., vol. 28, no. 6, pp. 1360–1370, Dec. 2012.
> [81] R. Antonova, A. Rai, and C. G. Atkeson, “Sample efficient optimization for learning controllers for bipedal locomotion,” in Proc. Humanoids, 2016, pp. 22–28.
> [82] R. Antonova, A. Rai, and C. G. Atkeson, “Deep kernels for optimizing locomotion controllers,” in Proc. CoRL, 2017, pp. 47–56.
> [85] A. Wilson, A. Fern, and P. Tadepalli, “Using trajectory data to improve Bayesian optimization for reinforcement learning,” J. Mach. Learn. Res., vol. 15, no. 1, pp. 253–282, 2014.
> [86] R. Lober, V. Padois, and O. Sigaud, “Efficient reinforcement learning for humanoidwhole-body control,” in Proc. Humanoids, 2016, pp. 253–282.
> [90] V. Papaspyros, K. Chatzilygeroudis, V. Vassiliades, and J.-B. Mouret, “Safety-aware robot damage recovery using constrained Bayesian opti- mization and simulated priors,” in Proc. Int.Workshop “BayesianOptim.: Black-box Optim. Beyond” at Neural Inf. Process. Syst., 2016.
> [92] F. Berkenkamp, A. P. Schoellig, and A. Krause, “Safe controller op- timization for quadrotors with gaussian processes,” in Proc. IEEE Int. Conf. Robot. Autom., 2016, pp. 491–496.
> 3876–3881.
> [127] M.Cutler and J. P.How, “Efficient reinforcement learning for robotsusing informative simulated priors,” in Proc. IEEE Int. Conf. Robot. Autom., 2015, pp. 2605–2612.
> [128] M. Saveriano, Y. Yin, P. Falco, and D. Lee, “Data-efficient control policy search using residual dynamics learning,” inProc. Int.Conf. Intell. Robots Syst., 2017, pp. 4709–4715.
> [132] S. Zhu, A. Kimmel, K. E. Bekris, and A. Boularias, “Fast model identifi- cation via physics engines for data-efficient policy search,” in Proc. Int. Joint Conf. Artif. Intell., 2018, pp. 3249–3256.
> [133] J. Bongard, V. Zykov, and H. Lipson, “Resilient machines through continuous self-modeling,” Science, vol. 314, pp. 1118–1121, 2006.
> [141] S. James, A. J. Davison, and E. Johns, “Transferring end-to-end visuo- motor control from simulation to real world for a multi-stage task,” in Proc. CoRL, 2017, pp. 334–343.
> [144] Y. Chebotar et al., “Closing the sim-to-real loop: Adapting simulation randomization with real world experience,” in Proc. IEEE Int. Conf. Robot. Autom., 2018, pp. 8973–8979.
> [147] M. Andrychowicz et al., “Hindsight experience replay,” in Proc. Conf. Neural Inf. Process. Syst., 2017, pp. 5048–5058

