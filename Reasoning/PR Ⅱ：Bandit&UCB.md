#! https://zhuanlan.zhihu.com/p/214454791
![Image](https://pic4.zhimg.com/80/v2-2c9b08f09d7ae72ddac0bdec81835674.jpg)

# PR Reasoning Ⅱ：Bandit问题与 UCB / UCT

![Image](https://pic4.zhimg.com/80/v2-f78c27f4f5e8546f7a0f5bc2378424e5.png)

[TOC]

TODO：

- ~~UCB~~
- UCT
- 受限步长的Bandit



依旧延续前一章的风格，本章还是在关注强化学习的基础——Bandit问题。

> 部分译自 [The Multi-Armed Bandit Problem and Its Solutions - Lil'Log](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)

## Bandit —— 一种 tradeoff 的思考

Bandit问题是  Exploitation vs Exploration trade-off 的具象。探索与利用是一种深度与广度的平衡。关于这方面，深度强化学习实验室出过一期十分全面的文章。

[](https://mp.weixin.qq.com/s/FX-1IlIaFDLaQEVFN813jA)

本文就不用惯用的多臂老虎机举例了，**赌博是不可能赌博的，还是关心一下中午吃什么吧**。帮帮下面这个可怜的、有**选择困难症**的机器人选一选它的午餐吧。如果它每天都去左边，它将十分有信心能吃上五号电池午餐，但是可能会错过旁边的四川火锅。

![Image](https://pic4.zhimg.com/80/v2-ff9765b71e01708f2b46608ad92abf93.png)

如果它了解了关于环境的所有信息，甚至可以通过它那拥有GPT-3级别算力的大脑，轻易地暴力求解最佳策略，更不用说其他许多智能方法了。

问题在于**不完整的信息**或**部分可观测**：我们需要收集足够的信息以做出最佳的总体决策，同时要控制风险。

- **Exploitation 利用**，我们可以利用我们所知的最佳选择。
- **Exploration 探索**，我们冒着一些风险来收集关于未知选项的信息。

最好的长期策略可能涉及短期牺牲。一次探索尝试可能会完全失败，但它警告我们在将来不要过于频繁地采取这种行动。比方说，右边的店其实是螺蛳粉店。所以，**选择真是个技术活**。

### Multi-arm Bandit

Multi-arm Bandit 问题是一个经典问题，很好地证明了探索与利用的困境。想象一下，我们在一个面对多台老虎机的赌场中，每台老虎机都配置有未知的概率，您一次玩游戏可获得奖励的可能性。问题是：获得最高长期奖励的最佳策略是什么？

> 顺便提一句，多臂老虎机长这样，当然我也没见过。反正我们的目标是: **找到能有最大概率搞到钱的那台老虎机，搞最多的小钱钱。**
> ![Image](https://pic4.zhimg.com/80/v2-18679c41cb506500237b188c4c8241bc.png)

一般我们讨论Bandit问题，都是仅讨论进行可以无限试验的假设。
因为有限数量的试验的限制引入了一种新型的利用问题。例如，如果试验次数少于老虎机的数目，我们甚至无法尝试每台机器来估计奖励概率，因此我们必须明智地行事。

然而，受限尝试才应该是符合实际场景的问题。试想，**谁会带着无限的钱去赌场，只为了试出哪台老虎机有最小的几率吞掉你的钱？**
不过，我们还是要从理想情况开始。

一个很**朴素**的方法是：连续使用一台机器进行许多回合，以便最终根据大数定律估计“真实”的奖励概率。然而，这是非常浪费的，并且肯定不能保证最佳的长期回报。

### Definition

一个 Bernoulli multi-armed bandit 问题可以描述为⟨A，R⟩的元组，其中：
- 我们面前有 $K$ 个机器，其奖励概率分别为 ${\theta_1,\dots, \theta_K}$
- 在每个时间步 $t$ 处，我们在一台老虎机上执行动作a并获得奖励r。
- $A$是动作集，每个动作均指与一个老虎机的交互。动作a的值是期望的奖励，$Q(a)=E[r|a]=θ$。如果在时间步t处的动作在第i台机器上，则 $ Q(a_t)=θ_i$。
- $R$是奖励函数。在Bernoulli bandit的情况下，我们以随机方式观察到奖励r。在时间步骤$t$，$r_t = R(a_t)$可能以概率 $Q(a_t)$ 返回奖励1，否则返回0（这就是为啥叫**Bernoulli/二项**）。

这就是一个简单的MDP，没有状态 $S$。我们的目标还是最大化累积奖励，搞最多的小钱钱。如果我们知道具有最佳奖励的最佳行动，那么目标就是通过不选择最佳行动来最大程度地减少潜在的遗憾或损失。

### Regret —— 性能的衡量

最优动作$a^∗$的最优奖励概率$θ^∗$为：
$$
\theta^{*}=Q\left(a^{*}\right)=\max _{a \in \mathcal{A}} Q(a)=\max _{1 \leq i \leq K} \theta_{i}\\
$$
那么对应的损失函数或者称作 **Regret**（后悔没有选择最优动作）：
$$
\mathcal{L}_{T}=\mathbb{E}\left[\sum_{t=1}^{T}\left(\theta^{*}-Q\left(a_{t}\right)\right)\right]\\
$$
**那么这个问题该怎么解决呢？快让孩子吃上饭吧，都快饿死了。**



## $ε$-greedy

$ε$-greedy 算法在大多数情况下会采取最佳行动，但偶尔也会进行随机探索。根据过去的经验，通过对与我们到目前为止（直到当前时间步长t）观察到的目标动作a相关的奖励进行平均，来估算动作值：
$$
\hat {Q}_t(a)= \frac 
{1} {N_t(a)} \sum_{\tau = 1} ^ t r_\tau \mathbb {1} [a_\tau = a]\\
$$
其中 $\mathbb{1}$ 是二进制指示函数，$N_t(a)$ 是到目前为止已选择动作a的次数，$N_t(a) = \sum_{\tau=1}^t \mathbb{1}[a_\tau = a]$。
根据 $ε$-greedy 算法，我们以较小的概率 $ε$ 采取**随机动作**，否则（在大多数情况下，概率为 $1-\epsilon$ ）我们选择到目前为止已知动作中最佳的动作：
$$
\hat{a}^{*}_t = \arg\max_{a \in \mathcal{A}} \hat{Q}_t(a)\\
$$

强化学习用的都是这个，简易且有用。



## Upper Confidence Bounds (UCB)

随机探索使我们有机会尝试一些我们不太了解的选项。但是，由于随机性，**我们有可能会反复探索已经确认过的不良行为**。
为了避免这种低效率的探索，一种方法是及时降低参数ε，另一种方法是对不确定性较高的选择持乐观态度，因此偏爱我们尚未有确定价值估算值的动作。换句话说，我们**倾向于探索最有潜力拥有最大价值的情况**。

Upper Confidence Bounds（UCB）算法通过奖励值的信赖上界 $\hat{U}_{t}(a)$ 来度量这种潜力。$\hat{U}_{t}(a)$ 是尝试次数 $N_t(a)$ 的函数，$N_t(a)$ 越大，信赖上界越小。在UCB算法中，我们始终选择最贪婪的操作以最大化信赖上界：
$$
a_{t}^{U C B}=\operatorname{argmax}_{a \in \mathcal{A}} \hat{Q}_{t}(a)+\hat{U}_{t}(a)\\
$$
**那么问题来了，如何估计这个信赖上界？**

### Hoeffding’s Inequality

如果我们不想指定有关分布的任何先验知识，则可以从“**Hoeffding不等式**”获得帮助-该定理适用于任何有界分布。

令 $(X_1，\dots，X_t )$ 为 $i.i.d.$ (独立同分布) 的随机变量，它们均在区间$[0，1]$内。
样本平均值为 $\overline{X}_t = \frac{1}{t}\sum_{\tau=1}^t X_\tau$。

那么对于$u> 0$，我们有Hoeffding’s Inequality：
$$
\mathbb{P} [ \mathbb{E}[X] > \overline{X}_t + u] \leq e^{-2tu^2}\\
$$
转到Bandit问题，给定一个目标动作$a$，让我们考虑：

- $r_t(a)$ 为随机变量，
- $Q(a)$ 为真实均值，
- $\hat{Q}_t(a)$ 为样本均值，
- $u$ 为随机变量最高置信界限 $u=U_t(a)$

那么就有
$$
\mathbb{P} [ Q(a) > \hat{Q}_t(a) + U_t(a)] \leq e^{-2t{U_t(a)}^2}\\
$$
我们希望选择一个边界，以便真正的均值很有可能是样本均值+置信上限。那么$e^{-2t U_t(a)^2}$  应该是一个小概率。假设我们可以设置一个很小的阈值p：
$$
e^{-2t U_t(a)^2} = p \\
\text{ Thus, } U_t(a) = \sqrt{\frac{-\log p}{2 N_t(a)}}\\
$$


### UCB1

一种启发式方法是及时降低阈值p，因为我们希望通过观察到更多的奖励做出更加自信的边界估计。设置 $p = t ^ {-4}$，我们得到UCB1算法：
$$
U_t(a) = \sqrt{\frac{2 \log t}{N_t(a)}}\\  a^{UCB1}_t = \arg\max_{a \in \mathcal{A}} Q(a) + \sqrt{\frac{2 \log t}{N_t(a)}}\\
$$

很明显，这符合我们的预想。$N_t(a)$ 越大，信赖上界越小。

信赖域的上下界可以理解为方差，$Q(a)$ 则是对应的均值。

- 对 $a_i$ 的探索的次数越多，我们越确定这家餐厅的口味，对他们厨师水平的不确定性就越小。
- 相反，随着对 $a_i$ 的探索，**我们对未探索/缺乏探索的 $a_j$ 更加感兴趣了**，这反映在 t 不断增加，而 $N_t(a_j)$ 不变，$U_t(a_j)$ 就会随之变大。

### Bayesian UCB

在UCB或UCB1算法中，我们不假设任何先验的奖励分配。因此，我们必须依靠Hoeffding 的不等式进行非常概括的估计。如果我们能够预先知道分布，则可以进行更好的边界估计。

例如，如果我们期望每个老虎机的平均奖励为 Gaussian 函数，如下图所示，则可以通过将 $\hat{U}_t(a)$设置为**两倍标准偏差 (2$\sigma$)**来将上限设置为95％置信区间。

![Gaussian prior](https://lilianweng.github.io/lil-log/assets/images/bern_UCB.png)

可以查看 Lilian Weng 的 [UCB1](https://github.com/lilianweng/multi-armed-bandit/blob/master/solvers.py#L76)和 [Bayesian UCB](https://github.com/lilianweng/multi-armed-bandit/blob/master/solvers.py#L99) 实现



## Thompson Sampling

在每个时间步长，我们都希望根据a最优的概率来选择动作a：
$$
\begin{aligned} \pi(a \; \vert \; h_t) &= \mathbb{P} [ Q(a) > Q(a'), \forall a' \neq a \; \vert \; h_t] \\ &= \mathbb{E}_{\mathcal{R} \vert h_t} [ \mathbb{1}(a = \arg\max_{a \in \mathcal{A}} Q(a)) ] \end{aligned}\\
$$
where $\pi(a \; \vert \; h_t)$ is the probability of taking action a given the history $h_t$.

对于 Bernoulli bandit，很自然地假设 $Q(a)$ 遵循Beta分布，因为 $Q(a)$ 本质上是伯努利分布中的成功概率 $\theta$。 $\text{Beta}(\alpha, \beta)$ 的值在区间[0，1]之内； α和β分别对应于我们获得奖励的成功或失败的次数。

首先，让我们基于对每个动作的一些先验知识或信念来初始化Beta参数α和β。例如：

- α= 1和β= 1；我们预计奖励的可能性为50％，但我们不是很自信。 
- α= 1000和β= 9000；我们坚信奖励的可能性是10％。

在每个时间t，我们从每个动作的先前的分布 $\text{Beta}(\alpha_i, \beta_i)$ 采样一个预期的回报，$\tilde{Q}(a)$，在样本中选择最佳操作：$a^{TS}_t = \arg\max_{a \in \mathcal{A}} \tilde{Q}(a)$。在观察到真实的奖励之后，我们可以相应地更新Beta分布，这实际上是在进行贝叶斯推理，以使用已知先验和获得采样数据的可能性来计算后验。
$$
\begin{aligned} \alpha_i & \leftarrow \alpha_i + r_t \mathbb{1}[a^{TS}_t = a_i] \\ \beta_i & \leftarrow \beta_i + (1-r_t) \mathbb{1}[a^{TS}_t = a_i] \end{aligned}\\
$$



## UCT —— AlphaGo的成功之秘

## Reference
1. [The Multi-Armed Bandit Problem and Its Solutions](https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html)