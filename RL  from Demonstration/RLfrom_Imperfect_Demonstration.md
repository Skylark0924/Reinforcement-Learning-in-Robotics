## Reinforcement Learning from Imperfect Demonstrations under Soft Expert Guidance

2019 | [Paper](https://arxiv.org/pdf/1911.07109.pdf) | Tsinghua University 

*Mingxuan Jing, Xiaojian Ma, Wenbing Huang, Fuchun Sun, Chao Yang, Bin Fang, Huaping Liu*

### 1 Abstract 

上一章的 Deep Q-learning from Demonstrations 单纯地将示范数据做为一种数据增广方式或者pre-training，这并没有使其得到充分利用。近期一些morden **RLfD (Reinforcement Learning from Demonstrations)** 方法吸取了Imitation Learning地思想，鼓励agent在环境反馈有限的情况下去模仿示范动作。Specifically，通过在reward中引入示范引导项来强迫agent像expert一样探索。

然而，上述的 RLfD 方法常常依赖于expert示范数据的准确性，但是这些数据存在不准确性，有时就会导致policy收敛到局部最优点。

- **quality**: 这些数据中有可能带有noise；
- **amount**: 数据量不足。

<img src="./RLfrom_Imperfect_Demonstration.assets\image-20200524205516760.png" alt="image-20200524205516760" style="zoom:50%;" />

左图代表了其他的RLfD方法，它们会把与expert behavior 之间的散度作为罚项，使其尽量收敛在expert policy的附近，如果expert不完美，就不能保证得到较好的agent policy，甚至反而会误导agent。

本文提出了右图的方法，让expert以一种**soft**方式的引领探索。**soft这个词，与softmax，soft actor-critic， soft update, soft Q leaning 同义，都是使条件不那么严格的一种做法。**右图中expert的行为变成了一个范围约束，如果agent的policy在这个范围内，那么它只受到环境的影响，而与示范数据无关。因此，RLfD问题转化为一种**有约束的策略优化问题**。



### 2 Preliminaries

**Occupancy measure（占用度量）**：表示在策略 $\pi$ 下，状态分布或动作-状态对分布的密度
$$
\begin{array}{l}
\rho_{\pi}(s) \triangleq \sum_{t=0}^{\infty} \gamma^{t} P\left(s_{t}=s | \pi\right) \\
\rho_{\pi}(s, a) \triangleq \rho_{\pi}(s) \pi(a | s)
\end{array}
$$
**将先前使用罚项的RLfD方法公式化**
$$
\min _{\pi} \mathcal{L}_{\pi}=-\eta(\pi)+\lambda \cdot \mathbb{D}\left[\rho_{\pi}(s, a) \| \rho_{E}(s, a)\right]
$$
其中，$\eta(\pi)=\mathbb{E}_{\pi}\left[\sum_{t}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right)\right]$ 代表累计回报



### 3 Methodology

#### 3.1 Proposed method

首先给出**Imperfect Expert Policy**定义：

<img src="./RLfrom_Imperfect_Demonstration.assets\image-20200524213207117.png" alt="image-20200524213207117" style="zoom:50%;" />

很显然，这导致 $\mathcal{L}_{\pi_\theta}$ 的计算方式在不完美的专家策略上无法收敛到最优策略。（具体定理见原文）

给出**约束策略优化**的公式：
$$
\begin{aligned}
\theta_{k+1}=\arg \max_\theta & \eta\left(\pi_{\theta_{k}}\right) \\
\text { s.t. } & \mathbb{D}\left[\rho_{\pi_{\theta_{k}}}(s, a) \| \rho_{\pi_{\theta-}}(s, a)\right] \leqslant d_{k} \\
& \mathbb{D}_{\text {KL }}\left[\pi_{\theta_{k}}(a | s) \| \pi_{\theta_{k+1}}(a | s)\right] \leqslant \delta
\end{aligned}
$$
这里的 $d_k$ 是通过**退火策略**自动更新的，$d_{k+1} \leftarrow d_{k}+d_{k} \cdot \epsilon$， $\epsilon$ 是退火系数。

（大家可详细地看一下文中是怎么分别解决 quality 和 amount 两个issues的）

#### 3.2 计算约束策略优化的trick

在两个约束的情况下，找到一个**可行的且具有高维扩展性的**方法很有挑战。

本文提出一种**近似解法**，在每次优化时，**将策略线性化**。

<img src="./RLfrom_Imperfect_Demonstration.assets\image-20200524215212011.png" alt="image-20200524215212011" style="zoom: 50%;" />

由此，就将其转化为凸优化问题，Hessian阵总是半正定，然后使用拉格朗日乘子法转化为无约束优化问题
$$
\max _{\lambda \geq 0 \atop \nu \geq 0}-\frac{1}{2 \lambda}\left(g^{T} u+2 \nu b^{T} u+\nu^{2} b^{T} r\right)-\nu c-\lambda \delta
$$
其中，$u=H^{-1} g, r=H^{-1} b, c=d_{k}-d_{\theta_{k}}$

最优解就可以表示为：
$$
\theta_{k+1}^{\star}=\theta_{k}-\frac{1}{\lambda^{\star}}\left(u+r \nu^{\star}\right)
$$


#### 3.3 Implementation

1. 使用无参距离度量方式 MMD 来度量expert和agent在 Occupancy measure 上的差异
2. 有时参数 $\theta$ 的随机初始化会导致优化过程一开始不可计算。本文提出了一种恢复策略，将约束变成目标函数来消除这个问题：$\theta^{\star}=\underset{\theta}{\arg \min } \mathbb{D}\left[\rho_{\pi_{\theta}}(s, a) \| \rho_{\pi_{\theta^{-}}}(s, a)\right]$
3. 由于近似误差，有可能满足不了约束。本文沿着 $\Delta \theta=-\lambda^{\star-1}\left(u+r \nu^{\star}\right)$ 使用了**回溯线搜索**来保证约束，并用共轭梯度法加速计算 Hessian 阵的逆。

### ![image-20200524215647493](./RLfrom_Imperfect_Demonstration.assets\image-20200524215647493.png)4 Experiments

![image-20200524221043525](./RLfrom_Imperfect_Demonstration.assets\image-20200524221043525.png)

- **pre-train:** 类似 Deep Q-learning form Demonstrations
- **POfD:**  (Kang, Jie, and Feng 2018), which proposed to lever- age Generative Adversarial Networks (GANs)  to evaluate the discrepancy between the occupancy measure of expert and agent. 
- **Penalty:** 将差异度量作为reward的罚项
- **Penalty + Annealing:** 带退货的Penalty
- **MMD-Imitation**：(denoted as MMD-IL) 直接最小化与expert之间的差异度量

#### 4.1 消融1：对示范数据的敏感性

![image-20200524221144933](./RLfrom_Imperfect_Demonstration.assets\image-20200524221144933.png)

左图amount的影响，右图quality的影响

#### 4.2 消融2：对约束公差的敏感度

![image-20200524221938482](./RLfrom_Imperfect_Demonstration.assets\image-20200524221938482.png)

左图是阈值 d 的影响，右图是退火系数的影响。

