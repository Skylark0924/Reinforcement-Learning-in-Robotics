## Preliminary of RL Ⅲ: on-policy, off-policy & Model-based, Model-free

### on-policy & off-policy

> On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy different from that used to generate the data。
>
> **差异：**更新Q值时所使用的方法是沿用既定的策略（on-policy）还是使用新策略（off-policy）。
>
> —— Sutton, RL: an introduction

**on-policy:** 若**交互/采样策略**和**评估及改善的策略**是同一个策略，可翻译为同策略。

**off-policy:** 若**交互/采样策略**和**评估及改善的策略**是不同的策略，可翻译为异策略。

这种差异有两种解读方式：

1. 策略迭代的策略不是当前交互的策略（Q-learning与Sarsa）
2. 策略迭代时候使用的经验不是以当前策略进行交互的（DQN等具有 experience replay的算法，这个好理解就不解释了）

以off-policy的Q-learning与on-policy的Sarsa为例：

Q-learning：
$$
\mathrm{Q}(\mathrm{s}, \mathrm{a}) \leftarrow \mathrm{Q}(\mathrm{s}, \mathrm{a})+\alpha\left(\mathrm{R}(\mathrm{s})+\gamma \max _{a}, \mathrm{Q}\left(\mathrm{s}^{\prime}, \mathrm{a}^{\prime}\right)-\mathrm{Q}(\mathrm{s}, \mathrm{a})\right)
$$
Sarsa:
$$
\mathrm{Q}(\mathrm{s}, \mathrm{a}) \leftarrow \mathrm{Q}(\mathrm{s}, \mathrm{a})+\alpha\left(\mathrm{R}(\mathrm{s})+\gamma \mathrm{Q}\left(\mathrm{s}^{\prime}, \mathrm{a}^{\prime}\right)-\mathrm{Q}(\mathrm{s}, \mathrm{a})\right)
$$
Q-learning 在 $T$ 步策略估计的时候，使用了具有最大 $Q$ 值的 $T+1$ 步action，是greedy的策略。然而，实际上下一步并不一定选择该action，因此是 off-policy。

Sarsa 在 $T$ 步策略估计的时候，使用了按照当前第 $T$ 步策略应该走的 $T+1$ 步action，就是 $T$ 步策略本身，故为 on-policy。

[其他多种描述可参见：[强化学习中on-policy 与off-policy有什么区别？](https://www.zhihu.com/question/57159315)]

---

#### Model-based & Model-free

**Model-based:** 先给model-based消个歧，强化学习中所说的model-based并不是已知环境模型，或者已知状态转移概率。而是要从经验中**学习到一个环境模型或者其他映射**，并利用这个 learned model 加速策略迭代的进程。这种方式更适合用于机器人等领域。

> 详细的Model-based笔记见本专栏的Model-based RL系列，我还会继续更新的。

**Model-free:** Model-free就是我们常听到的 DQN, DDPG, PPO 等SOTA算法。它和 model-based 的区别就在于是否利用经验做策略迭代之外的事。显然，所有 model-free 都可以转变为 model-based，model-based只是一个框架，任意的 model-free 算法都可以嵌套进去。

**经验的其他用途：**除了用于策略迭代外，经验还可用于：

- 拟合环境模型以及即时奖励模型：$R,S'\leftarrow Model(S,A)$，作为新的数据源补充算法的训练

  > Dyna, ME-TRPO, NAF

- 拟合未来的值函数以及即时奖励：$R,V'\leftarrow Model(S,A)$，辅助决策

  > VPN, I2A

- 拟合未来的 Q 值：$Q\leftarrow Model(S,A)$，用于增加 Q 值预估的质量，将其在环境模型中展开(rollout)

  > MVE, STEVE, MBPO



**Reference**

[1] [[Model-based]基于模型的强化学习论文合集](https://zhuanlan.zhihu.com/p/72642285)

---

#### Rollout

这个词经常会出现在 model-based 算法中，我一般常译作'**展开**'，或'**模型展开**'，用于描述如何使用 learned model 加速training过程。

**实际意义**: 在 current state 上，从每一个可能的action出发，根据给定的 policy 进行路径采样，最后根据多次采样的奖励和来对 current state 的每一个action的Q值进行估计。形象地描述就是，**站在路口(current state)先按照大脑中的map(learned model)想象一下接下来每条路(action)的后果(future reward)。**

- MC 中，采样是为了逐步使信息更准确，进而更准确地改善策略。
- Rollout 中，采样是采出每一步之后的一定信息，利用信息更新后，然后做出选择让这一步进入下一个状态（思想依然是主要关注当前状态）。

