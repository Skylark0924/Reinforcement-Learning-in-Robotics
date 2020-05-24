## Deep Q-learning from Demonstrations

2017 | [Paper](https://arxiv.org/pdf/1704.03732.pdf) | DeepMind | AAAI

*Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Z. Leibo*

### 1 Abstract

DRL在很多现实场景下很受限制，比方说这个专栏主要讲的Robotics。此外，最近我面试华为决策推理岗，跟那边的研发聊了聊，他们主要做的是终端的推送以及多终端的智能互联，RL对于Robotics问题还是可以通过构建simulator来解决的，但是用户推荐这种场景真的很难在simulator里构建一个可以交互的用户画像。因此，我们的宗旨还是**一如既往地寻求脱离simulator之道**。总而言之，本文要解决的还是**data-efficiency**的问题。

DeepMind的一群大佬在本文提出一个叫做 **Deep Q-learning from Demonstrations (DQfD)** 的算法，意图用少量的示范，极大地加速学习过程，并且利用 **prioritized replay mechanism (优先回放机制，一种DQN在采样上的改进方案)**来自动评估示范数据的重要性。

DQfD 的工作原理是**将时序差分与对于示范动作的监督学习分类结合在一起**。



### 2 Preliminary

本文的baseline是 Prioritized Dueling Double Deep Q-Networks (PDD DQN，仿佛是拼多多DQN hhhhh)。

[](https://zhuanlan.zhihu.com/p/141268549)

[](https://zhuanlan.zhihu.com/p/141268851)

[](https://zhuanlan.zhihu.com/p/140348314)



### 3 Methodology

#### 3.1 Pre-training

在与real-world交互之前，先用示范数据集的数据进行预训练，用value function来模仿示范者，以实现agent与env在**交互之初**就可以用上较为完善的TD。

> 从示范集中采样，并用四个loss来训练网络：
>
> - **1步Q-learning损失**, 用于保证这个value func网络满足Bellman方程；
> - **n步Q-learning损失**, 同上；
> - **监督学习的大间距分类损失**，用于示范动作的分类；
> - **参数的L2正则**。
>
> $$
> J(Q)=J_{D Q}(Q)+\lambda_{1} J_{n}(Q)+\lambda_{2} J_{E}(Q)+\lambda_{3} J_{L 2}(Q)
> $$

**Supervised loss:**

其中，supervised large margin classification loss至关重要，因为示范集的一大问题就是只包含了一小部分的状态空间，很多状态-动作根本就没有数据。如果只是用Q-learning update的方式更新，网络会朝着那些ungrounded variables的方向更新，并且受到bootstrap的影响，这将传播到其他state。

这里用了大间距分类损失作为监督学习的loss：
$$
J_{E}(Q)=\max _{a \in A}\left[Q(s, a)+l\left(a_{E}, a\right)\right]-Q\left(s, a_{E}\right)
$$
$a_E$是expert的示范动作；$l(a_E,a)$是margin function，$a=a_E$的时候为0，其余为正值。

与以往的imitation learning有很大的不同，这里学的是action的Q值，而不是单纯的模仿action。这个loss迫使agent的动作的Q值至少比示范动作Q值低一个margin。引入了这个loss就可以使未发生的动作的value确定为合理的value，并使greedy的policy能够受到这个模仿了示范者的value func的引导。

> 权衡一下，利用SL与RL中的Bellman equation，将专家数据看成一个软的初始化约束，在per-training的时候，约束专家数据中的action要比这个state下其他的action高一个 $l$ 值。这里其实是做了一个loss的权衡:这个 $l$ 值导致的action差别的loss高，还是不同action导致达到下一个状态的 $s'$ 的产生的loss高，如果专家的SL带来loss高的话，那么以专家的loss为主，如果是RL的loss高的话，那么这个约束就会被放宽，选择RL中的action。

**Q-learning loss:**

1步Q-learning loss用于约束连续状态之间的value并使学习到的Q网络满足Bellman方程。n步Q-learning loss用于将expert的trajectory传递到更早的状态，以实现更好地pre-training：
$$
r_{t}+\gamma r_{t+1}+\ldots+\gamma^{n-1} r_{t+n-1}+\max _{a} \gamma^{n} Q\left(s_{t+n}, a\right)
$$
**L2 loss:**

毫无疑问这是用来防止过拟合的。

综合loss公式里的 $\lambda$ 用于选择那些loss在某些情况下不被使用。例如，在下一节的交互中产生self-generated数据，在这些数据上训练明显没必要使用监督loss嘛($\lambda_2=0$)。

#### 3.2 Interacting with the real system

交互的数据保存在与示范数据不同的新buffer里，并用prioritized sampling来决定以多大比例从两个数据集中分别采样。

#### 3.3 Pseudo-code

![image-20200522121546661](./Deep_Q_From_Demonstration.assets\image-20200522121546661.png)

### 4 Experiment

- Full DQfD
- PDD DQN
- 监督学习的imitaion learning，没有与环境的交互
- Accelerated DQN with Expert Trajectories (ADET) (Lakshminarayanan, Ozair, and Bengio 2016)
- Human Experience Replay (HER) (Hosu and Rebedea 2016)
- Replay Buffer Spiking (RBS) (Lipton et al. 2016)

![image-20200522121626682](./Deep_Q_From_Demonstration.assets\image-20200522121626682.png)



![image-20200522122022338](./Deep_Q_From_Demonstration.assets\image-20200522122022338.png)



## Reference

1. [缓解cold start--Deep Q-learning from Demonstrations笔记](https://zhuanlan.zhihu.com/p/37884112)



