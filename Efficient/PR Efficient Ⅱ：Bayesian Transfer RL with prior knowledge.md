#! https://zhuanlan.zhihu.com/p/359620737
![](https://pic4.zhimg.com/80/v2-3f3e395d0e8d39077bf79f8209fa9b36.jpg)
# PR Efficient Ⅱ：Bayesian Transfer RL with prior knowledge

> **总第100篇文章纪念**
> 
> **每天一篇 Efficient，离 robot learning 落地更进一步。**

![](https://pic4.zhimg.com/80/v2-03acccf1226547eb91fe93c9a08401ef.png)

## Bayesian Transfer RL
本文将 off-policy RL 中的 behavior policy 定义为一个 Bayesian posterior distribution，用来结合task-specific的先验。期望以这种方式实现 transfer learning 以及 meta-learning。基于当前Q函数估计的 behaviour policy 通常采用 softmax 或Boltzmann形式：

![](https://pic4.zhimg.com/80/v2-9d9417040f1b04af1cb785731b5b5745.png)

这个策略只用到了 task-specific information, 例如任务的 state-action value function，我们可以这样将先验结合进来：

![](https://pic4.zhimg.com/80/v2-95d118a3a976b4d012a9d29cd0d6e90c.png)

这里的 $f(\alpha;\mathcal{M}_t)\ge 0$是个非负的函数，且是一个非归一的概率分布，当动作 $\alpha_t$ 符合先验时返回 high value，反之 small value。$\mathcal{M}_t$ 表示表达先验知识并计算 $f(\cdot)$ 值所需的所有存储信息。这里的超参数 $\beta$ 也起到了平衡 task-specific information 与先验的作用。

显然，这是避免了机器人对无关紧要的部分进行重复探索，并且可以算是一种实现 sample-efficient 的方式。

> 这里原文说的是：It is **the only way** to achieve sample efficient learning in very large or continuous state spaces. 太过确定，不敢苟同。

除了可以用于 off-policy 算法之外，本文还给出了 on-policy 算法结合的方式，实际上和 of-policy 没什么区别：

![](https://pic4.zhimg.com/80/v2-bb1646b02e2591aa28edeee45457a771.png)

**先验函数 $f(\alpha;\mathcal{M}_t)$ 该怎么定义？**

本文给出了先验思想是：In a roughly deterministic and stationary world, past plans that resulted in no progress for solving a task need to be tried out less frequently in the future Humans. 

实际上就是减少 useless re-exploration。所以文中给出的例子也是很简单的列出一些不应该执行的 deterministic rules，并根据 rollout length 定义为 M-order rules:
- 1-order rule: 假设我们处于状态$s_t = s$，执行动作 $α_t=α$ 并且发现状态保持不变，即下一个状态还是 $s_{t+1} = s$。 那么，无论何时我们处于状态s时，都不应尝试采取动作 $α$，除非停留在 $s$ 有 reward。
- 2-order rule: 假设我们处于状态 $s_t = s$，应用动作 $α_t=α$，移动到状态 $s_{t + 1} = s'\neq s$，执行第二个动作 $α_{t+ 1} =α'$，我们返回到初始状态 $s_{t + 2} = s$。
那么，永远不要尝试在处于状态 $s$ 时采取一系列动作$(α，α')$的计划，除非停留在$s$或$s'$有reward。

> 这个先验还真是相当 simple & naive。很明显文中这种先验的结合会更适合用在概率RL上，例如 soft-Q 或 SAC 之类的有熵正则的策略梯度算法。所以本文接下来开始讨论 probabilistic RL。

## Probabilistic RL
考虑一个 episodic RL，我们根据如下的联合分布来生成状态和动作：

![](https://pic4.zhimg.com/80/v2-073fbaaf9d173cf46471498b7a6ff5aa.png)

由于 RL 学习的目标是最大化 $\sum^{h-1}_{t=0} r_t$，那么可以把它视为一种约束并加入到联合分布中：

![](https://pic4.zhimg.com/80/v2-d3300aa6b7efc6450286ae9fbd38088c.png)

这里将上式的整体分解视为一个势函数（类似于无向图模型），可以定义以下后验分布，

![](https://pic4.zhimg.com/80/v2-791a05823a382cf96d31bd90344a7f98.png)

假设我们处于状态 $s_t$，我们感兴趣的是根据所有奖励 $r_{0:h-1}$以及所有过去状态和动作$(s_{0:t-1}，α_{0:t})$来计算作用 $α_t$ 的边际后验分布。基于 Markov 独立性，这个条件分布可以简化为：

![](https://pic4.zhimg.com/80/v2-3d47c4ea2d2dfdbe1c780985ab17166d.png)

这样的策略满足如下所述的Bellman型递归方程

![](https://pic4.zhimg.com/80/v2-768acf02973169cc2cb88e40f039dfd5.png)

在常规强化学习中，状态作用值函数 $B(s_t,α_t)$ 与最佳Q函数联系在一起:

![](https://pic4.zhimg.com/80/v2-13ec33fbe4a0dc62c1bb90d3273823cf.png)

更准确地说，对于离散状态和动作，我们希望直接近似命题1中的Bellman方程，以便随机近似状态动作值 $B(s_t，α_t)$。

![](https://pic4.zhimg.com/80/v2-bd3ec5dbcd18886afb34e132ebfd31a5.png)

然后在此基础上进行 stochastic optimization update：
![](https://pic4.zhimg.com/80/v2-94d2d922f40e30aa226d266fc8629516.png)

最后将这个近似策略和先验像上一节一样结合起来：

![](https://pic4.zhimg.com/80/v2-fbf72cc31434c5c0e55ebf5f38f9a99c.png)

## Experiment

实验部分堪称潦草，对比了一下无先验、1/2-order 在迷宫环境下的曲线，倒是确实有些 efficient，但是也没那么明显：

![](https://pic4.zhimg.com/80/v2-1ce5ee85186c816df7ed2173d3781030.png)

## Conclusion 
虽然是篇没什么人看的半成品文章，但是他想把 probabilistic RL 和先验结合的思路是可以借鉴借鉴的，所以看多了大佬的文章，也要看看这种集思广益一下。

Btw. 我发现一作竟然是个专攻 probability theory in learning 的  DeepMind Research Scientist，但是仍然无法改变这是个实验十分敷衍的半成品的事实。