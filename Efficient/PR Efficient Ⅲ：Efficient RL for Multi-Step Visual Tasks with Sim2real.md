#! https://zhuanlan.zhihu.com/p/359776893
![](https://pic4.zhimg.com/80/v2-28be419e774a12f5102ca7a986bfdc39.jpg)
# PR Efficient Ⅲ：像训练狗狗一样高效地训练机器人
> **每天一篇 Efficient，离 robot learning 落地更进一步。**

![](https://pic4.zhimg.com/80/v2-c40c8775ca0291e53756dd5e210c4d93.png)

这是一篇20年的 RA-L，题目要素过多，既有 efficient 又可以 multi-step 还能考虑到 sim2real，让人看了直呼 **Good**！

> Current Reinforcement Learning (RL) algorithms
struggle with long-horizon tasks where time can be wasted exploring dead ends and task progress may be easily reversed. We develop the SPOT framework, which explores within action safety zones, learns about unsafe regions without exploring them, and prioritizes experiences that reverse earlier progress to learn with remarkable efficiency

摘要直接指明当前RL算法在探索死胡同或无用的状态上浪费了太多时间，且任务进度很容易被逆转 (progress reverse, 稍后解释)。因此，本文提出了 SPOT 框架，实现了以下三点：
1. 在安全区内探索；
2. 即使没有探索也能学习非安全区的知识；
3. 优先考虑可以逆转早期学习进度的经验。

从而实现高效的强化学习。下图展示了他们实机测试的结果，十分令人振奋！

![SPOT 针对的是方块堆叠任务以及将他们排成一排，将这两个任务的成功率都提升至 100%，效率提升 30%，使训练过程只需要 1-20k 步。且这些在仿真器中训练的模型可以在不进行 fine-tuning 的情况下迁移到 real world，并获得 100% 的成功率！](https://pic4.zhimg.com/80/v2-fd2fb6d35c24dc9902755811e713dfff.png)

## Idea
**什么是无用的探索？**

具体来说，以人类的角度，上述任务中机器人在空气中乱抓明显是没意义的，然而 RL 算法们基本都在这种情况上浪费了大量的时间。

**如何避免无效探索？**

本文提出的框架，结合了 **common sense constraints**，从而显著提高学习效率与最终任务效率。借鉴了一种叫做 **"Positive Conditioning"** 的人性化与高效的宠物训练方法。这里直接放个 wikipedia 的链接吧：[操作性条件反射](https://zh.wikipedia.org/wiki/%E6%93%8D%E4%BD%9C%E5%88%B6%E7%B4%84)
> Bionics 万岁！话说 RL 本身好像就是 Bionics啊，和这个操作性条件反射没啥区别吧。有趣的是，我发现生活中这些都属于操作性条件反射：
> 1. 在滑雪课上执行转弯后，你的教练喊道：“辛苦了！”
> 2. 在工作中，你超出了本月的销售配额，因此你的老板给了您奖金。
> 3. 在心理学课程中，你观看有关人脑的视频并撰写关于所学知识的论文。你的老师会为你的工作提供20个额外的学分。

**什么是 progress reverse?**

简单来说，这就是个汉诺塔，基础不牢，堆的越高错的越惨。Progress reverse 就是描述这样一个推倒重来的情景。
![](https://pic4.zhimg.com/80/v2-ed72b57791fb39072d5a0d8ba6c16aa8.png)

## Approach
![Main Framework](https://pic4.zhimg.com/80/v2-2db633e08bd5ded442d3afaf55da95bc.png)
作者认为在机器人应用中，Q-learning算法的动作空间和尝试成本很大，并且 efficient 也会有一定的 safe exploration 保证。此外，算法的性能很大程度取决于奖励函数的设置，因此本文先从 task-specific reward shaping 入手。

### Reward shaping
本文列出了几种奖励函数的设置，其中的 $\mathbf{1}$ 是二进制的 indicator function：
1. $R_{base}(s_{t+1}, a_t)=W(\phi_t)\mathbf{1}_a[s_{t+1}, a_t]$
   > $\mathbf{1}_a[s_{t+1}, a_t]$等于1代表 $a_t$ 在 sub-task $\phi$ 中成功，反之为0；$W$ 是每个 sub-task 的权重。
2. $R_{SR}(s_{t+1}, a_t)=\mathbf{1}_{SR}[s_t, s_{t+1}]R_{base}(s_{t+1}, a_t)$
   > 定义一个全局任务进度函数 $\mathcal{P}: S\rightarrow \mathbb{R}\in [0,1]$，当$\mathcal{P}(s_t)=1$的时候，全局任务完成；$\mathbf{1}_{SR}[s_t, s_{t+1}]$ 在 $\mathcal{P}(s_{t+1})\ge\mathcal{P}(s_t)$时为1。
3. $R_\mathcal{P}=\mathcal{P}(s_{t+1})R_{SR}(s_{t+1}, a_t)$

> 解释一下为什么要 2 和 3 区分开，以及什么情况 $\mathcal{P}(s_{t+1})\ge\mathcal{P}(s_t)$

> 本文将 $\mathcal{P}$ 设定为木块堆积高度与期望总高度的比值，摆成一排的任务也是如此。所以这是一个 $[0,1]$ 的离散值，而非二进制。

以上三种奖励函数都可以及时获得，但是没有考虑早期的错误，所以本文又给出了下面这个贯穿整个训练过程的奖励函数。

### Situation Removal: SPOT Trail Reward
![](https://pic4.zhimg.com/80/v2-52570c1b2a829e51f3989b9dd9f359fa.png)

其中，$R_*$ 可以是 $R_{SR}, R_\mathcal{{P}}$ 的任意一种。$N$ 代表任务终止，$\gamma = 0.65$。$\mathbf{R}_{trial}$ 的意义是，未来的奖励只会在子任务完成时回传。

以下面的例子说明，**Situation removal 的零奖励切断了未来奖励的回传，致使算法聚焦在那些短且成功的序列上**。
![](https://pic4.zhimg.com/80/v2-04320a29db3c26d87d46439bf53ffa17.png)

### SPOT-Q Learning and Dynamic Action Spaces
这一节作者引入环境先验来加速学习。假设了一个**先知** $M(s_t, a)\rightarrow {0,1}$，当前 action 失败就返回0。根据这个先知，我们可以定义一个动态的动作空间（也就是利用先知缩小动作空间，去掉那些不值得探索的动作）:
$$
M_t(A) = \{a\in A|M(s_t, a)=1\}\\
$$
所以就有了压缩动作空间之后的 **SPOT-Q learning**：
![](https://pic4.zhimg.com/80/v2-db48a5088597d3b753ded2a63488ba69.png)

算法伪码如下
![使用了PER技术](https://pic4.zhimg.com/80/v2-4cd0661539c90059958436b813c54a31.png)

## Experiment
![Simulation 实验](https://pic4.zhimg.com/80/v2-12689547c741efb3a369e4dc67369bca.png)

![Real world 实验](https://pic4.zhimg.com/80/v2-eb0cc1bf72801430cfdb51951d9b58d9.png)

## Conclusion
SPOT 框架根据 zero-reward guidance, a masked action space, situation removal 这三方面提升了任务效果。

正如文末指出的，本文的主要 weakness：
1. 需要有一个从 data 结合 Situation removal 的learning structure；
2. 用于动作空间压缩的先知mask $M$ **竟然是手工设定的**，我读文章时候一直以为是类似于 model learning 的操作。

总的来说，实验效果是好的，但是提出的算法看不出什么技术含量，毕竟 task-specific 的 reward shaping 只是 task-specific。不过，压缩动作空间来实现 efficient 的想法是值得借鉴的。