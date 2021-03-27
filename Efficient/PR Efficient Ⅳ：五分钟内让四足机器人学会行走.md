#! https://zhuanlan.zhihu.com/p/360314680
![https://www.google.com.hk/url?sa=i&url=https%3A%2F%2Fwww.muralswallpaper.com%2Fshop-murals%2Fblue-illustrated-landscape-mountains-wallpaper-mural%2F&psig=AOvVaw00PD4CDisHkg2wbVygGPgM&ust=1616908627866000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPDBzJXcz-8CFQAAAAAdAAAAABAs](https://pic4.zhimg.com/80/v2-6be9e4b0e24e0bb8659add81f6fbdd0f.png)
# PR Efficient Ⅳ：五分钟内让四足机器人自主学会行走
> **每天一篇 Efficient，离 robot learning 落地更进一步。**

经我观察，一般做 Efficient RL 的文章，都可以同时实现 safe exploration。这两个需求之间是有相通性的，毕竟 efficient 主要是通过压缩state或action维度来实现，那么在这个过程中压缩掉一些 unsafe exploration 就很好理解了。Efficient 与 safe exploration 的结合使强化学习在 real-world 机器人中的落地更进一步。

今天看一篇2019年 Google Robotics 的 CoRL
![](https://pic4.zhimg.com/80/v2-d6aef314ce748e24fa688ff3b5a4849d.png)

这篇文章的效果还是很惊艳的，从机器人随机动作采集数据开始，到机器人学会 walking 步态，一共只需要 4.5 min 的数据采集。就算是加上 rollout 和 experiment resets，也不过10分钟，36个 episodes (45,000 control steps)。

## Main Contribution
我们都知道 model-based 相较于 model-free RL 在机器人上是更合理且更高效的做法，这里就不赘述了。

针对 model-based RL in real-world Robotics，本文主要解决下述**三个问题**：
1. The learned model needs to be sufficiently **accurate** for long-horizon planning。即使短期内的小误差，经过 long horizon 的累积，对控制的影响都是致命的；
2. **Real-time** action planning at a high control frequency。实时性是机器人任务，尤其是底层控制器不得不面对的现实问题；
3. **Safe** data collection for model learning is nontrivial。防止 mechanical failures/damages。

本文给出的对应**解决方案**：
1. **Accurate**：使用 **multi-step loss** 来防止 long-horizon prediction 中的 error accumulation；
2. **Real-time**：将 planning 和 controlling **并行化**，并借助 learned model 进行超前预测，弥补 planning 和 controlling 之间的时差，实现**异步控制**；
3. **Safe**：使用 trajectory generator 确保 planned action 的**平滑性**。

下面我们详细展开一下这些贡献。

## Multi-step loss
**为什么用 multi-step loss，而非 MVE/STEVE/MBPO 常用的 model ensemble 来消除 model error的影响？**

虽然 ensemble 也不错，但是它增加了 planning time，影响了实时性。

**Multi-step loss 的形式**
普通的单步损失：

![](https://pic4.zhimg.com/80/v2-8f625df91d94c23affe4c470278c08a7.png)

多步损失：
![](https://pic4.zhimg.com/80/v2-6a385eaf96364d5266437c06e8d9bd1d.png)

其中，$f_\theta(s_t, a_t)$ 是用于拟合 model-based 惯用的状态差 $s_{t+1} - s_t$ 的神经网络。至于其消除 long-horizon model error 的作用还是显而易见的。

## Parallel and asynchronous
![](https://pic4.zhimg.com/80/v2-03cdaa25177d0664d543cfc2c96aee27.png)

并行必定会导致两个环节处理时长不匹配的问题。所以这里就借助了上一节的 learned model 让 planner 能够在 $t$ 时刻超前预测 $t+T$ 时刻的 planning，从而弥补了 planning 的滞后。

## Smoothness
![](https://pic4.zhimg.com/80/v2-2c380cf4434abc16db662b4e4b438a60.png)
这里的 Trajectory generators (TGs) 概念借用自 [Policies Modulating Trajectory Generators - Google, CoRL 2018 ](https://arxiv.org/pdf/1910.02812.pdf)，我简单看了一下，这是一个可以为控制器结合 **memory** 和 **prior knowledge** 控制行为生成器，是机器人能够生成更符合人类期望的复杂行为或步态。

除此之外，本文对输出的连续 actions 也做了平滑处理。
![](https://pic4.zhimg.com/80/v2-51f00c01e7e9552042e1ca3b53e857b2.png)

## Experiments
## On-robot experiments
![](https://pic4.zhimg.com/80/v2-fe98b9f7572eef25213ef355b8a7d4e6.png)

### 与 model-free 的效率对比
学习效率较 model-free 方法高出一个数量级。不过这里并没有出现与 SOTA 的model-based 方法的比较，但是考虑到五分钟的训练就有这样的效果，还是不错的了。

![](https://pic4.zhimg.com/80/v2-055ecfa2fbaf62f087e209043ba91b40.png)

## Conclusion
从 efficient 的角度来看，效果还是不错的，也考虑并解决了很多**实际工程问题**，值得借鉴。