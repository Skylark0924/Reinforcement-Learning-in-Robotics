#! https://zhuanlan.zhihu.com/p/395562430
# 近期 Robot Learning 领域大事件及思考

结束了半个多月的毕业旅行，马上又要开始博士的新征程了。几个月没学习，所以今天总结一下近期 Robot Learning 以及 Reinforcement Learning 领域的进展及大事件（吃瓜）。
> 记载的都是一些个人想法，可能与以往不同，有些杂乱。

## Pure RL 的瓜
先说说 pure RL 吧，最大的瓜莫过于俞老师在[**强化学习领域目前遇到的瓶颈是什么？**](https://www.zhihu.com/question/449478247/answer/2001407526)这个问题下的回答：

[强化学习领域目前遇到的瓶颈是什么? - 俞扬的回答 - 知乎](https://www.zhihu.com/question/449478247/answer/2001407526)

上面的答案是修改过的，原答案已经有很多好心人截过图了。

就三个字

![](https://pic4.zhimg.com/80/v2-be44f5202b10acefca5d081f2401b231.png)

没法用倒也是很大一部分事实，毕竟强化学习这种方法，亮点是在其trial and error的精神上，而在其他方面真的不能对它报有太高的期望。从两个方面来看：

1. 很多单调、重复、精准或安全性要求高的tasks真的不需要它，PID 就已经很 perfect 了，何必要搞这些幺蛾子。举个例子，这种情况在工业机器人上就是显而易见的。
2. 目前学界主流的强化学习都在搞游戏、搞仿真数据，说实话感觉都没啥实际意义。强化学习agent在游戏中把我们都打败了，那我们玩游戏的乐趣是什么？更有趣的是，前段时间新学校的博导让我审了IROS2021的两个稿，其中一篇本来说好做的是森林自动化伐木+运输。我看完Intro中作者的问题描述直接好家伙，这么复杂，这要是能做出来AGI不是梦了。结果没成想，接下来作者进行了大量的简化，直接把问题简化成了一个[FetchPickAndPlace-v1](https://gym.openai.com/envs/FetchPickAndPlace-v1/)。我傻眼了，所以这和森不森林、伐不伐木到底有啥关系？作者想在实际问题上做出贡献是值得称赞的，但是也不至于这样强扯吧。
   
   ![](https://pic4.zhimg.com/80/v2-acca314a8d57a73b83832742a11b85c3.png)

尽管如此，俞老师作为一位强化学习的大佬竟然给出这样以偏概全的答案，我第一时间就觉得事情并不简单。

![](https://pic4.zhimg.com/80/v2-20e5c40701bd7f34856cba1cca8c962a.png)

在结合俞老师后续的更新以及其专栏文章 [关于强化学习“没法用”的吐槽 - 俞扬的文章 - 知乎](https://zhuanlan.zhihu.com/p/391032165)之后，我才意识到，这可能是一场蓄意的拨乱反正。简单来说，其旨在强调强化学习研究道路应该更贴近实际问题，而不应该是在单纯刷SOTA。我寻思这不就是我们Robo专业要干的事嘛，毕竟这领域不仅是典型的实用型工程领域，而且好像也没啥benchmark可刷。所以经常看看大佬发言是很有好处的，可以时刻给自己增强道路自信。

## RL in Robotics
再说说将RL或者其他learning方法用到Robotics上这件事吧。

Robot Learning 一个震惊的消息就是RL龙头研究所 OpenAI，抵不住金钱的诱惑，裁掉了重资产且数据低效的Robo组，全力搞能赚钱的项目了。不过这也能接受，毕竟机器人“龙头”公司倒闭或被卖也不是一回两回了，但这也再次提醒我：既然踏上这条路，就不能一门心思呆呆地搞研究，还要时时刻刻想着商业化前景，否则不是被饿死就是老老实实当打工人。

想到不久前，我还将[OpenAI单手解魔方机器人](https://openai.com/blog/solving-rubiks-cube/)视频作为激励师弟们入门的素材，向他们展示机器人智能化的美好未来，然而现在才知机器人革命远未到来。视频如下，大家可以重温一下这“神迹”。

好在这条路上总有人在前赴后继着，一个OpenAI Robo倒下了，还会有千千万万个Robo组站起来，直到机器人革命的星星之火可以燎原。Alphabet 近期又一次开始了机器人商业化探索，成立了全新的Robotics + AI公司 Intrinsic，面向机器人更智能的工业控制软件设计。



近期和新博导开了几次会，有组会也有单独的，算是互相了解了一下。博导给出的研究计划主旨是 real world robot Learning，并且考虑通过表征的方式，增加机器人对环境的理解。当然了，这也可以反过来讲，即对环境（both external and internal）以某种方式进行表征 or 降维 or 转化到 latent space 。。。反正含义是一样的，然后再进行机器人的决策。这其实和我当初给他的 research proposal 如出一辙，也不知我怎么找到方向这么一致的导师的，所以申请学校这件事，完全是看机缘。回到研究内容上，本专栏从创刊开始，就在目录文章中写过

> This is a private learning repository about **R**einforcement learning techniques, **R**easoning, and **R**epresentation learning used in **R**obotics, founded for **Real intelligence**.

因此，本专栏从一开始就不只是在关注 Reinforcement Learning 这一种方法，而是所有适用的 learning 方法在 robotics 上的应用。接下来我在博士期间也会继续围绕 robot learning 写一些笔记，欢迎大家来为我这个菜鸡 **点赞+收藏+关注** 捧场呀！




本文纯属个人感悟，不构成投资建议！