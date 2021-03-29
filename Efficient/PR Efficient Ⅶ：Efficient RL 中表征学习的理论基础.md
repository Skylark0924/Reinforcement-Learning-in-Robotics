#! https://zhuanlan.zhihu.com/p/360596249
![](https://pic4.zhimg.com/80/v2-b4bdef3e2d79be32d13434c1789ef211.jpg)
# PR Efficient Ⅶ：表征学习对 Efficient RL 影响的理论研究
> 封面自 [Link](https://www.google.com.hk/url?sa=i&url=http%3A%2F%2Fwww.allwhitebackground.com%2Fblue-minimalist-wallpapers.html%2Fdownload%2F29937&psig=AOvVaw234IM3ZqsztPUI2kQMh9vw&ust=1617032639790000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCKCE6pOq0-8CFQAAAAAdAAAAABAJ)

> 每天一篇 Efficient，离 robot learning 落地更进一步。

> 想要专栏作家勋章，大家快关注专栏 RL in Robotics帮帮我~

本文来自 ICLR 2020，[Is a Good Representation Sufficient for Sample Efficient Reinforcement Learning?](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.03016.pdf).
![](https://pic4.zhimg.com/80/v2-d8750086a7cdca378075342d3eeee2ee.png)


Recent work 仅针对 model-based 中 learned model error 进行研究，却很少有研究 Efficient RL 的必要条件。本文发现，从统计学的角度来看，对满足 sample-efficient RL的表征比传统的近似观点中要求**更加严格**。本文的主要结果为 RL 提供了清晰的门槛，表明在构成良好的函数逼近（就表征的维数而言）方面存在严格的限制。这些下限突显出，**除非其函数逼近的质量超过某些严格的阈值，否则一个良好的表征不足以实现 Efficient RL。** 本文试图了解当我们能够获得准确的（紧凑的）参数表征时，是否有可能进行 efficient 的学习？

## Proof 
由于我理论基础不太好，具体证明就不写了（/捂脸笑，人菜还想分析paper说的就是我），推荐参照 [@张楚珩](https://www.zhihu.com/people/zhang-chu-heng) 的文章：

[【强化学习 101】Representation Lower Bound](https://zhuanlan.zhihu.com/p/100213425 'card')

## Results
1. 对 value-based learning，本文证明即使所有策略的 $Q$ 函数都可以由给定表征的线性函数近似且具有近似误差 $\delta=\Omega\left(\sqrt{\dfrac{H}{d}}\right)$ ，其中 $d$ 是表征维度，$H$ 是 planning horizon，agent 仍然需要采样指数级的样本才能找到接近最优的策略；
2. 对 model-based learning，本文证明即使过渡矩阵和奖励函数可以由给定表征的线性函数逼近而具有近似误差 $\delta=\Omega\left(\sqrt{\dfrac{H}{d}}\right)$，agent 仍然需要采样指数级的样本才能找到接近最优的策略；
3. 对 policy-based learning，本文证明即使可以通过给定表征的线性函数以严格的 positive margin 完美预测最佳策略，agent 仍然需要采样指数级的样本才能找到接近最优的策略。

这些下限即使在确定性系统，甚至已知 transition model 的情况中也是如此。本文的结果突出了以下见解：
1. 对于最坏情况下表征的逼近质量，存在严格的阈值下限；
2. 我们发现 efficient 的问题不是由于传统观念下的 exploration 导致的，未知的奖励函数足以使问题变得棘手；
3. 我们的下限不是由于 agent 无法执行 efficient 的监督学习而引起的，因为如果数据分布固定，我们的假设确实允许多项式样本复杂度的上限；
4. 最大的困难来自于 distribution mismatch。

## Separations
本文还对比分析了不同的算法设定

Perfect representation vs. good-but-not-perfect representation

结论：**更好的表征形式具有 provable exponential benefit。**

Value-based learning vs. policy-based learning

结论：**表征预测 Q 函数的能力比预测最优策略的能力强得多。**

Supervised learning vs. reinforcement learning

结论：**样本复杂度对 planning horizon H 的依赖性是指数级的。**

Imitation learning vs. reinforcement learning

结论：**使用函数拟合时，policy-based RL 比 IL 差一个数量级。**

