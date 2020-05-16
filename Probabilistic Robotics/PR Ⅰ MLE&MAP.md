## Probabilistic in Robotics Ⅰ

> 本系列结合 Probabilistic Robotics 以及 Deep Learning 两本书，从**贝叶斯(Bayes)**视角学习**推断(inference)**。

我们先**强化**一下**情怀**：

> Wouldn’t it be great if all our cars were able to safely steer themselves, making car accidents a notion of the past? Wouldn’t it be great if robots, and not people, would clean up nuclear disasters sites like Chernobyl? Wouldn’t it be great if our homes were populated by intelligent service robots that would carry out such tedious tasks as loading the dishwasher, and vacuuming the carpet, or walking our dogs? And lastly, a better understanding of robotics will ultimately lead to a better understanding of animals and people.

### Why Probabilistic?

因为相比于游戏AI，机器人所面临的 **Uncertainty** 更多。

1. Environment：物理世界的环境更不可预测。
2. Sensors：传感器有精度限制。机器人的激光雷达和摄像头可没有游戏AI的全局视野哦。此外，还容易受到噪音干扰。
3. Robots：机器人的控制也是有误差的。完全不是你想的那样指哪打哪。
4. Models：对环境以及机器人本体的动力学、运动学建模也是漏洞百出。在 **Model-Based RL 系列** (Ⅰ, Ⅱ, Ⅲ) 我们应该已经深有感触了。
5. Computation：作为real-time系统，机器人的智能控制系统的计算量就受到了限制。

All of these factors 给机器人系统带来了极大的 uncertainty。**概率的方法能够让机器人了解自身的不确定性，从而在应对以上不确定性时更加鲁棒robust**。



## 频率派与贝叶斯派

这是一个绕不过的话题，在进行概率机器学习之前，我们必须清楚地了解这两种思路在探讨**不确定性**时，出发点和立足点的差异。

- **频率派**：从**事件**的角度，试图通过多次独立重复实验，**以事件发生的频率逼近事件的概率**，以此来描述**事件本身的随机性**。
- **贝叶斯派**：从**观察者**的角度，以**观察者知识不完备(先验prior belief)**为出发点，通过多次独立重复实验**(统计证据evidence)**，使观察者对事件发生的概率具有完备地了解**(后验posterior belief)**。因此随机性并不源于事件本身，而是**用于描述观察者对事件地知识状态**。

另一种解释是：

- **频率派**：其特征是把需要推断的参数 θ 视作固定且未知的常数，而样本 X 是随机的，其着眼点在样本空间，有关的概率计算都是针对X的分布。
- **贝叶斯派**：他们把参数θ视作随机变量，而样本X是固定的，其着眼点在参数空间，重视参数θ的分布，固定的操作模式是通过参数的先验分布结合样本信息得到参数的后验分布。

二者从世界观上具有根本性的差异，但各具优势。贝叶斯派在推断的过程中加入了先验，更符合人或机器解决问题的思路，因此在机器学习中大放异彩。频率派则对不适合引入先验知识、十分追求严谨的应用场景更有优势。而在**机器人学、强化学习以及后面可能会讲的因果推理**中，我们更倾向于贝叶斯派。



## 最大似然估计MLE和最大后验概率MAP

两个学派对应了如下两种经典的推断方法：

- 频率学派（Frequentist）- 最大似然估计（MLE, Maximum Likelihood Estimation）
- 贝叶斯学派（Bayesians）- 最大后验估计（MAP, Maximum A Posteriori）

**什么叫似然？**

[详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解](https://blog.csdn.net/u011508640/article/details/72815981) 

这篇文章写的很好，我就直接抄了。

似然，和“概率”，“可能性”的意思差不多。在统计里面，似然函数和概率函数却是两个不同的概念（其实也很相近就是了）。

对于函数 $p(x|\theta)$， x 表示某一个具体的数据；θ 表示模型的参数。

- **似然函数**：如果x是已知确定的，θ是变量，这个函数叫做似然函数(likelihood function), 它描述对于不同的模型参数，出现x这个样本点的概率是多少。
- **概率函数**：如果θ是已知确定的，x是变量，这个函数叫做概率函数(probability function)，它描述对于不同的样本点x，其出现概率是多少。



我们来看看同一个问题，MLE和MAP怎么求解。

> 假设有一个造币厂生产某种硬币，现在我们拿到了一枚这种硬币，想试试这硬币是不是均匀的。即想知道抛这枚硬币，正反面出现的概率（记为θ）各是多少？
>
> 于是我们拿这枚硬币抛了10次，得到的数据（$x_0$）是：反正正正正反正正正反。我们想求的正面概率θ是模型参数，而抛硬币模型我们可以假设是二项分布。
>
> 那么，出现实验结果 $x_0 $（即反正正正正反正正正反）的似然函数是多少呢？



### 最大似然估计MLE

$$
\begin{aligned}
f\left(x_{0}, \theta\right)&=(1-\theta) \times \theta \times \theta \times \theta \times \theta \times(1-\theta) \times \theta \times \theta \times \theta \times(1-\theta)\\
&=\theta^{7}(1-\theta)^{3}\\
&=f(\theta)
\end{aligned}
$$

注意，这是个只关于 θ 的函数。而最大似然估计，顾名思义，就是要最大化这个函数。大多数时候，我们会对这个函数取对数，毕竟可以化乘法为加法，还不影响最大化。
$$
\begin{aligned}
lnf(\theta)&=ln\theta^7+ln(1-\theta)^3\\&=7ln\theta+3ln(1-\theta)
\end{aligned}
$$
令 $ln'\theta=0$，即有 $\dfrac{7}{\theta}+\dfrac{3}{1-\theta}=0$，解得 $\theta=0.7$

我们可以画出 f(θ) 的图像：

![likeli](https://img-blog.csdn.net/20170531003926799?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTUwODY0MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

可以看出，在 θ=0.7 时，似然函数取得最大值。

这样，我们已经完成了对 θ 的最大似然估计。即，抛10次硬币，发现7次硬币正面向上，最大似然估计认为正面向上的概率是0.7。（ummm..这非常直观合理，对吧？）

且慢，一些人可能会说，硬币一般都是均匀的啊！ 就算你做实验发现结果是 “反正正正正反正正正反”，我也不信θ=0.7，这不符合我们的常识嘛。

这里就包含了贝叶斯学派的思想了——要考虑先验概率。为此，我们需要从最大后验概率估计的角度看看。

#### 求解步骤：

1. 确定似然函数
2. 将似然函数转换为对数似然函数
3. 求对数似然函数的最大值（求导，解似然方程）$\theta^*_{ML}=\arg\max_\theta{\sum^N_{i=1}lnp(x_i|\theta)}$



### 最大后验概率估计MAP

- 最大似然估计是求参数 $θ$, 使似然函数 $P(x_0|θ)$ 最大。

- 最大后验概率估计则是想求 $\theta$，使 $P\left(x_{0} | \theta\right) P(\theta)$ 最大。求得的 $\theta $ 不单单让似然函数大，θ 自己出现的先验概率也得大。（这有点像正则化里加惩罚项的思想，不过正则化里是利用加法，而MAP里是利用乘法）

MAP其实是在最大化 $\boldsymbol{P}\left(\theta | x_{0}\right)=\frac{P\left(x_{0} | \theta\right) P(\theta)}{P\left(x_{0}\right)}$，不过因为 $x_0 $ 是确定的（即投出的“反正正正正反正正正反”），$P(x_0)$是一个已知值，所以去掉了分母$P(x_0)$。

最大化 ${P}\left(\theta | x_{0}\right)$ 的意义也很明确，$x_0$ 已经出现了，要求 $θ$ 取什么值使 ${P}\left(\theta | x_{0}\right)$ 最大（条件概率嘛）。顺带一提， ${P}\left(\theta | x_{0}\right)$ 即后验概率，这就是“最大后验概率估计”名字的由来。

对于投硬币的例子来看，我们认为（先验地知道）$θ=0.5$ 的概率很大，取其他值的概率小一些。我们用一个高斯分布来具体描述我们掌握的这个先验知识，例如假设P(θ)为均值0.5，方差0.1的高斯函数，如下图：

![ptheta](https://img-blog.csdn.net/20170531004009269?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTUwODY0MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

则 $P\left(x_{0} | \theta\right) P(\theta)$ 的函数图像为：

![map1](https://img-blog.csdn.net/20170531003829147?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTUwODY0MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注意，此时函数取最大值时，θ 取值已向左偏移，不再是0.7。实际上，在θ=0.558 时函数取得了最大值。即，用最大后验概率估计，得到θ=0.558。
最后，那要怎样才能说服一个贝叶斯派相信 $θ=0.7$ 呢？你得多做点实验。。

如果做了1000次实验，其中700次都是正面向上，这时似然函数为:

![likeli2](https://img-blog.csdn.net/20170530235524800?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTUwODY0MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如果仍然假设P(θ)为均值0.5，方差0.1的高斯函数，$P\left(x_{0} | \theta\right) P(\theta)$ 的函数图像为：

![map2](https://img-blog.csdn.net/20170531003953909?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTUwODY0MA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在 θ=0.696 处， $P\left(x_{0} | \theta\right) P(\theta)$ 取得最大值。

这样，就算一个考虑了先验概率的贝叶斯派，也不得不承认得把θ估计在0.7附近了。

PS. 要是遇上了顽固的贝叶斯派，认为 $P(θ=0.5)=1$，那就没得玩了。。 无论怎么做实验，使用MAP估计出来都是θ=0.5。这也说明，一个合理的先验概率假设是很重要的。（通常，先验概率能从数据中直接分析得到）

#### 求解步骤：

1. 确定参数的先验分布以及似然函数
2. 确定参数的后验分布函数
3. 将后验分布函数转换为对数函数
4. 求对数函数的最大值（求导，解方程）$\theta_{M A P}^{*}=\operatorname{argmax}_{\theta}\{p(\theta | x)\}=\operatorname{argmax}_{\theta}\{p(x | \theta) p(\theta)\}$




## Reference

1. [机器学习 · 总览篇 III](https://kangcai.github.io/2018/11/04/ml-overall-3/)
2. [详解最大似然估计（MLE）、最大后验概率估计（MAP），以及贝叶斯公式的理解](https://blog.csdn.net/u011508640/article/details/72815981) 

