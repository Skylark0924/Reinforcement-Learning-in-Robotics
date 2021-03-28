#! https://zhuanlan.zhihu.com/p/360526111
![https://www.google.com.hk/url?sa=i&url=https%3A%2F%2Fwww.goodfon.com%2Fwallpaper%2Fbbbbbbb-vvvvvvv-ccccccc-fffffff-ddddddddddddd.html&psig=AOvVaw0M5NS7MuaCa-V9PKUgaKIm&ust=1617006887932000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCMjvhZ_K0u8CFQAAAAAdAAAAABAD](https://pic4.zhimg.com/80/v2-21561c1fcf054afa0f71d8c185923409.jpg)
# PR Efficient Ⅴ：自预测表征，让RL agent高效地理解世界
>每天一篇 Efficient，离 robot learning 落地更进一步。

> 想要专栏作家勋章，大家快关注专栏 RL in Robotics帮帮我~

本文来自 ICLR 2021，初步满足了我对于**将 model-based RL 与 Representation Learning 结合**的想法。
![](https://pic4.zhimg.com/80/v2-63566cf1f0f815aca37498280f0578fa.png)

## Motivation
本文受近期 semi-supervised 和 self-supervised learning 领域的启发，通过将状态空间映射到 latent space，并使用自监督的方法从数据的自然结构中，即时生成无限地训练数据来提高数据效率。RL 算法可以利用这个 learned representation model 来预测未来多步的 latent state representation。 

## Method
![Full SPR method](https://pic4.zhimg.com/80/v2-96ce2800086c0e9a4f290fd317d8777f.png)
### Online and Target Encoder
设计一个 **online encoder** $f_o$，将观测状态 $s_t$ 转换为表征 $z_t\triangleq f_o(s_t)$。该表征函数只用于当前时刻。

真正用到未来状态表征预测的是 target encoder $f_m$，它不随梯度更新，其参数与 online encoder 的关系为：

$$
\theta_m\leftarrow \tau \theta_m + (1-\tau)\theta_o\\
$$

### Transition Model
$$
\hat{z}_{t+k+1} \triangleq h(\hat{z}_{t+k},a_{t+k})
$$

### Projection Heads
分别将 online 和 target encoder 得到的隐式表征投影到更小的 latent space，方便学习，并设置了一个 prediction head $q$ 函数来学习 online projection 和 target projection 之间的关系。
![](https://pic4.zhimg.com/80/v2-92aeef3e9af5286413ab9c534fe0cfbe.png)

### Prediction Loss
通过对 $t\sim t+K$ 时刻预测值和观测值之间的余弦相似度求和，计算出SPR的未来预测损失。（这和上一篇文章的多步损失一样，用于限制 long-horizon 的 learned model error）
![](https://pic4.zhimg.com/80/v2-a60e58e70b1abac80167b9f9a38d3d6d.png)

Prediction Loss 是以 auxiliary loss 的形式参与到训练中的，具体来说就是 $\mathcal{L}_\theta^{total} = \mathcal{L}_\theta^{RL}+\lambda \mathcal{L}_\theta^{SPR}$。

### Algorithm
![](https://pic4.zhimg.com/80/v2-574a00c3efa6be7f9afc3796ba016e54.png)

## Conclusion
其实本文的亮点在第三章 Related Work，**列举了大量 data-efficient RL 以及 RL 结合 state representation 的最新进展**，我在这里汇总一下：
1. **SiMPLe** (Kaiser et al., 2019)：学习 Atari 的像素级 transition model，以生成模拟训练数据，从而在100k帧设定下的几场比赛中取得了不错的成绩，但仍需要**花费数周**的训练时间；
2. **Data-Efficient Rainbow (DER) and OTRainbow** (Hasselt et al. (2019) and Kielak (2020))：引入了针对样本效率进行了调整的Rainbow变体，它们可通过更少的计算获得相当或更高的性能；
3. (Hafner et al., 2019; Lee et al., 2019; Hafner et al., 2020)：利用 **reconstruction loss** 进行训练的 latent-space model 来提高样本效率；
4. **DrQ** (Yarats et al., 2021) and **RAD** (Laskin et al., 2020)：发现适度的 data augmentation 可以大大提高 RL 中的样本效率；
5. **CURL** (Srinivas et al., 2020)：提出将 image augmentation and a contrastive loss 相结合来执行RL的表征学习。但是，RAD的跟踪结果 (Laskin et al., 2020) 表明，CURL的大部分利好来自图像增强，而不是对比损失。（所以本文一直在强调其模型具有兼容数据增强的特性）
6. **CPC** (Oord et al., 2018), **CPC|Action** (Guo et al., 2018), **ST-DIM** (Anand et al., 2019) and **DRIML** (Mazoure et al., 2020)：优化强化学习环境中的各种 temporal contrastive losses；
7. Kipf et al. (2019)：提出通过训练基于**图神经网络**的**结构化 transition model** 来学习面向对象的 **contrastive representations**。
8. **DeepMDP** (Gelada et al., 2019)：训练了具有未归一化的L2损失的 transition model，以预测未来状态的表征以及奖励预测目标函数；
9. **PBL** (Guo et al., 2020)：直接通过梯度下降训练的两个独立的 target networks，预测未来状态的表征。



这篇文章一个重大缺陷是，**自预测表征依赖于全局观测，使其应用场景受限于 video game**，仍不适合 real-world Robotics。