# $\mathcal{QD}$-learning: A collaborative distributed strategy for multi-agent reinforcement learning through consensus + innovations

Paper

[TOC]

$$
\begin{array}{l}
Q_{i, u}^{n}(t+1)=Q_{i, u}^{n}(t)-\beta_{i, u}(t) \sum_{l \in \Omega_{n}(t)}\left(Q_{i, u}^{n}(t)-Q_{i, u}^{l}(t)\right) \\
+\alpha_{i, u}(t)\left(c_{n}\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)+\gamma \min _{v \in \mathcal{U}} Q_{\mathbf{x}_{t+1}, v}^{n}(t)-Q_{i, u}^{n}(t)\right)
\end{array}
$$



## Convergence of $\mathcal{QD}$-learning

<img src="D:\Github\Reinforcement-Learning-in-Robotics\MARL\MARL Ⅱ：QD-learning.assets\image-20200914210232950.png" alt="image-20200914210232950" style="zoom:50%;" />

带有 域流 $\mathcal{F}$ 的完整概率空间

对于每个代理 n，定义局部 QD-learning 算子 $\mathcal{G}^{n}$。其中，对于每一个 state-action pair $(i, u)$为：
$$
\mathcal{G}_{i, u}^{n}(Q)=\mathbb{E}\left[c_{n}(i, u)\right]+\gamma \sum_{j \in \mathcal{X}} p_{i, j}^{u} \min _{v \in \mathcal{U}} Q_{j, v}
$$

- $c_n(i,u)$ 为 cost
- $p^u_{i,j}$ 为 $x_t =i, x_{t+1} =j$ 的状态转移矩阵 $P\left(\mathbf{x}_{l+1}=j \mid \mathbf{x}_{l}=i, \mathbf{u}_{l}=u\right)=p_{i, j}^{u}, \forall i, j \in \mathcal{X}, u \in \mathcal{U}$

在 M1 下，可以得到：
$$
\begin{aligned}
\mathbb{E}\left[c_{n}\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right) \mid \mathcal{F}_{t}\right] &=\mathbb{E}\left[c_{n}(i, u)\right] \\
\mathbb{E}\left[\min _{v \in \mathcal{U}} Q_{\mathbf{x}_{t+1}, v}^{n}(t) \mid \mathcal{F}_{t}\right] &=\sum_{j \in \mathcal{X}} p_{i, j}^{u} \min _{v \in \mathcal{U}} Q_{j, v}^{n}(t)
\end{aligned}
$$
故 QD-learning 可以写作
$$
\begin{aligned}
Q_{i, u}^{n}(t+1)=Q_{i, u}^{n}(t)-\beta_{i, u}(t) \sum_{l \in \Omega_{n}(t)}\left(Q_{i, u}^{n}(t)-Q_{i, u}^{l}(t)\right) \\
+\alpha_{i, u}(t)\left(\mathcal{G}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}\left(\mathbf{Q}_{t}^{n}\right)-Q_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}(t)+\boldsymbol{\nu}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}(t)\right)
\end{aligned}
$$
其中残留项为：
$$
\boldsymbol{\nu}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}(t)=c_{\boldsymbol{n}}\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)+\gamma \min _{v \in \mathcal{U}} Q_{\mathbf{x}_{t+1}, v}^{n}(t)-\mathcal{G}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}\left(\mathbf{Q}_{t}^{n}\right)
$$


### Boundedness 有界性

Lemma 5.1  对于每个代理 n，连续的优化序列 $\{Q^n_t\}$ 是按路径限制的，
$$
\mathbb{P}\left(\sup _{t \geq 0}\left\|\mathbf{Q}_{t}^{n}\right\|_{\infty}<\infty\right)=1
$$
对于每个代理 n，每一个 state-action pair $(i, u)$
$$
\begin{aligned}
Q_{i, u}^{n}(t+1)=Q_{i, u}^{n}(t)-\beta_{i, u}(t) \sum_{l \in \Omega_{n}(t)}\left(Q_{i, u}^{n}(t)-Q_{i, u}^{l}(t)\right) \\
+\alpha_{i, u}(t)\left(\mathcal{G}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}\left(\mathbf{Q}_{t}^{n}\right)-Q_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}(t)+\boldsymbol{\nu}_{\mathbf{x}_{t}, \mathbf{u}_{t}}^{n}(t)\right)
\end{aligned}
$$
因此，所有 agents 的 $Q_{i,u}$ 为
$$
\begin{aligned}
\mathbf{Q}_{i, u}(t+1)=\left(I_{N}-\beta_{i, u}(t) L_{t}-\alpha_{i, u}(t) I_{N}\right) \mathbf{Q}_{i, u}(t) & \\
+\alpha_{i, u}(t)\left(\mathcal{G}_{i, u}\left(\mathbf{Q}_{t}\right)+\boldsymbol{\nu}_{i, u}(t)\right) &
\end{aligned}
$$
其中  $\mathcal{G}_{i, u}\left(\mathbf{Q}_{t}\right)=\left[\mathcal{G}_{i, u}^{1}\left(\mathbf{Q}_{t}^{n}\right), \ldots, \mathcal{G}_{i, u}^{N}\left(\mathbf{Q}_{t}^{N}\right)\right]^{T}$















- 如果 $r^j_{t}$ 小，说明临近路口 $j$ 的最大排队数小，所有路口的车辆都较少，意味着我应该给朝着 $j$ 路口的phase放行
- 如果 $r_t^j$ 大，说明紧邻路口 $j$ 较堵，这时要讨论是否应该避讳这个路口
  - 如果obs中的end car较多，意味着这个相位确实这条路确实很堵，不应该被放行
  - 如果obs中的end car 较少，说明路口 $j$ 堵得是其他方向的路

我们的目标是，要让greenwave这种情况得到奖励！要让越堵越放的情况得到惩罚！

当前 phase == 1 北上 南下

- Greenwave 的情况：





要知道邻近路口的全观测，12 * 4 + 1 phase= 49

$w_{t+n}^j(a_t)$: if we take $a_t$ in intersection $i$，it represents the waiting car number in corresponding lane of intersection $j$ at time $t+n$

所以对这个附加项，一是要展示我们放行相位的未来效果，二是要展示未放行相位的后悔程度。

放行相位：



```
sorted_keys:
['road_0_1_0_0', 'road_0_1_0_1', 'road_0_1_0_2', 'road_1_0_1_0', 'road_1_0_1_1', 'road_1_0_1_2', 
'road_1_2_3_0', 'road_1_2_3_1', 'road_1_2_3_2', 
'road_2_1_2_0', 'road_2_1_2_1', 'road_2_1_2_2']
```

$$
\begin{aligned}
	Q(o_t^i,a_t^i)=&Q(o_t^i,a_t^i)+\alpha (r^i_t+\gamma\cdot r^i_t \cdot \text {tanh} [ \sum_{j \in \mathcal{F}}(\frac{R^{j}_{t+n}}{r^{j}_t}-c)]\\&+\gamma' \max_{a_{t+1}^i}Q(o_{t+1}^i,a_{t+1}^i)-Q(o_{t}^i,a_{t}^i))
	\end{aligned}
$$

