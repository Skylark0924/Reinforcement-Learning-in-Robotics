# $\mathcal{QD}$-learning: A collaborative distributed strategy for multi-agent reinforcement learning through consensus + innovations

Paper


$$
\begin{aligned}
	Q(o_t^i,a_t^i)=&Q(o_t^i,a_t^i)+\alpha (r^i_t+\gamma\cdot r^i_t \cdot \text {tanh} [ \sum_{j \in \mathcal{F}}(\frac{R^{j}_{t+n}}{r^{j}_t}-c)]\\&+\gamma' \max_{a_{t+1}^i}Q(o_{t+1}^i,a_{t+1}^i)-Q(o_{t}^i,a_{t}^i))
	\end{aligned}
$$

$$
\begin{array}{l}
Q_{i, u}^{n}(t+1)=Q_{i, u}^{n}(t)-\beta_{i, u}(t) \sum_{l \in \Omega_{n}(t)}\left(Q_{i, u}^{n}(t)-Q_{i, u}^{l}(t)\right) \\
+\alpha_{i, u}(t)\left(c_{n}\left(\mathbf{x}_{t}, \mathbf{u}_{t}\right)+\gamma \min _{v \in \mathcal{U}} Q_{\mathbf{x}_{t+1}, v}^{n}(t)-Q_{i, u}^{n}(t)\right)
\end{array}
$$

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

