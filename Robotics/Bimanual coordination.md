# Bimanual coordination

- [Bimanual coordination](#bimanual-coordination)
  - [Function-based](#function-based)
    - [Bimanual twisting](#bimanual-twisting)
    - [Redundancy Resolution for Dual-Arm Robots Inspired by Human Asymmetric Bimanual Action: Formulation and Experiments](#redundancy-resolution-for-dual-arm-robots-inspired-by-human-asymmetric-bimanual-action-formulation-and-experiments)
      - [机器人建模](#机器人建模)
  - [Learning-based](#learning-based)

## Function-based

### Bimanual twisting
![](https://pic4.zhimg.com/80/v2-da307375e6be4998099ae113ef8f5c82.png)

![](https://pic4.zhimg.com/80/v2-e39e7c71103548f39eb12248ad58969c.png)



### Redundancy Resolution for Dual-Arm Robots Inspired by Human Asymmetric Bimanual Action: Formulation and Experiments
#### 机器人建模
![](https://pic4.zhimg.com/80/v2-3054574f214119ced5cf4c04ff071109.png)
A相对于B的速度即为：
$$
\dot{\mathrm{x}}_{R}=-\dot{\mathrm{x}}_{A}+\dot{\mathrm{x}}_{B}=\mathbf{J}_{R} \dot{\mathbf{q}}
$$
其中，相对雅克比为
$$
\mathbf{J}_{R}=\left[-{ }^{R} \mathbf{R}_{A} \mathbf{J}_{A} \quad{ }^{R} \mathbf{R}_{T}{ }^{T} \mathbf{R}_{B} \mathbf{J}_{B}\right]
$$
A在零空间内的绝对运动为：
$$
\dot{\mathbf{q}}=\mathbf{J}_{R}^{+} \dot{\mathbf{x}}_{R}+\left(\mathbf{I}-\mathbf{J}_{R}^{+} \mathbf{J}_{R}\right)\left[\begin{array}{ll}
\mathbf{J}_{A} & \mathbf{0}
\end{array}\right]^{+} \dot{\mathbf{x}}_{A}
$$
其中，+为伪逆。最后一项意味着右手（工具末端执行器）相对于左手（参考末端执行器）移动。此外，左手的绝对运动不受限制，因此可以在可触及的工作空间中的任何地方。这种无节制的运动可用于回避障碍或关节限制回避等自我运动，即左臂调整工作空间位置，使右臂在执行不对称双手任务时不会达到其关节极限。

关节加速度方面的冗余分辨率为
$$
\ddot{\mathbf{q}}=\mathbf{J}_{R}^{+}\left(\ddot{\mathbf{x}}_{R}-\dot{\mathbf{J}}_{R} \dot{\mathbf{q}}\right)+\left(\mathbf{I}-\mathbf{J}_{R}^{+} \mathbf{J}_{R}\right) \boldsymbol{\eta}
$$

**相对阻抗控制**
扭矩级控制器采用时间延迟估计设计:
$$
\boldsymbol{\tau}_{u}=\overline{\mathbf{M}} \ddot{\mathbf{q}}+\boldsymbol{\tau}_{u(t-L)}-\overline{\mathbf{M}} \ddot{\mathbf{q}}_{(t-L)}
$$
$\ddot{\mathbf{q}}$ 是从上上式计算得出的，其中 ̈ xR 是从目标动力学计算得出的，目标动力学表示两个末端执行器之间相对运动和相对力的动态关系，如下所示：
$$
\mathbf{f}_{R}=\mathbf{M}_{R d}\left(\ddot{\mathbf{x}}_{R d}-\ddot{\mathbf{x}}_{R}\right)+\mathbf{B}_{R d}\left(\dot{\mathbf{x}}_{R d}-\dot{\mathbf{x}}_{R}\right)+\mathbf{K}_{R d}\left(\mathbf{x}_{R d}-\mathbf{x}_{R}\right)
$$
$\mathbf{f}_{R}$ 代表相对接触力， $\mathbf{M}_{R d}, \mathbf{B}_{R d}, \mathbf{K}_{R d}$分别代表预期的质量、阻尼和刚度矩阵。

## Learning-based

