# LINEAR-QUADRATIC REGULATION FOR NON-LINEAR SYSTEMS USING FINITE DIFFERENCES

[[TOC]]

翻译自：https://studywolf.wordpress.com/2015/11/10/linear-quadratic-regulation-for-non-linear-systems-using-finite-differences/

Linear-quadratic regulator (LQR) 分为 finite-horizon 和 infinite-horizon 两种，这里我们关注 infinite-horizon。

## 线性系统下的 LQR
线性系统：

$$
\dot{\mathbf{x}}=\mathbf{A} \mathbf{x}+\mathbf{B u}
$$

其中，$\mathbf{x}$ 为状态 state，$\mathbf{u}$ 为 input， $\mathbf{A}$和 $\mathbf{B}$ 则 capture 状态和输入对导数物的影响。

接下来，定义一个损失函数（二次方程）：

$$
J=\int_{0}^{\infty}\left(\left(\mathbf{x}-\mathbf{x}^{*}\right)^{T} \mathbf{Q}\left(\mathbf{x}-\mathbf{x}^{*}\right)+\mathbf{u}^{T} \mathbf{R} \mathbf{u}\right) d t
$$

其中，$\mathbf{x}^{*}$ 是**目标状态**，$
\mathbf{Q}=\mathbf{Q}^{T} \geq 0 \text { and } \mathbf{R}=\mathbf{R}^{T} \geq 0
$ 分别是**不处于目标状态**和**应用控制信号**的cost 权重。Q 越高，尽快到达目标状态越重要，R 越高，在进入目标状态时保持控制信号小越重要。

LQR 的目标即为：**计算一个返回增益矩阵 $\mathbf{K}$ 使其能够让系统达到 target。**

$$
\mathbf{u}=-\mathbf{K} \mathbf{x}
$$

当系统是具有二次成本函数的线性系统时，可以最佳地完成此工作。

##  非线性系统下的 LQR
将 LQR 应用于非线性系统，并使用有限的差异来执行，当您手头有一个易于访问的系统模拟时，这一点就会起作用。有趣的是，通过使用有限的差异，你可以得到这个工作，而无需自己计算出动态方程。

显然，非线性系统违反了 LQR 的第一个假设，即系统是线性的。这并不意味着我们不能应用它，它只是意味着它不会是最佳的。LQR 的性能有多差取决于两个重要因素：
1. 系统动态实际上是多么非线性，
2. 更新反馈增益矩阵 K 的频率。

要将 LQR 应用于非线性系统，我们只需闭上眼睛，假装系统动态是线性的，即它们适合该形式

$$
\dot{\mathbf{x}}=\mathbf{A} \mathbf{x}+\mathbf{B u}
$$

!!
我们将通过线性地近似于系统的实际动态来做到这一点。然后，我们将解决我们的增益值 K， 生成我们的控制信号为这个时间步骤， 然后重新近似动态再次在下一个时间步骤， 并解决 K 从新状态。系统动态越非线性，当我们离开状态时，用于生成控制信号的 K 就越不合适K;这就是为什么 LQR 的更新时间可以成为一个重要因素。

### 使用有限差分来近似系统动力学

如何近似一个非线性系统？
即，如何计算我们用来得到 K 的 A 和 B 矩阵？如果我们知道系统的动力学是

$$
\dot{\mathbf{x}}=f(\mathbf{x}, \mathbf{u})
$$

那么我们可以计算：

$$
\mathbf{A}=\frac{\partial f(\mathbf{x}, \mathbf{u})}{\partial \mathbf{x}}, \quad \mathbf{B}=\frac{\partial f(\mathbf{x}, \mathbf{u})}{\partial \mathbf{u}} .
$$

这个方程的解是很复杂的。有几种方法可以避开这一点。在这里，我们将假设所控制的系统是模拟的，或者我们至少可以访问一个准确的模型，并使用有限的差异方法来计算这些值。

有限差异背后的理念是通过在 x 点取样 f 并使用差值来计算 $\dot{f}(x)$ 来近似 x 点的函数 f 变化速率。下面是 1D 系统的图片：

![](https://pic4.zhimg.com/80/v2-97d5fe859b599e5d6cb636e578919701.png)

当前的状态 x 是蓝点，红点表示样本点 $x + \Delta x$ 和 $x - \Delta x$。然后，我们可以计算

$$
\dot{f}(x) \approx \frac{f(x+\Delta x)-f(x-\Delta x)}{2 \Delta x},
$$

看到在蓝色虚线中绘制的 x 点 f 的实际变化速率，以及使用绘制在红色虚线中的有限差异计算的近似变化速率。我们还可以看到，近似的衍生物只在 x（蓝点）附近准确。

回到我们的多维系统中，使用有限的差异来计算与状态和输入的导数，我们将逐一更改状态的每个维度和输入量，逐个计算每个维度和输入量。这里有一大块伪代码， 希望澄清这个想法：

```
eps = 1e-5
A = np.zeros((len(current_state), len(current_state))
for ii in range(len(current_state)):
    x = current_state.copy()
    x[ii] += eps
    x_inc = simulate_system(state=x, input=control_signal)
    x = current_state.copy()
    x[ii] -= eps
    x_dec = simulate_system(state=x, input=control_signal)
    A[:,ii] = (x_inc - x_dec) / (2 * eps)
 
B = np.zeros((len(current_state), len(control_signal))
for ii in range(len(control_signal)):
    u = control_signal.copy()
    u[ii] += eps
    x_inc = simulate_system(state=current_state, input=u)
    u = control_signal.copy()
    u[ii] -= eps
    x_dec = simulate_system(state=current_state, input=u)
    B[:,ii] = (x_inc - x_dec) / (2 * eps)
```

现在，我们能够生成我们的 A 和 B 矩阵， 我们有我们需要解决的一切， 我们的反馈增益矩阵 K！太棒了

### 关于在连续和离散系统设定中使用有限差分的注意

很重要的一点就是上面代码中的simulate_system函数究竟返回了什么。
1. 连续系统的话，系统被表示为 $
\dot{\mathbf{x}}=\mathbf{A} \mathbf{x}+\mathbf{B u}
$；使用 algebraic Riccati equation 的连续解求解 K，那么返回的就是 $\dot{\mathbf{x}}(t)$。

1. 离散系统被定义为 $\mathbf{x}(t+1)=\mathbf{A} \mathbf{x}(t)+\mathbf{B u}(t)$；使用 algebraic Riccati equation 的离散解求解 K，那么返回的就是 $\mathbf{x}(t+1)$。 


详细代码见 [Code](./1_LQR_code.py)
