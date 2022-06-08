# DYNAMIC MOVEMENT PRIMITIVES PART 2: CONTROLLING END-EFFECTOR TRAJECTORIES

[[TOC]]

动态运动基元 （DMP） 框架专为轨迹控制而设计。碰巧，在以前的帖子中，我们已经建立了几个手臂模拟，已经成熟，可以将轨迹控制器置于顶层，这就是我们将在这篇文章中所做的。我们将在这里控制的系统是3关节机械臂模型与操作空间控制器（OSC），将最终效应力转化为联合扭矩。这里的 DMP 将控制手的 $(x，y)$ 轨迹，OSC 将负责将所需的手力转化为可以应用于手臂的扭矩。

## Controlling a 3 link arm with DMPs
要做到这一点很容易，我们将通过测量我们的 DMP 系统状态和设备状态之间的差异，从我们的 DMP 系统生成设备的控制信号，使用它将设备驱动到 DMP 系统的状态。

$$
u=k_{p}\left(y_{\mathrm{DMP}}-y\right)
$$

如果 $y_{\mathrm{DMP}}$ 是 DMP 系统的状态，$y$ 是设备的状态，并且$k_p$ 是位置误差增益项。

一旦我们拥有了这个，我们只是继续前进，并逐步推进我们的DMP系统，并确保控制信号上的增益值足够高，设备遵循DMP的轨迹。差不多就是这样， 只要运行 Dmp 系统到轨迹的末尾， 然后停止你的模拟。 为了演示 DMP 控制，我设置了 DMP 系统，以遵循 SPAUN 手臂遵循的相同数字轨迹。正如您所看到的，DMP 和操作空间控制的组合比我之前的实现要有效得多。

![](https://pic4.zhimg.com/80/v2-8f6073460ac0b560807a44f3f519cb1f.png)

## Incorporating system feedback
实施上述控制的问题之一是，我们必须小心 DMP 轨迹移动的速度，因为虽然 DMP 系统不受任何物理动力学的限制，但设备是。根据运动的大小，DMP 轨迹可能移动一英尺一秒或一英寸一秒。你可以在上面看到，手臂没有完全绘制出在DMP系统移动太快的地方和尖锐的角落所需的轨迹。在没有反馈的情况下解决这个问题的唯一方法是让 DMP 系统在整个轨迹上运动得更慢。相反，好的是说'只要设备状态在离你一定阈值的距离内，就尽可能快地走'，而这正是系统反馈的用武之地。

使用**系统反馈**实现此目标是非常简单的：如果设备状态偏离 DMP 状态，则减慢 DMP 的执行速度，以便设备有时间赶上。这样做，我们只需要将DMP的timestep $dt$ 乘以一个新的项：

$$
1 /\left(1+\alpha_{\text {err }}\left(y_{\text {plant }}-y_{\text {DMP }}\right)\right)
$$

当出现误差时，所有这些新项所做的都是减慢规范系统的速度，可以将其视为timestep的缩放。此外，该项的敏感性可以调制缩放项 $\alpha_{err}$ 关于设备和 DMP 状态之间的差异。当在运行中引入误差项时，我们可以通过查看规范系统的动力学来了解这如何影响系统：

![](https://pic4.zhimg.com/80/v2-6f591b15682871a37ca929d4505c3085.png)

当误差被引入时，系统的动力学会减慢。

让我们看看没有与有反馈项的区别。以下是无反馈绘制数字 3 ：
![](https://studywolf.files.wordpress.com/2013/12/3nofb.gif)

下面是有反馈项的：
![](https://studywolf.files.wordpress.com/2013/12/3wfb.gif)

这两个例子是将反馈词包含到您的 DMP 系统中的一个很好的例子。在第二种情况下，您仍然可以看到指定的轨迹没有精确追踪到，这其实可以启动 $\alpha_{err}$，使 DMP 时间步慢下来。

## Interpolating trajectories for imitation

当模仿轨迹时，可能存在一些问题，即有足够的样本点以及如何将其与规范系统的时间框架相适应。我采取的方法是始终以1秒间隔运行规范系统 ，每当通过轨迹时，应模仿该轨迹的 x 轴，使其在 0 到 1 之间。然后，我将其通过插值器，并使用生成的函数生成大量间隔良好的样本点，供 DMP 模仿器匹配。下面是代码：

```
# generate function to interpolate the desired trajectory
import scipy.interpolate
 
path = np.zeros((self.dmps, self.timesteps))
x = np.linspace(0, self.cs.run_time, y_des.shape[1])
 
for d in range(self.dmps):
 
# create interpolation function
path_gen = scipy.interpolate.interp1d(x, y_des[d])
 
# create evenly spaced sample points of desired trajectory
for t in range(self.timesteps):
path[d, t] = path_gen(t * self.dt)
 
y_des = path
```

## Direct trajectory control vs DMP based control
现在，使用上述插值函数，我们可以直接使用其输出来指导我们的系统。事实上，当我们这样做时，我们得到非常精确的末端执行器控制，比DMP控制更精确。原因是我们的 DMP 系统通过一组基函数逼近理想轨迹，这在近似过程中会失去了一些精度。因此，如果我们使用插值函数来驱动设备，可以得到指定的点。出现这种情况的时候，尤其是你试图模仿的轨迹特别复杂的时候。有办法通过更恰当地设置基函数来解决 DMP 问题，但如果您只是在寻找输入轨迹的精确复制（通常人们都是），则这是一个更简单的方法。以下是使用插值函数绘制的单个单词的比较：

![](https://studywolf.files.wordpress.com/2013/12/draw_word_traj1.gif)

下面是使用每个 DOF 具有 1，000 基函数的 DMP 系统绘制的相同单词：

![](https://studywolf.files.wordpress.com/2013/12/draw_word_dmp.gif)

我们可以看到，只要在这里使用插值函数，我们就能确定确切的路径，在使用 DMP 时，我们会有一些误差，并且此误差会随着所需轨迹的大小而增加。但这是完全按照给定的轨迹的情况，通常不是用在这种情况。DMP 框架的优点是轨迹是一个动态系统。这让我们可以仅仅通过改变目标而不是重新调整整个轨迹来扩展轨迹：

![](https://studywolf.files.wordpress.com/2013/12/draw_3_scaling.gif)

## Conclusions
DMP的力量在于其泛化性，而不是特定路径的精确再现。如果我有一个复杂的轨迹，我只想让末端执行器复现一次，那么应该只插值该轨迹并将坐标给到手臂控制器，而不是通过设置DMPs来完成整个过程。

！！！不过，绘制单词只是使用 DMP 框架的一个基本示例。这是一个非常简单的应用程序，真的不能公正地对待DMP的灵活性和权力。其他示例应用程序包括打乒乓球之类的内容。这是通过创建一个所需的轨迹，显示机器人如何摆动乒乓球桨，然后使用视觉系统跟踪传入的乒乓球的当前位置，并改变运动的目标，以动态补偿。也有一些真正棒东西与对象避免， 这是通过添加另一个术语与一些简单的动态到 Dmp 实现。在这里讨论，基本上你只是有另一个系统，移动你远离对象与强度相对于你与对象的距离。您还可以使用 DMP 来控制 PD 控制信号上的增益值，这对对象操作等内容很有用。！！！