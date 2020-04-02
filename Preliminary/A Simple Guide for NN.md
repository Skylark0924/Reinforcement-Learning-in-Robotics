# Preliminary: A Simple Guide for NN

> 为了准备暑期实习(虽然大概率是去不了，在家闲着也是闲着)，开始从头整理基础知识。
>
> 今天从基础的神经网络和卷积神经网开始，当然后面会有RL的基础知识。
>
> 因为NN已经是老生常谈的知识了，所以会比较简略，会写一些公式化的东西。



### 神经网络的反向传播

<img src="./A Simple Guide for NN.assets/853467-20160630141449671-1058672778.png" alt="img" style="zoom: 50%;" />

> 除了输入层外的每个节点都包含 net 和 out 两个部分，分别代表激活前和激活后，这里激活函数使用 Sigmoid

#### 前向传播

1. **输入层->隐含层**

计算神经元 $h_1$ 的输入加权和：
$$
net_{h 1}=w_{1} * i_{1}+w_{2} * i_{2}+b_{1} * 1
$$
激活后的输出 $o_1$:
$$
o u t_{h 1}=\frac{1}{1+e^{-{n e t}_{h_1}}}
$$
其余隐层节点同理。

2. **隐含层->输出层**

计算输出神经元 $o_1 $ 的值：
$$
net_{o 1}=w_{5} *out_{h 1}+w_{6} *out_{h 2}+b_{2} * 1 
$$

$$
o u t_{o 1}=\frac{1}{1+e^{-{n e t}_{o_1}}}
$$

其余输出节点同理。

#### 反向传播

1. **计算总误差**

总误差：
$$
\begin{aligned}
E_{total}&=\sum\dfrac{1}{2}(target-output)^2\\
&=\dfrac{1}{2}(target_{o_1}-output_{o_1})^2+\dfrac{1}{2}(target_{o_2}-output_{o_2})^2
\end{aligned}
$$

2. **隐含层与输出层之间的权重更新**

以权重 $\omega_5$ 为例：
$$
\frac{\partial E_{t o t a l}}{\partial w_{5}}=\frac{\partial E_{t o t a l}}{\partial o u t_{o 1}} * \frac{\partial o u t_{o 1}}{\partial n e t_{o 1}} * \frac{\partial n e t_{o 1}}{\partial w_{5}}
$$
以偏导数表示 $\omega_5$ 对整体误差的影响，实际计算使用链式法则。

![img](./A Simple Guide for NN.assets/853467-20160630152018906-1524325812.png)

- $\dfrac{\partial E_{total}}{\partial {out}_{o_1}}$:

$$
E_{total}=\dfrac{1}{2}(target_{o_1}-output_{o_1})^2+\dfrac{1}{2}(target_{o_2}-output_{o_2})^2
\\
\dfrac{\partial E_{total}}{\partial {out}_{o_1}}=2 * \frac{1}{2}\left(\text { target }_{o 1}-o u t_{o 1}\right)^{2-1} *-1+0
$$

- $\dfrac{\partial out_{o_1}}{\partial {net}_{o_1}}$:

$$
out_{o_1}=\dfrac{1}{1+e^{-net_{o_1}}}\\
\dfrac{\partial out_{o_1}}{\partial {net}_{o_1}}=out_{o_1}(1-out_{o_1})
$$

- $\dfrac{\partial net_{o_1}}{\partial {\omega}_5}$:

$$
net_{o_1}=\omega_5*out_{h_1}+\omega_6*out_{h_2}+b_2*1\\
\dfrac{\partial net_{o_1}}{\partial {\omega}_5}=1*out_{h_1}*\omega_5^{(1-1)}+0+0
$$

- 最后三者相乘，按下式更新：

$$
\omega^{T+1}_5=\omega^T_5-\eta*\frac{\partial E_{t o t a l}}{\partial w_{5}^T}
$$

$\eta$为 learning rate，其余权重同理。

3. **输入层与隐层之间的权重更新**

<img src="./A Simple Guide for NN.assets/853467-20160630154317562-311369571.png" alt="img" style="zoom:50%;" />

- $\dfrac{\partial E_{t o t a l}}{\partial out_{h_1}}$：

$$
\begin{aligned}
\dfrac{\partial E_{t o t a l}}{\partial out_{h_1}}&=\dfrac{\partial E_{o_1}}{\partial out_{h_1}}+\dfrac{\partial E_{o_2}}{\partial out_{h_1}}\\
&=\dfrac{\partial E_{o_1}}{\partial {net_{o_1}}}*\dfrac{\partial {net_{o_1}}}{\partial out_{h_1}}+\dfrac{\partial E_{o_2}}{\partial {net_{o_2}}}*\dfrac{\partial {net_{o_2}}}{\partial out_{h_1}}
\end{aligned}
$$

这里的计算方法和过程2中一样。

- $\dfrac{\partial out_{h_1}}{\partial net_{h_1}}$:
  $$
  out_{h_1}=\dfrac{1}{1+e^{-net_{h_1}}}\\
  \dfrac{\partial out_{h_1}}{\partial {net}_{h_1}}=out_{h_1}(1-out_{h_1})
  $$
  

同样是求sigmoid函数的导数，和过程2中的一样。

- $\dfrac{\partial net_{h_1}}{\partial \omega_{1}}$:

$$
net_{h_1}=\omega_1*i_1+\omega_2*i_2+b_1*1\\
\dfrac{\partial net_{h_1}}{\partial {\omega}_1}=i_1
$$

- 三者相乘，按下式更新：
  $$
  \omega^{T+1}_1=\omega^T_1-\eta*\frac{\partial E_{t o t a l}}{\partial w_{1}^T}
  $$
  

其余权重同理。

按上述步骤多次迭代，即可得到最优解（理想情况下）。这就是神经网络的本质：**通过计算误差、不断修正权重以拟合输入输出的映射函数曲线**。

#### Reference:

[1] [一文弄懂神经网络中的反向传播法——BackPropagation](https://www.cnblogs.com/charlotte77/p/5629865.html)



### 卷积神经网络

![img](./A Simple Guide for NN.assets/20160707204048899.gif)

卷积过程这张动图展示的很清楚，摘自 [CNN笔记：通俗理解卷积神经网络](https://blog.csdn.net/v_JULY_v/article/details/51812459)

CNN和NN的结构很类似，只是这里的输入和权重都变成了二维的。此外，为了防止过拟合，引入了池化层进行降采样。这些知识都可以看[这篇blog](https://blog.csdn.net/v_JULY_v/article/details/51812459)复习复习。

#### 卷积层尺寸计算原理

摘自 [CNN中卷积层的计算细节](https://zhuanlan.zhihu.com/p/29119239)

- **输入矩阵**(in)格式：四个维度，依次为：样本数、图像高度、图像宽度、图像通道数

- **输出矩阵**(out)格式：与输出矩阵的维度顺序和含义相同，但是后三个维度（图像高度、图像宽度、图像通道数）的尺寸发生变化。

- **权重矩阵**(卷积核 kernel)格式：同样是四个维度，但维度的含义与上面两者都不同，为：卷积核高度、卷积核宽度、输入通道数、输出通道数（卷积核个数）

- **填充值**(zero-padding)：在外围边缘补充若干圈0，方便从初始位置以步长为单位可以刚好滑倒末尾位置，通俗地讲就是为了总长能被步长整除。

- **步长**(stride)：决定滑动多少步可以到边缘。

- **深度**(depth)：神经元个数，决定输出的depth厚度。同时代表滤波器个数。

- **输入矩阵、权重矩阵、输出矩阵这三者之间的相互决定关系**

- - 卷积核的输入通道数（in depth）由输入矩阵的通道数所决定。（红色标注）
  - 输出矩阵的通道数（out depth）由卷积核的输出通道数所决定。（绿色标注）
  - 输出矩阵的高度和宽度（height, width）这两个维度的尺寸由输入矩阵、卷积核、扫描方式所共同决定。计算公式如下。（蓝色标注）

![[公式]](./A Simple Guide for NN.assets/equation-1584587282205.svg)

#### 标准卷积计算举例

> 以 AlexNet 模型的第一个卷积层为例，
> \- 输入图片的尺寸统一为 227 x 227 x 3 （高度 x 宽度 x 颜色通道数），
> \- 本层一共具有96个卷积核，
> \- 每个卷积核的尺寸都是 11 x 11 x 3。
> \- 已知 stride = 4， padding = 0，
> \- 假设 batch_size = 256，
> \- 则输出矩阵的高度/宽度为 (227 - 11) / 4 + 1 = 55

![[公式]](./A Simple Guide for NN.assets/equation-1584587725413.svg)

#### 附：TensorFlow 中卷积层的简单实现

```python3
def conv_layer(x, out_channel, k_size, stride, padding):
    in_channel = x.shape[3].value
    w = tf.Variable(tf.truncated_normal([k_size, k_size, in_channel, out_channel], mean=0, stddev=stddev))
    b = tf.Variable(tf.zeros(out_channel))
    y = tf.nn.conv2d(x, filter=w, strides=[1, stride, stride, 1], padding=padding)
    y = tf.nn.bias_add(y, b)
    y = tf.nn.relu(y)
    return x
```

- 输入 x：[batch, height, width, in_channel]
- 权重 w：[height, width, in_channel, out_channel]
- 输出 y：[batch, height, width, out_channel]