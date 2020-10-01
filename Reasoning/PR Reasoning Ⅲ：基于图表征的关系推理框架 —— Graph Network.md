![Image](https://pic4.zhimg.com/80/v2-9c258943ac73f612b17471fea328608b.jpg)

# PR Reasoning Ⅲ：基于图表征的关系推理框架 —— Graph Network

本文基于提出 Graph Network 框架的论文

[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/pdf/1806.01261.pdf)
![Image](https://pic4.zhimg.com/80/v2-9f8fb14bb733c76705c794e1e079b254.png)

以及其在**机器人**中的应用论文

[Graph Networks as Learnable Physics Engines for Inference and Control]()
![Image](https://pic4.zhimg.com/80/v2-865d0b5b8108cfc3fe8da3e05defe578.png)
来总结 **Deepmind** 提出的这个**图表征的关系推理框架**。


## Preliminaries
**关系推理 (relational reasoning) 与关系归纳偏置 (relational inductive bias)** 详见

[【重磅综述】Relational Inductive bias 关系归纳偏置及其在深度学习中的应用]( 'card')

还可以稍微了解一下 **图神经网络**

[PR StructureⅠ：Graph Neural Network An Introduction](https://zhuanlan.zhihu.com/p/158984343 'card') 

对于结构化建模，也可以了解一下

[PR Structured Ⅱ：Structured Probabilistic Model](https://zhuanlan.zhihu.com/p/161703636 'card')

## Graph Networks
提出了一个 graph networks (GN) framework，该框架定义了一类**基于图结构表征的关系推理的函数**。GN框架概括并扩展了各种图神经网络，MPNN和NLNN方法，并支持从简单的构建模块构建复杂的体系结构。

**注意**：我们避免在 “graph networks” 中使用术语 “neural” 来反映**它可以用除神经网络以外的功能来实现**，尽管这里我们的重点是神经网络的实现。

> 这就仿佛 Bayesian Network 和 Bayesian Neural Network 的区别。所以 Deepmind 这一系列的工作，是旨在推广 Bayesian Network 形式的**结构化计算**到更通用的领域。

