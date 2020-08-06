# Uncertainty-Aware Reinforcement Learning for Collision Avoidance

[Paper](http://arxiv.org/abs/1702.01182) | [Code](https://github.com/w0617/Uncertainty-aware-Reinforcement-Learning-for-Collision-Avoidance) | [Video](https://sites.google.com/site/probcoll) | 2017

*Gregory Kahn, Adam Villaflor, Vitchyr Pong, Pieter Abbeel, Sergey Levine*

> **Paper Reading**: Task of **Collision Avoidance**



## Introdution

Pieter 和 Sergey 组的文章，旨在利用对 uncertainty 的估计，使机器人能够**安全**的探索。本文是一个 model-based 结构，利用探索数据学习碰撞预测模型，并用此来评估模型的不确定性。在有可能碰撞的地方降低速度，在足够confidence的地方加快速度。



## Uncertainty-Aware Collision Prediction

简述一下思想：

- 以当前状态 $x_t$ ，obs $o_t$ 和控制序列 $u_{t:T+H}$ 为 input，经过NN以及logistic function L，输出机器人碰撞的概率
- risk-averse 碰撞评估器：
- 构建碰撞损失函数，并加入到总损失函数中；
- 为使NN获得正确的不确定性估计，本文使用了 bootstrapping 和 dropout 技术。