# Long Range Neural Navigation Policies for the Real World

[Paper](http://arxiv.org/abs/1903.09870) | Code | 2019

*Ayzaan Wahid, Alexander Toshev, Marek Fiser and Tsang-Wei Edward Lee*

## Introduction

震惊！**第一篇real-world文**，终于不再是Simulation了！本文的思想类似 hierarchical RL：

- High-level policy：理解图像并给出长期规划；
- Low-level policy：将长期规划转化为安全且鲁棒的特定平台底层指令。

不过相应的，obs的信息也更多了。不再是单纯的视觉信息，而是SLAM处理过的信息。