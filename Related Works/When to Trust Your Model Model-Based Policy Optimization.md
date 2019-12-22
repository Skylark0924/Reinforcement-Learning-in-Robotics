# When to Trust Your Model: Model-Based Policy Optimization

![image-20191215201141993](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\When to Trust Your Model Model-Based Policy Optimization.assets\image-20191215201141993.png)

Paper: 

Code: https://github.com/JannerM/mbpo

Website: https://people.eecs.berkeley.edu/~janner/mbpo/

## Main Idea

- Formulate model-based RL algorithms with monotonic improvement guarantees
- Use short model-based rollouts



## Keywords

Model-based, monotonic improvement guarantee, branched rollout

## Preliminaries

1. 对于rollout的具体含义, 我一直比较困惑, 只知道是一次试验、一个trajectory。今天特意查了一下，得到了如下的解释：
    > The standard use of “rollout” (also called a “playout”) is in regard to an execution of a policy from the current state when there is some uncertainty about the next state or outcome - it is one simulation from your current state. The purpose is for an agent to evaluate many possible next actions in order to find an action that will maximize value (long-term expected reward).

    这个词源于 [Tesauro and Galperin NIPS 1997](http://papers.nips.cc/paper/1302-on-line-policy-improvement-using-monte-carlo-search.pdf) 这篇文章

2. Horizon: the number of time steps that we sample or simulate.

## Method


$$
\eta[\pi] \geq \hat{\eta}[\pi]-C
$$

$$
\eta[\pi] \geq \hat{\eta}[\pi]-\underbrace{\left[\frac{2 \gamma r_{\max }\left(\epsilon_{m}+2 \epsilon_{\pi}\right)}{(1-\gamma)^{2}}+\frac{4 r_{\max } \epsilon_{\pi}}{(1-\gamma)}\right]}_{C\left(\epsilon_{\left.m, \epsilon_{\pi}\right)}\right.}
$$

### Branched Rollout

$$
\eta[\pi] \geq \eta^{\mathrm{branch}}[\pi]-2 r_{\max }\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^{2}}+\frac{\gamma^{k}+2}{(1-\gamma)} \epsilon_{\pi}+\frac{k}{1-\gamma}\left(\epsilon_{m}+2 \epsilon_{\pi}\right)\right]
$$



### Model Generalization


$$
\eta[\pi] \geq \eta^{\mathrm{branch}}[\pi]-2 r_{\max }\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^{2}}+\frac{\gamma^{k} \epsilon_{\pi}}{(1-\gamma)}+\frac{k}{1-\gamma}\left(\epsilon_{m^{\prime}}\right)\right]
$$

$$
\hat{\epsilon}_{m^{\prime}}\left(\epsilon_{\pi}\right) \approx \epsilon_{m}+\epsilon_{\pi} \frac{\mathrm{d} \epsilon_{m^{\prime}}}{\mathrm{d} \epsilon_{\pi}}
$$

