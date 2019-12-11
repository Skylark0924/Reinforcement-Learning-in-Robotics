# The Predictron: End-To-End Learning and Planning

Paper: https://arxiv.org/pdf/1612.08810.pdf



![image-20191211163322741](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\Untitled.assets\image-20191211163322741.png)

## Main Idea

The main thought of this paper is that for a model-based algorithm the model is no need to be figurative for human since it is a model for agents. In addition, previous studies essentially separate modeling and planning(use the imagined trajectories to estimate), it will lead to undermatch. Therefore, they try to construct **a new architecture** called "**predictron**" which can integrate learning and planning into one end-to-end training procedure in order to build an **abstract** model only for agents.

**At every step, a model is applied to an internal state, to produce a next state, reward, discount, and value estimate. This model is completely abstract and its only goal is to facilitate accurate value prediction.**

## Keywords

Model-based, abstract modeling, end-to-end

## Method

### Components

The predictron is composed of four main components.

- **Representation**: A state representation $\textbf{s} = f(s)$ that encodes raw input s (this could be a history of observations, in partially observed settings, for example when f is a recurrent network) into an internal (abstract, hidden) state $\textbf{s}$.
- **Model**: $\textbf{s', r,}\mathbb{\gamma }=m(\textbf{s}, \beta)$ ($\beta$ is the noise for randomness)
- **Value**: $\textbf{v}=v(\textbf{s})$, represents the remaining internal return from internal state $\textbf{s}$ onwards.
- **Accumulator**: These internal rewards, discounts and values are combined together by an accumulator into an overall estimate of value $\textbf{g}$. 

### Accumulators

There are two kinds of accumulators (as shown above):

- **k-step predictron**: *predictron return*(preturn) $\mathbf{g}^{k}$, is the internal return obtained by accumulating $k$ model steps, plus a discounted final value $\mathbf{v^k}$ from the $k$th step:
  $$
  \mathbf{g}^{k}=\mathbf{r}^{1}+\gamma^{1}\left(\mathbf{r}^{2}+\gamma^{2}\left(\ldots+\gamma^{k-1}\left(\mathbf{r}^{k}+\gamma^{k} \mathbf{v}^{k}\right) \ldots\right)\right)
  $$

- **$\lambda$-predictron**: combines together many k-step preturns 
  $$
  \mathbf{g}^{\lambda}=\sum_{k=0}^{K} \boldsymbol{w}^{k} \mathbf{g}^{k}
  $$

  $$
  w^{k}=\left\{\begin{array}{ll}{\left(1-\lambda^{k}\right) \prod_{j=0}^{k-1} \lambda^{j}} & {\text { if } k<K} \\ {\prod_{j=0}^{K-1} \lambda^{j}} & {\text { otherwise }}\end{array}\right.
  $$

### Update

- **k-step predictron**: (the joint parameters $θ$ of the state representation, model, and value function)
  $$
  \begin{array}{l}{L^{k}=\frac{1}{2}\left\|\mathbb{E}_{p}[\mathbf{g} | s]-\mathbb{E}_{m}\left[\mathbf{g}^{k} | s\right]\right\|^{2}} \\ {\frac{\partial l^{k}}{\partial \boldsymbol{\theta}}=\left(\mathbf{g}-\mathbf{g}^{k}\right) \frac{\partial \mathbf{g}^{k}}{\partial \boldsymbol{\theta}}}\end{array}
  $$
  where $l^{k}=\frac{1}{2}\left\|\mathbf{g}-\mathbf{g}^{k}\right\|^{2}$ is the sample loss, $\mathbf{g}$ is a target(e.g. the Monte-Carlo return for the real environment)

- **$\lambda$-predictron**: (the parameters $\boldsymbol{η}$ of the λ-accumulator are updated to learn the weights $\boldsymbol{\omega}^k$)

  The λ-predictron combines many k-step preturns. So it can be written as follows:
  $$
  \begin{array}{l}{L^{0 : K}=\frac{1}{2 K} \sum_{k=0}^{K}\left\|\mathbb{E}_{p}[\mathbf{g} | s]-\mathbb{E}_{m}\left[\mathbf{g}^{k} | s\right]\right\|^{2}} \\ {\frac{\partial l^{0 : K}}{\partial \boldsymbol{\theta}}=\frac{1}{K} \sum_{k=0}^{K}\left(\mathbf{g}-\mathbf{g}^{k}\right) \frac{\partial \mathbf{g}^{k}}{\partial \boldsymbol{\theta}}}\end{array}
  $$
  the sample losses can be weighted by corresponding $\boldsymbol{\omega}^k$, and change into $\sum_{k=0}^{K} \boldsymbol{w}^{k}\left(\mathbf{g}-\mathbf{g}^{k}\right) \frac{\partial \mathbf{g}^{k}}{\partial \boldsymbol{\theta}}$. 

  he $\boldsymbol{λ}^k$ weights (that determine the relative weighting  $\boldsymbol{\omega}^k$ of the k-step preturns) depend on additional parameters $η$, which are updated so as to minimise a mean-squared
  $$
  \begin{array}{l}{L^{\lambda}=\frac{1}{2}\left\|\mathbb{E}_{p}[\mathbf{g} | s]-\mathbb{E}_{m}\left[\mathbf{g}^{\lambda} | s\right]\right\|^{2}} \\ {\frac{\partial l^{\lambda}}{\partial \boldsymbol{\eta}}=\left(\mathbf{g}-\mathbf{g}^{\lambda}\right) \frac{\partial \mathbf{g}^{\lambda}}{\partial \eta}}\end{array}
  $$

### Consistency updates

Since model-based method uses both real and imagined trajectories for learning, the λ-preturn $\mathbf{g}^{\lambda}$ can be used to adjusting each preturn $\mathbf{g}^{k}$
$$
\begin{array}{l}{L=\frac{1}{2} \sum_{k=0}^{K}\left\|\mathbb{E}_{m}\left[\mathbf{g}^{\lambda} | s\right]-\mathbb{E}_{m}\left[\mathbf{g}^{k} | s\right]\right\|^{2}} \\ {\frac{\partial l}{\partial \boldsymbol{\theta}}=\sum_{k=0}^{K}\left(\mathbf{g}^{\lambda}-\mathbf{g}^{k}\right) \frac{\partial \mathbf{g}^{k}}{\partial \boldsymbol{\theta}}}\end{array}
$$
This is especially relevant in the **semi-supervised** setting, where these consistency updates allow us to exploit the **unlabelled** inputs.

## Experiments

### Maze

The result of predictions are almost perfect!

![image-20191211202507981](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\Untitled.assets\image-20191211202507981.png)

## Advantage





## Disadvantage

