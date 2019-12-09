# End-to-End Robotic Reinforcement Learning without Reward Engineering

paper: https://arxiv.org/pdf/1904.07854.pdf

website: https://sites.google.com/view/reward-learning-rl/home

## Main Idea

They propose an approach for removing the need for manual engineering of reward specifications by enabling a robot to learn from a modest number of examples of successful outcomes, followed by actively solicited queries, where the robot shows the user a state and asks for a label to determine whether that state represents successful completion of the task. Learning requires minimal user supervision and **only 1-4 hours** of interaction time, which is substantially less than that of prior work.

> The most inconceivability is that this methods can directly train on real robots and just need a small number of additional queries from a human user to judge whether the action is positive. Let's analyze how they achieve this goal.

## Preliminaries

### Maximum Entropy RL

In this paper, they use **off-policy soft actor-critic** (SAC), a kind of maximum entropy RL algorithm. 
$$
J(\pi)=\sum_{t=0}^{T} E_{\tau \sim \pi}\left[r\left(\mathbf{s}_{t}, \mathbf{a}_{t}\right)-\log \pi\left(\mathbf{a}_{t} | \mathbf{s}_{t}\right)\right]
$$
Maximum entropy RL have two benefits:

- tend to produce stable and robust policies for real-world reinforcement learning
- makes it straightforward to integrate their method with VICE

### Classifier-Based Rewards

When using image observations, define a reward function is difficult. An usual alternative is to use **goal classifier**, where users provide a dataset of example state which contains success or failure images. This is actually a basic binary classification problem. However, prior works generally need a comprehensive set of negative examples covering the entire state space otherwise it will lead to a crash. 

If the classifier provides a distribution $p_g(y|\textbf{s})$, then a particularly convenient form for the reward is given by $\log p_g(y|\textbf{s})$.

## RL with Active Queries (RAQ)

### Query Mechanism

#### Which state to query?

To make the query practical, they need to decide which state should be labeled. They found that the most effective mechanism to select which states to label was to select the previously-unlabeled states with the **highest probability** of success according to the classifier. 
$$
k=\arg\max_t \log p_g(y=1|\textbf{s}_t) \quad \forall t \text{ since last query}
$$
It is said this approach is better than maximum entropy heuristic. 

#### How often to query?

25-75 active queries for a single run.

### Classifier-based RL with active queries

It is really simple, just add the labeled state into dataset $\mathcal{D}$ after each query and then continue training.

## Off-policy VICE with Active Queries

### VICE

![image-20191208214135640](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\End-to-End Robotic Reinforcement Learning without Reward Engineering.assets\image-20191208214135640.png)VICE uses **Adversarial inverse reinforcement learning (AIRL)** to tame the high-dimensional and continuous state space. alternates between training a discriminator to discriminate between the positive examples and the current policy’s rollouts, and optimizing the policy with respect to the maximum entropy objective in Equation 1, using $\log p(y_t = 1|s_t, a_t)$ as the reward. The discriminator in AIRL is parameterized by $ψ$ and given by the following equation:
$$
D_{\psi}(\mathbf{s}, \mathbf{a})=\frac{\exp \left(f_{\psi}(\mathbf{s}, \mathbf{a})\right)}{\exp \left(f_{\psi}(\mathbf{s}, \mathbf{a})\right)+\pi(\mathbf{a} | \mathbf{s})}
$$

While the basic VICE is on-policy which use TRPO and is difficult for real-world learning.

### Off-Policy VICE

As I said above, they decide to use SAC as a off-policy improvement for VICE. Since AIRL is said to have an ability to **drop the importance weights both in theory and in practice**, there is no need to use importance weights.

### Off-Policy VICE-RAQ

It is naturally to use RAQ for enlarging the set of positive data. To integrate RAQ with VICE, they simply add the active queries like what I said in "Classifier-based RL with active queries". Then use it to update $f_\psi$, the policy and the Q-function.

![image-20191209102302915](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\End-to-End Robotic Reinforcement Learning without Reward Engineering.assets\image-20191209102302915.png)

### VICE-RAQ for manipulation

The output probabilities of the classifier need to be smoothly transitioned, so that they employ **mixup** regularization for smoothing the classifier predictions. It can generate the following **virtual** training distribution:
$$
\begin{array}{l}{\tilde{\mathbf{s}}=\lambda \mathbf{s}_{i}+(1-\lambda) \mathbf{s}_{j}} \\ {\tilde{y}=\lambda y_{i}+(1-\lambda) y_{j}}\end{array}
$$
where $s_i, s_j$ are any two inputs in the replay buffer and $y_i, y_j$ are the corresponding labels. $\lambda$ is a hyperparameter represents the level of mixup.

## Real-world experiments

They try three tasks: 

Each of them starts with 80 successful examples.

|      Tasks       | Number of Queries | Query Frequency | Total timesteps | Total Time |
| :--------------: | :---------------: | :-------------: | :-------------: | :--------: |
|  Visual Draping  |        50         |       500       |       25k       |     4h     |
|  Visual Pushing  |        25         |       250       |      6.2k       |   90min    |
| Visual Bookshelf |        75         |       250       |       19k       |     3h     |

![image-20191209104206015](D:\Github\Reinforcement-Learning-in-Robotics\Related Works\End-to-End Robotic Reinforcement Learning without Reward Engineering.assets\image-20191209104206015.png)

## Conclusion for the note

VICE-RAQ is really a simple idea. It is mainly based on the previous study (VICE) which uses AIRL. The RAQ they proposed is just a method of Data Augmentation. It is unbelievable to achieve these complex goals by this simple idea. Perhaps the success is owed to the breakthrough of **AIRL**, and I really need to read their code carefully. If I have any new discoveries, I will update this blog.