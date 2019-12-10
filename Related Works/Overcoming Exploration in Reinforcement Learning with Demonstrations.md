# Overcoming Exploration in Reinforcement Learning with Demonstrations

Paper: https://arxiv.org/pdf/1709.10089.pdf

website: http://ashvin.me/demoddpg-website/

## Main Idea

They combine **demonstrations** with RL to overcome the **sparse reward** problem and successfully learn to perform long-horizon, multi-step robotics tasks with continuous control such as stacking blocks with a robot arm.

This method is based on **Deep Deterministic Policy Gradients** (DDPG) and **Hindsight Experience Replay** (HER, which is state-of-the-art method to overcome sparse rewards problem)

## Method

### Demonstration Buffer

They build a second replay buffer $R_D$ to store the demonstration in the same format as $R$ and train together with exploration experience.

### Behavior Cloning Loss (BC Loss)

$$
L_{B C}=\sum_{i=1}^{N_{D}}\left\|\pi\left(s_{i} | \theta_{\pi}\right)-a_{i}\right\|^{2}
$$

use only on the R_D examples for training actor. They use Equation 1 as an auxiliary loss for RL and find that it can improve learning significant. 
$$
\lambda_{1} \nabla_{\theta_{\pi}} J-\lambda_{2} \nabla_{\theta_{\pi}} L_{B C}
$$
where $\theta_\pi$ is the actor parameters. Note that we maximize J and minimize $L_{BC}$.

**Using this loss directly prevents the learned policy from improving significantly beyond the demonstration policy, as the actor is always tied back to the demonstrations.**

### Q-Filter

This is a trick to avoid **the possibility that demonstration is not better than current policy**. By creating a filter which just uses **BC Loss** when $Q(s_i;a_i)>Q(s_i;\pi(s_i))$.

### Resets to demonstration states

To overcome the sparse rewards and make it a long horizon task, they reset several episodes to start with given state in Demonstration Buffer and use the final state of it as a target.

### In conclusion

**Main method = (BC Loss + Q-Filter + HER + Reset) + DDPG**

## Experiment

### Task

Block Stacking: stack 2-6 blocks 

### Ablation Experiment 

- BC Loss: Without the behavior cloning loss, the method is significantly worse in every task they try. Since stacking the tower higher is risky and could result in lower reward if the agent knocks over a block that is already correctly placed, the agent chooses to stop after achieving a certain reward. BC Loss forces it to continue.
- Q-Filter: Accelerate learning process
- Reset: The same reason as BC Loss. Resetting from demonstration states alleviates this problem because the agent regularly experiences higher rewards. 

## Disadvantage

Impractical outside of simulation due to the low sample efficient. It will take about **1 million timesteps**, which is about **6 hours** of real world interaction time.

## Conclusion for the note

Actually, this paper shows me the tininess of humanity. By piling up such number of tricks and training nearly 1 million timesteps, the robot created by human can only stack 2-6 blocks tremblingly. What a complicated world!