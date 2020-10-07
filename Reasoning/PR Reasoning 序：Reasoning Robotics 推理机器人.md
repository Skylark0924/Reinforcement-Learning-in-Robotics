#! https://zhuanlan.zhihu.com/p/262568794
![Image](https://pic4.zhimg.com/80/v2-ada8c7b12921893969a0bef8cb4681fb.jpg)
# PR Reasoning 序：Reasoning Robotics 推理机器人学习路线与资源汇总

> 写了三章的 PR Reasoning 分章才意识到这好像是个很大的课题，需要整理一下资源和学习路线，所以今天补一个序章。欢迎评论区大佬补充。

## 学习路径
当我们想要让机器人具备推理能力时，我们实际是想实现什么效果呢？
- **状态推理**：在世界模型的前提下，当前传感器输入会产生什么结果？
- **动作推理**：接下来最可能采取什么动作？
- **动作的影响推理**：采取某个动作后会发生什么？
- **规划**：如何到达预定目标？ 
- **智能故障排除**

根据下一节推荐的几本书，可以归纳出如下学习路径：\
推理分两种：**符号主义的Logic Reasoning 和贝叶斯主义的 Probabilistic Reasoning**

- Logic Reasoning 从离散数学的谓词逻辑学起，first order logic等；

- Probabilistic Reasoning 的知识大致就是 PR 系列大部分的内容，包括概率图模型、精确和近似推断、inductive bias等。

## 书籍资源
1. 首先，要祭出 2016 年 **Springer Handbook of Robotics** 大辞典\
[Springer Handbook of Robotics - 2016](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Springer%20handbook%20of%20robotics_Siciliano%2C%20Khatib_2016.pdf 'card')\
其十四章介绍了 AI Reasoning Methods for Robotics。主要内容为 
    ![Image](https://pic4.zhimg.com/80/v2-62e065b9515a99cdee65c2f6d756bf59.png)

2. 这里还有一个2008版的，可以对照着看。\
[AI Reasoning Methods for Robotics - 2008](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/AI%20Reasoning%20Methods%20for%20Robotics_Hertzberg%2C%20Chatila_2008.pdf 'card')

3. [Reasoning Robots: The Art and Science of Programming Robotic Agents - 2005](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Reasoning%20Robots_Unknown_2005.pdf 'card')\
   作者给出了 reasoning robot 的定义：
    > A **reasoning robot** exhibits higher cognitive capabilities like following complex and long-term strategies, making rational decisions on a high level, drawing logical conclusions from sensor information acquired over time, devising suitable plans, and reacting sensibly in unexpected situations. All of these capabilities are characteristics of human-like intelligence and ultimately distinguish truly intelligent robots from mere autonomous machines.
    
    开头有一句话我很喜欢：
    > A fundamental paradigm of Artificial Intelligence says that **higher intelligence is
    grounded in a mental representation of the world and that intelligent behavior is the result of correct reasoning with this representation**.

## 论文资源
### Survey
1. [A Survey of Knowledge-based Sequential Decision Making under Uncertainty - 2020](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/A%20Survey%20of%20Knowledge-based%20Sequential%20Decision%20Making%20under%20Uncertainty_Zhang%2C%20Sridharan_2020.pdf)
2. [Open-World Reasoning for Service Robots - 2019](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Open-world%20reasoning%20for%20service%20robots_Jiang%20et%20al._2019.pdf)
3. [Artificial Intelligence for Long-Term Robot Autonomy: A Survey - 2018](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Artificial%20Intelligence%20for%20Long-Term%20Robot%20Autonomy%20A%20Survey_Kunze%20et%20al._2018.pdf)
4. [A Survey of Cognitive Architectures in the Past 20 Years - 2018](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/A%20survey%20of%20cognitive%20architectures%20in%20the%20past%2020%20years_Ye%2C%20Wang%2C%20Wang_2018.pdf)
5. [The Internal Reasoning of Robots - 2017](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/The%20internal%20reasoning%20of%20robots_Perlis%20et%20al._2017.pdf)
6. [Logical Formalizations of Commonsense Reasoning: A Survey Ernest - 2017](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Logical%20formalizations%20of%20commonsense%20reasoning%20A%20survey_Davis_2017.pdf)
9. [Integrated Uncertainty inKnowledgeModelling and Decision Making - 2015](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Integrated%20Uncertainty%20inKnowledgeModelling%20and%20Decision%20Making_Goebel_2015.pdf)
10. [Cognitive Robotics: Introduction to Knowledge Representation and Reasoning - 2001](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Cognitive%20Robotics%20Introduction%20to%20Knowledge%20Representation%20and%20Reasoning_Scherl_2001.pdf)

### Research Papers
1. [Purposive learning: Robot reasoning about the meanings of human activities | Science Robotics - 2019](https://github.com/Skylark0924/Reinforcement-Learning-in-Robotics/blob/master/Reasoning/BOOKS/Purposive%20learning%20Robot%20reasoning%20about%20the%20meanings%20of%20human%20activities_Cheng%20et%20al._2019.pdf)

