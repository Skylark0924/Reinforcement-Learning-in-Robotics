## 记忆系统：走向新范式

[TOC]

### PR Structured Ⅲ：Memory systems 2018 – towards a new paradigm

[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6389412/pdf/nihms-1516287.pdf)

*J. Ferbinteanu*

> 声明：这是神经科学方面关于记忆结构的综述，由于本人专业限制，只是对该综述进行了翻译、理解，如若有误，烦请大佬们指正。

**我一个做决策推理的，为什么要读这篇神经科学的综述？**

- 因为现有决策方式都是建立在人对自身大脑的仿生学研究之上，决策推理以及因果推理的新思想一定会越来越多地来自神经科学的发现。
- 在经过了 PR Structured 系列前两篇文章的学习，我逐渐发现无论是结构化概率建模还是图神经网络，其本质是在维护一个显式的记忆结构。那么我何不深入了解一下真实的记忆结构呢？
- 近年来 Deepmind 提出了许多神经科学相关的创新，包括多巴胺、海马体、episodic control 等一系列papers，这得益于其拥有神经科学学位的创始人 Demis Hassabis 的高瞻远瞩。



## 1 Introduction

记忆系统的研究开始于20世纪，并逐渐形成了**古典主义记忆系统**的认知，即 **multiple memory systems theory (MMS)** 。MMS 假设大脑基于许多模块的**独立且并行运行来存储信息**，每个模块具有不同的属性、动力学和神经基础。该理论的许多证据来自dissociation 研究，这些研究表明对受限大脑区域的损害会导致固定类型的记忆缺陷。 
MMS在记忆研究中已经成为最流行的范例，知乎上大多数的文章介绍的都是MMS的思想，直到我看到这篇paper。

最近，一些实验结果表明，**记忆系统并不总是可分离的**，由单个记忆系统形成的表征形式可以促进一种以上类型的 memory-driven 的行为策略。
可以通过将动态网络角度应用于内存体系结构来解决此问题。
根据这种观点，记忆网络可以响应环境需求而重新配置或瞬时耦合。此时，位于特定记忆系统之下的神经网络可以充当独立单元或更高阶 meta-network 的集成组件。这种动态网络模型提出了一种方法，可以将质疑不同记忆系统概念的实验证据纳入模块化记忆系统结构中。该模型还提供了一个框架，以说明行为级别上展示的**记忆系统之间的复杂交互**。

> **Motto**: 
>
> ‘For I was afraid of memory; I knew that our memories and reminiscences are like icebergs. We see only the tips in passing, but the mass of land under water slips by unseen and inaccessible. We do not feel their immeasurable weight simply because they lie submerged in time, as in water. But, if we carelessly find ourselves in their way, we shall run aground against our own past and be shipwrecked.’ 
>
> Father Theoctist Nikolsky in Milorad Pavić-Dictionary of the Khazars



## 2 The Theory of MMS

多重记忆系统理论起源于1953年的一次内侧颞叶的双侧切除手术。

1953年9月1日，Connecticut 州 Hartford 医院的一名年轻神经病患者 **Henry Molaison (H.M.)** 接受了内侧颞叶的双侧手术切除，以控制癫痫病，并且醒来（几乎）治愈了癫痫病，但严重的健忘症持续了他一生。尽管他的健忘症很严重，但是却可以学习运动技能。

> 正如癌症研究中的 **海拉细胞** 一样，H.M.因为他的内侧颞叶切除术而闻名神经科学。

从现象学上说，不同类型的记忆是由不同的记忆模块或系统的活动导致的，每个模块或系统都有自己的处理方式和属性。 
H.M.表现出的记忆力不足的解释是对内侧颞叶的损害，内侧颞叶是专用于**事实和事件记忆系统**的神经基础，也称为 **declarative memory**。 

经过不断地发展完善，神经学家将古典主义的记忆系统理论归纳为 **medial temporal lobe (MLT) 模型**：

- **Declarative memory**：学习并记住事实信息或自身事件，包括 **episodic and semantic memory**；
- **Procedural memories**：运动技能的记忆和习惯，其特点是通过重复动作逐渐获得，在无意识回忆的情况下表达，并且缺乏灵活性。**存储于 neostriatum 新纹状体**；（俗称肌肉记忆？）
- **Affective memory**
- **Priming memory**
- **Classical conditioning involving skeletal responses**: 瞬膜条件反射
- **Non-associative learning**: 习惯和敏感性



## 3 Further theoretical developments

### 3.1 Attribute model of memory

这个模型包含三部分，每种记忆系统均由许多属性或记忆形式（time, place, response, reward value, sensory perception, to which language is added in humans）：

- **event-based memory system**: 处理新的和传入的信息，这些信息是以自我为中心，面向个人经历的事件。（类似MTL的 episodic memory）
- **knowledge- based memory system**: 包含永久性表征形式，存储在长期记忆中，是关于世界的一般性知识。（类似MTL的 semantic memory）
- **rule-based memory  system**：通过规则和以后续行动为目标的策略，整合 event-based 的系统和knowledge- based系统中的信息。

与MTL模型不同，Attribute model 不会根据记忆是 declarative vs. procedural，还是 conscious vs. unconscious 的方式来对记忆进行分类。它还强调了一种想法，即每种记忆都来自广泛的大脑网络的综合活动，大脑网络包含多个大脑结构。因此，从这个角度来看，记忆系统不具有确定其信息处理特征样式的核心结构。


### 3.2 Knowledge (representational) systems

记忆系统理论必须解决的问题是记忆系统的潜在无限增长。

根据这种观点，知识既支持行为又构成了记忆的基础，这是建设性过程的结果。

可以区分五个不同的知识系统：

- what happened
- where and when
- who was involved
- how to act (procedural knowledge)
- whether it was positive or negative (affective valence).

与古典记忆系统理论一样，Knowledge systems 具有独特的神经生物学基础，但是它们超越了长期记忆和短期记忆的划分，并且没有将感知与记忆分开。实验证明了这个观点，短期和长期记忆的区别并没有最初想象的那么明显。

顶叶后皮层通常与感知和注意力相关，在情节性记忆中也有作用，而传统上被认为具有执行功能的前额叶皮层代表感觉信息（Sestieri，Shulman and Corbetta，2017； Ester，Sprague and Serences，2015）。
内侧颞叶皮层的皮层上皮可能同时具有知觉和记忆功能（Baxter，2009；Suzuki，2009）。
Knowledge systems 模型反映了记忆领域中对大脑中信息处理的组织方式的认识不断变化。随之而来的是强调大的、整体的大脑网络而不是单个大脑区域在记忆中的作用，解释了失忆症患者为何保留远程记忆的原因，并合理地解释了记忆为何以及如何随时间变化（Nadel＆Moscovitch，1997； Moscovitch ＆Nadel，1998; Moscovitch et al.，2016）。

### 3.3 Processing modes model

该模型提出了基于处理模式定义记忆系统的建议，以消除将记忆划分为显式和隐式类别的问题。显式/隐式或声明性/程序上的区别是基于人类的研究，是指口头陈述记忆内容的能力。这种分类实际上并未包含在Schacter和Tulving的记忆系统定义中（Schacter＆Tulving，1994），实验最终对它提出了挑战。正如 dual-process 模型所提出的，在执行特定任务时，意识和潜意识的处理可以混合在一起，而相同的核心记忆结构可以同时支持显式和隐式记忆。

该模型根据三种不同的处理类型提出了三种不同的记忆系统：

- **rapid encoding of flexible associations**：以episodic memory为例， 是基于海马和新皮层的；
- **slow encoding of rigid associations**：包括 procedural memory, classical conditioning, and semantic memory，涉及基底神经节，小脑和新皮层；
- **rapid encoding of single items**：包括 familiarity and priming，涉及海马旁回和新皮层。

然而这个模型没有很好的兼容 emotional memories，因为对于情感记忆的学习，无法确定地划分快与慢。



### 3.4 Expanded parallel model of memory systems

制定该模型是为了解决以下事实：一个记忆系统获取的信息可以被另一个记忆系统利用。与记忆模块地竞争或合作交互不同，这表示记忆系统之间的存在其他类型的交互方式。

Expanded parallel model 强调了记忆系统的核心结构不仅可以接收信息，而且还可以将信息发送回皮层，皮层被认为可以长期存储这些记录。在共享感官输入 and/ or 运动输出的情况下，由不同记忆系统创建的皮质 memory traces 最终共享神经基础。因此，**记忆系统可以通过充当传递点或“集线器”的通用皮质表示进行间接通信**，以提供可以交换信息的神经基质。应当指出，“皮质”在这里并不意味着所有存储系统都可以投射到一个特定的大脑区域，而是一个通用术语，涵盖了诸如前额叶，内侧颞叶，运动等其他分布式皮质网络。

通过允许记忆模块之间的通信，Expanded parallel model 还可以说明记忆的多面却统一的性质：**我们可以回顾事实或事件，但同时我们也可以“带回”过去情节的procedural or emotional elements **，一切都紧密结合在一起。



### 3.5 Heterarchic model of memory systems

该模型背后的想法源自观察到的海马体病变产生的顺行性遗忘 (anterograde) 作用比逆行性遗忘 (retrograde) 作用小。层次结构是一种组织形式，其中级别不是像上级那样以上级或下级关系组织，而是循环地组织。这是通过各层之间的交叉连接实现的，其结果是产生非传递关系。在此模型的上下文中，层次结构被理解为不同记忆系统核心结构的集合，其层次结构由它们影响大脑活动和直接行为的能力决定。当这些核心结构之一被破坏时，层次结构就会改变。

海马体通常位于此层次结构的顶部，它**接收经过高度处理的多种感官模态作为输入**，并**将输出返回到相同的区域**（此处要注意：广泛的输入/输出是通过内嗅皮层实现的）。根据收到的经过高度处理的输入，**海马体会生成皮质中环境提示的联合表征**。然后，**将这些表征形式整合到几个可以影响运动动作的效应器系统中**（杏仁核，额叶皮层，纹状体和小脑）。

最初，提示连接的表征依赖于海马体的重新激活，但是随着这些连接的反复经验以及皮质表征的海马依赖性的激活，在皮质与其他记忆结构之间（除了海马）形成了关联。新的关联会影响内存记忆之间的交叉连接（这是赋予内存结构多变的特征），并使得在反复暴露后能够独立于海马体检索记忆信息。**如果在完全实现交叉连接之前损坏海马体，则会损害记忆。否则，信息检索将不受阻碍地进行。**

顺行性失忆和逆行性失忆之间的差异可以通过海马体在恢复分布式记忆轨迹中的关键作用来解释：

> 逆行性失忆更具破坏性，因为海马体是恢复记忆轨迹所必需的。

该模型还考虑了遮盖作用，即海马体干扰信息获取和存储在其他存储电路中的过程：**海马体形成的联合表征，使用了与其他记忆系统用于自身表征的相同的路径来控制运动输出。**



### 3.5 Evolutionary accretion model

**该模型构成了记忆系统理论的最清晰，最彻底的修订。**

该模型的基础是两个前提：

1. 在进化过程中的不同时间点开发的各种记忆系统，每个新系统都赋予了更高的环境适应性；
2. 皮层区域的专业化是其产生的表征，而不是心理过程。

从最旧到最新，列出了七个不同的记忆系统：

- **reinforcement**：包括基底神经节，杏仁核和小脑的记忆电路。代表刺激，反应和结果之间的关联；
- **navigation**：用于引导环境中的 journeys；
- **biased competition**：调解已经存在的记忆系统之间的竞争；
- **manual foraging**：用于将视觉信息转换为指标，并根据当前需求调整操作的价值；
- **feature**：有两个分别用于属性和度量的子系统；
- **goal**：通过将目标表征与上下文、动作和结果表示相结合来减少错误；
- **social-subjective**：代表自我和他人。

尽管仍基于模块化的记忆理论，但该解释与当前关于大脑记忆组织的观点有很大不同。在系统发育上，较新的记忆系统通过重新表征的过程集成了较旧的记忆系统生成的表征，即从较低级别表示的信息的高级抽象，从某种意义上说，记忆系统是按层次结构组织的。因此，产生更多抽象表征形式的“晚期”记忆系统，并不是独立且从头开始构建的，而是与功能增强的现有记忆系统集成在一起的。由于这种组织，“早期”存储系统在存储过程中具有更广泛的影响。例如，在进化上最古老的 reinforcement system 与所有其他记忆系统的功能息息相关。

根据该模型，显性记忆和习惯之间的二分法没有意义，因为它们都包含了基底神经节功能。**人类独有的显式记忆被认为不是源于中间时态记忆系统的功能，而是navigation, feature, goal, and social-subjective memory systems 之间的交互作用。**当前包括在颞颞叶内侧系统中的大脑区域（海马加上内嗅，皮层和海马旁皮质）产生不同类型的表征（scene memory and perception; conjunctions of objects and goals; feature conjunctions; conjunctions of objects and locations, respectively）**本质上既是记忆的又是感性的**。

