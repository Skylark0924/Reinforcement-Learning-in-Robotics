## MuJoCo建模指南

> 机器人仿真环境建模这里一直是我的co-worker在负责，网上的相关资源真的少到哭，还好大佬很强，分分钟解决了。

本来知乎上已经有一篇很好的教程了

[MuJoCo的机器人建模](https://zhuanlan.zhihu.com/p/99991106)

但是  写的主要是**如何从 Solidworks/其他三维软件 ->  URDF模型 -> MuJoCo模型**的过程。而我们一开始做的时候还没有这篇文章，所以我们直接从 .xml 写起，直接在MuJoCo上画出了模型。这听上去是很麻烦，但是有些模型的配合关系比较复杂，结构却相对简单（一些自己设计的非标机器人），反倒不如直接写 .xml。此外本文还会介绍OpenAI 开发的 mujoco-py的API及用法。因此，本文内容如下：

- **如何写好MuJoCo模拟器的 .xml 文件**；
- **如何使用 mujoco-py。**

### 如何写好 XML

按理说一个MuJoCo模拟器是包含三部分的：

- STL文件，即三维模型；
- XML 文件，用于定义运动学和动力学关系；
- 模拟器构建py文件，使用mujoco-py将XML model创建成可交互的环境，供强化学习算法调用。

但是STL也是分块集成在XML中的，所以本文就不提STL的事情了。



#### XML结构

XML主要分为以下三个部分：

- `<asset>` ： **用`<mesh>` tag导入STL文件**；
- `<worldbody>`：**用`<body> `tag定义了所有的模拟器组件**，包括灯光、地板以及你的机器人；
- `<acutator>`：**定义可以执行运动的关节**。定义的顺序需要按照运动学顺序来，比如多关节串联机器人以工具坐标附近的最后一个关节为joint0，依此类推。

```xml
<mujoco model="example">
    <!-- set some defaults for units and lighting -->
    <compiler angle="radian" meshdir="meshes"/>
 
    <!-- 导入STL文件 -->
    <asset>
        <mesh file="base.STL" />
        <mesh file="link1.STL" />
        <mesh file="link2.STL" />
    </asset>
 
    <!-- 定义所有模拟器组件 -->
    <worldbody>
        <!-- 灯光 -->
        <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1" />
        <!-- 添加地板，这样我们就不会凝视深渊 -->
        <geom name="floor" pos="0 0 0" size="1 1 1" type="plane" rgba="1 0.83 0.61 0.5"/>
        <!-- the ABR Control Mujoco interface expects a hand mocap -->
        <body name="hand" pos="0 0 0" mocap="true">
            <geom type="box" size=".01 .02 .03" rgba="0 .9 0 .5" contype="2"/>
        </body>
 
        <!-- 构建串联机器人 -->
        <body name="base" pos="0 0 0">
            <geom name="link0" type="mesh" mesh="base" pos="0 0 0"/>
            <inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/>
 
            <!-- nest each child piece inside the parent body tags -->
            <body name="link1" pos="0 0 1">
                <!-- this joint connects link1 to the base -->
                <joint name="joint0" axis="0 0 1" pos="0 0 0"/>
 
                <geom name="link1" type="mesh" mesh="link1" pos="0 0 0" euler="0 3.14 0"/>
                <inertial pos="0 0 0" mass="0.75" diaginertia="1 1 1"/>
 
                <body name="link2" pos="0 0 1">
                    <!-- this joint connects link2 to link1 -->
                    <joint name="joint1" axis="0 0 1" pos="0 0 0"/>
 
                    <geom name="link2" type="mesh" mesh="link2" pos="0 0 0" euler="0 3.14 0"/>
                    <inertial pos="0 0 0" mass="0.75" diaginertia="1 1 1"/>
 
                    <!-- the ABR Control Mujoco interface uses the EE body to -->
                    <!-- identify the end-effector point to control with OSC-->
                    <body name="EE" pos="0 0.2 0.2">
                        <inertial pos="0 0 0" mass="0" diaginertia="0 0 0" />
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
 
    <!-- 定义关节上的执行器 -->
    <actuator>
        <motor name="joint0_motor" joint="joint0"/>
        <motor name="joint1_motor" joint="joint1"/>
    </actuator>
 
</mujoco>
```

#### (world)body组件

```XML
<!-- name: 实体名，pos: 与上一个实体的偏移量(推荐在body上设置，后面的pos都可以写作"0,0,0"，便于调试) (注意是相对位置！) -->
<body name="link_n" pos="0 0 1">
    
    <!-- name: 关节名，pos: 与上一个实体的偏移量 -->
    <joint name="joint_n-1" axis="0 0 0" pos="0 0 0"/>
    
    <!-- 定义其几何特性。引用导入的shoulder.STL模型并命名，可以用euler来旋转STL以实现两个实体间的配合 -->
    <geom name="link_n" type="mesh" mesXMLh="shoulder" pos="0 0 0" euler="0 0 0"/>
    
    <!-- 定义实体的惯性。如果不写inertial其惯性会从geom中推断出来 -->
    <inertial pos="0 0 0" mass="0.01" diaginertia="0 0 0"/>
 
    <!-- nest the next joint here -->
</body>
```

其余属性名称及含义见MuJoCo官方给出的XML API：

[MuJoCo XML Reference](http://www.mujoco.org/book/XMLreference.html)

> 这个网页没有跳转的功能看着真的难受。

**(world)body组件下属组件：**inertial, joint, freejoint, geom, site, camera, light, composite

其中**composite应该是最amusing的部分了：**这不是模型元素，而是一个宏，它用于表示一个复合对象。仅当模型在局部坐标中时才能定义复合对象。在全局坐标中使用它们会导致编译器错误。旨在模拟**粒子系统、绳索、布料和软体**。

<img src="./MuJoCo机器人建模教程.assets\particle2.png" alt="img" style="zoom:33%;" />

其下又包括一些下属组件：

- joint, geom, site
- tendon：用于定义绳子、线

<img src="./MuJoCo机器人建模教程.assets\grid1.png" alt="img" style="zoom: 50%;" />

```xml
<body name="B10" pos="0 0 1">
    <freejoint/>
    <composite type="rope" count="21 1 1" spacing="0.04" offset="0 0 2">
        <joint kind="main" damping="0.005"/>
        <geom type="capsule" size=".01 .015" rgba=".8 .2 .1 1"/>
    </composite>
</body>
```

- skin：用于定义二维网格面的皮肤、贴图

<img src="./MuJoCo机器人建模教程.assets\grid2.png" alt="img" style="zoom:50%;" />

```xml
<composite type="grid" count="9 9 1" spacing="0.05" offset="0 0 1">
    <skin material="matcarpet" inflate="0.001" subgrid="3" texcoord="true"/>
    <geom size=".02"/>
    <pin coord="0 0"/>
    <pin coord="8 0"/>
</composite>
```

- pin：将一个实体固定在某处。例如挂在墙上的一块布。

[具体看链接的Composite objects部分](http://www.mujoco.org/book/modeling.html#CComposite)



### 如何使用 mujoco-py

mujoco-py是允许python3使用MuJoCo的一个中间件，

github地址：

[Github](https://github.com/openai/mujoco-py)

文档地址：

[Documentation](https://openai.github.io/mujoco-py/build/html/index.html)

以下给出一个简易模板：

```python
from mujoco_py import load_model_from_path, MjSim
class my_env():
    def __init__(self, env, args):
        # super(lab_env, self).__init__(env)
        # 导入xml文档
        self.model = load_model_from_path("your_XML_path.xml")
        # 调用MjSim构建一个basic simulation
        self.sim = MjSim(model=self.model)
        
	def get_state(self, *args):
    	self.sim.get_state()
        # 如果定义了相机
        # self.sim.data.get_camera_xpos('[camera name]')
        
    def reset(self, *args):
        self.sim.reset()
        # 如果定义了相机
            
    def step(self, *args):
        self.sim.step()
```

#### 高级用法

使用 PyMjData 数据类，来对模型进行控制或特定状态读取。上面定义的 `self.sim.data` 就是一个MjSim自带的可选PyMjData变量，具体可以调用的属性太多了，详见链接：

[PyMjData](https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data)



### Reference

1. [BUILDING MODELS IN MUJOCO](https://studywolf.wordpress.com/2020/03/22/building-models-in-mujoco/)
2. [MuJoCo XML Reference](http://www.mujoco.org/book/XMLreference.html)
3. [OpenAI mujoco-py](https://openai.github.io/mujoco-py/build/html/index.html)