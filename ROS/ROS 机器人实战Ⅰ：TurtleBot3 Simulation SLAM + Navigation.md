#! https://zhuanlan.zhihu.com/p/265184095
![](https://pic4.zhimg.com/80/v2-299bf7a99f200330501279cfd208fdf4.jpg)
# ROS 机器人实战Ⅰ：TurtleBot3 Simulation SLAM + Navigation
![](https://pic4.zhimg.com/80/v2-774260995a3dba20d25895f5b84ac482.png)
**强烈推荐原文网站 [ROBOTIS e-Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)**

> OS version: Ubuntu 18.04\
> ROS version: melodic

TurtleBot3支持在仿真中使用虚拟机器人进行编程和开发的开发环境，这就对吃土的年轻人们很有好了。为此，有两种开发环境，一种使用fake节点和3D可视化工具`RViz`，另一种使用3D机器人模拟器`Gazebo`。\
**fake节点**方法适用于使用机器人模型和运动进行测试，**但不能使用传感器**。如果需要测试SLAM和导航，**建议使用`Gazebo`**，它可以在仿真中使用IMU，LDS和摄像机等传感器。

**First of all，打开一个终端运行**
```
roscore
```
再开始下面的故事。

## 安装 TurtleBot3 Simulation
首先，为`TurtleBot3`控件**安装依赖软件包**。
```
$ cd ~/catkin_ws/src
$ git clone https://github.com/ROBOTIS-GIT/hls_lfcd_lds_driver.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
$ sudo apt-get install ros-melodic-rosserial-python ros-melodic-tf
```
**构建软件包**(每次安装完新的软件包都要记得要重新`catkin_make`一下)
```
$ cd ~/catkin_ws && catkin_make
```
如果catkin_make命令已完成而没有任何错误，则`TurtleBot3`的准备工作已经完成。

在工作空间**下载`turtlebot3_simulation`**
```
cd ~/catkin_ws/src/
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/catkin_ws && catkin_make
```

要在`rviz`中**启动并键鼠控制虚拟机器人**，请**在两个不同终端中**执行程序包中的turtlebot3_fake.launch文件turtlebot3_fake。
> `${TB3_MODEL}`是你正在使用的模型的名称`burger`，`waffle`，`waffle_pi`。\
> (可以都试试看，只是机器人的样子不同而已)\
> **接下来的例子都以 `burger` 模型为例，方便大家复制。**
![](https://pic4.zhimg.com/80/v2-caa6f19e05e92c0fcf326e3184d19afb.png)
```
$ #export TURTLEBOT3_MODEL=${TB3_MODEL} (可自选)
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_fake turtlebot3_fake.launch
```
![](https://pic4.zhimg.com/80/v2-4a7c06e8e2bd792399bedd606abc65c8.png)

`teleop`开头的一般都是**键盘控制**
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
![](https://pic4.zhimg.com/80/v2-78a9f2d8f0c23c205a741e4c502ab98c.png)
自己用 `w` `a` `s` `x` `d` 控制一下机器人玩玩吧。\
至此，安装完成（无报错的话）。


## Gazebo 模拟环境
### 空世界
在**新终端**运行
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
![](https://pic4.zhimg.com/80/v2-e970427c10409b44ed98bd9aead9a662.png)

### TurtleBot3世界
TurtleBot3 world主要用于SLAM和Navigation等测试。  
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch
```
![](https://pic4.zhimg.com/80/v2-bd14cc4e2bc15edf2414e2f31ca0e05b.png)
> `Crtl+Shift+鼠标` 可控制画面旋转

![](https://pic4.zhimg.com/80/v2-5793e8971c7180217b1be0ae4d93a5d8.png)

### TurtleBot3之家
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch
```
![](https://pic4.zhimg.com/80/v2-9e5418fecda72e23d1a0f4f72a4bfc40.png)

## SLAM 仿真
打开 `Gazebo`
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch
```
开始 SLAM
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
```
鼠标控制移动，以完成对整个地图的建图
```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
当看起来扫描地比较完全地时候，将所建地地图保存下来。（左gazebo，右rviz）
```
$ rosrun map_server map_saver -f ~/map
```
![](https://emanual.robotis.com/assets/images/platform/turtlebot3/simulation/virtual_slam.png)\
在文件夹中你可以看到如下图的地图：
![](https://emanual.robotis.com/assets/images/platform/turtlebot3/simulation/map.png)


## Navigation 仿真
打开 `Gazebo`
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch
```
用刚刚保存的地图开始导航
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml
```

### 估计初始位姿
首先，应该执行机器人的初始姿态估计。在`RViz`菜单中按`2D Pose Estimate`，会出现一个非常大的绿色箭头。将其移动到给定地图中实际机器人所在的位姿，并在按住鼠标左键的同时，将绿色箭头拖动到机器人正面朝向的方向：
1. 点击`2D Pose Estimate`按钮。
2. 单击地图中`TurtleBot3`所在的近似点，然后拖动光标以指示`TurtleBot3`正面朝向的方向。
3. 然后使用`turtlebot3_teleop_keyboard`来回移动机器人（使用后终止掉），以收集周围的环境信息并找出机器人当前在地图上的位置。
    ```
    $ export TURTLEBOT3_MODEL=burger
    $ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
    ```
4. 完成此过程后，机器人将绿色箭头指定的位置和方向用作初始姿势，以估算其实际位置和方向。如果图形未正确显示图形，请单击`2D Pose Estimate`按钮重复定位`TurtleBot3 `。

![](https://emanual.robotis.com/assets/images/platform/turtlebot3/simulation/virtual_navigation.png)

### 发送导航目标
如果按RViz菜单中的`2D Nav Goal`，则会出现一个非常大的绿色箭头。该绿色箭头是可以指定机器人目的地的标记。箭头的根是机器人的x和y位置，箭头所指的theta方向是机器人的方向。在机器人将要移动的位置单击此箭头：

1. 点击`2D Nav Goal`按钮。
2. 单击地图中的特定点以设置目标位置，然后将光标拖到TurtleBot最终应面向的方向。
3. 机器人将根据地图创建一条路径，以避开障碍物。
4. 然后，机器人会沿着这个路径移动。当突然检测到障碍物，机器人会避开障碍物而移动到目标点。

![](https://emanual.robotis.com/assets/images/platform/turtlebot3/navigation/2d_nav_goal.png)
