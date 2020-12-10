#! https://zhuanlan.zhihu.com/p/264917666
![Image](https://pic4.zhimg.com/80/v2-a60e43faa3dbb62c1ab5db2604be27c0.jpg)
# ROS Ⅱ：报错解决方案集锦

## rosdep: command not found
### **Solution**
大概率是真的没有 rosdep，那就装一下
```
sudo pip install -U rosdep
```
然后再
```
sudo rosdep init
rosdep update
```

### Link
[rosdep: command not found](https://answers.ros.org/question/32875/rosdep-command-not-found/)


## ImportError: No module named rospkg

### **Solution**
如果确实没有rospkg，那么就（注意pip对应的python版本要是ros使用的那个）
```
pip install rospkg
```
装完还是找不到，就终端运行（python版本指向ros使用的）
```
export PYTHONPATH=$PYTHONPATH:/usr/lib/python2.7/dist-packages
```
### Link
[ImportError: No module named rospkg](https://answers.ros.org/question/39657/importerror-no-module-named-rospkg/)

## module 'enum' has no attribute 'IntFlag'

### **Solution**
pip 直接 uninstall enum 是不行的，因为它不在 pip list 中，所以先查看enum的位置。终端运行ros所用的python别名：
```
python
```
```
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import enum
>>> enum.__file__
'/usr/lib/python2.7/dist-packages/enum/__init__.py'
```
然后在这个位置手动删除带 enum 的两个文件夹，用`sudo rm -rf enum....`

### Link
[AttributeError: module 'enum' has no attribute 'IntFlag'](https://www.jianshu.com/p/9c8237bb3598)


## rospack:command not found
运行
```
rosrun turtlesim turtle_teleop_key
```
报错
```
rospack:command not found
```
### **Solution**
```
sudo gedit  ~/.bashrc
```
在.bashrc中加入一行
```
source /opt/ros/melodic/setup.bash
```
### Link
[Linux报错笔记：rospack:command not found](https://www.cnblogs.com/polipolu/p/12836479.html)


## ModuleNotFoundError: No module named 'error'
```
skylark@lab-server:~$ rosrun rqt_graph rqt_graph
Traceback (most recent call last):
  File "/opt/ros/melodic/lib/rqt_graph/rqt_graph", line 5, in <module>
    from rqt_gui.main import Main
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rqt_gui/main.py", line 41, in <module>
    import rospy
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/__init__.py", line 47, in <module>
    from std_msgs.msg import Header
  File "/opt/ros/melodic/lib/python2.7/dist-packages/std_msgs/msg/__init__.py", line 1, in <module>
    from ._Bool import *
  File "/opt/ros/melodic/lib/python2.7/dist-packages/std_msgs/msg/_Bool.py", line 6, in <module>
    import genpy
  File "/opt/ros/melodic/lib/python2.7/dist-packages/genpy/__init__.py", line 34, in <module>
    from . message import Message, SerializationError, DeserializationError, MessageException, struct_I
  File "/opt/ros/melodic/lib/python2.7/dist-packages/genpy/message.py", line 48, in <module>
    import yaml
  File "/usr/lib/python2.7/dist-packages/yaml/__init__.py", line 2, in <module>
    from error import *
ModuleNotFoundError: No module named 'error'
```

### **Solution**
编辑 `rqt_graph` 文件
```
sudo gedit /opt/ros/melodic/lib/rqt_graph/rqt_graph
```
在文件第一行加入
```python
#!/usr/bin/env python2
```
其他软件打开遇到类似错误也是这样改。

### Link
[ModuleNotFoundError: No module named 'error'](https://answers.ros.org/question/314971/modulenotfounderror-no-module-named-error/)

## Couldn't find executable named rqt_graph below /opt/ros/melodic/share/rqt_graph

### Solution
把你的`catkin_ws`工作空间删了重新用你想用的python版本来一遍，按照ROS教程\
[安装并配置ROS环境](http://wiki.ros.org/cn/ROS/Tutorials/InstallingandConfiguringROSEnvironment)
### Link
[简书 2018-11-09[rosrun] Couldn't find executable named serial_node.py below /opt/ros/kinetic/share/ros...
](https://www.jianshu.com/p/ff9f3f9ad704)
