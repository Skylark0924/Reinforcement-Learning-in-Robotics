# MuJoCo 详细使用指南


- [MuJoCo 详细使用指南](#mujoco-详细使用指南)
  - [Installation](#installation)
    - [Download](#download)
    - [Testing](#testing)
  - [Modeling](#modeling)
  - [Python API](#python-api)
    - [mujoco-py](#mujoco-py)


##  Installation
### Download
MuJoCo无需安装，在[官网下载对应版本](https://mujoco.org/download)，解压所得压缩包即可。其文件组成如下：

```
mujoco210
  bin     - dynamic libraries, executables, MUJOCO_LOG.TXT
  doc     - README.txt and REFERENCE.txt
  include - header files needed to develop with MuJoCo
  model   - model collection (extra models available on the Forum)
  sample  - code samples and makefile need to build them
```
### Testing
```
Windows:           simulate ..\model\humanoid.xml
Linux and macOS:   ./simulate ../model/humanoid.xml
```

## Modeling



## Python API
### mujoco-py






