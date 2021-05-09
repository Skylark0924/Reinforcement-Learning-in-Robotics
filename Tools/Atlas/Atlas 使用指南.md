# Atlas 使用指南

[TOC]

设备信息：

- 运行环境：Atlas 200DK
- 开发环境：Ubuntu 18.04
- Atlas Root 密码：sjtu1005SJTU
- 登录密码：登录名：HwHiAiUser 密码：Mind@123



## 总览

首先，总览Atlas 200DK结构



<img src="D:\Github\Reinforcement-Learning-in-Robotics\Tools\Atlas\Atlas软硬件架构.png" style="zoom:20%;" />

## 制卡

### 官方教程

https://support.huaweicloud.com/Atlas200DK202/

### 步骤

#### 下载软件包

1. 下载Arm64 Server版本的Ubuntu18.04.5：https://repo.huaweicloud.com/ubuntu-cdimage/releases/18.04.5/release/ubuntu-18.04.5-server-arm64.iso 

   > 注意：是ARM64不是AMD64

2. 下载1.0.9.alpha版NPU驱动：https://repo.huaweicloud.com/ubuntu-cdimage/releases/18.04.5/release/ubuntu-18.04.5-server-amd64.iso

3. 下载CANN软件包(ARM平台推理引擎软件包，Ascend-cann-nnrt_3.3.0.alpha006_linux-aarch64.run)：https://www.hiascend.com/zh/software/cann/community

   > 仅支持 20.2 和 3.3.0 开头的CANN

#### 制卡





## 连接

### 网线连接

IP

- 内网：192.168.1.143  
- 外网：202.120.48.24:30200

插网线，直接 ssh 连接

```
ssh HwHiAiUser@192.168.1
```







## 安装Mindstudio

https://support.huaweicloud.com/MindStudioC76/index.html



## 安装CANN

https://support.huaweicloud.com/instg-cli-cann330-alpha002/atlasrun_03_0002.html





## 模型转译

开源模型库 https://gitee.com/ascend/modelzoo 



->.om



## MindStudio 模型加载

### 启动MindStudio

```
cd /home/lab/Ascend/MindStudio/bin
./MindStudio.sh
```



加载.om

