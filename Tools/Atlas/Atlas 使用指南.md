# Atlas 使用指南

[[TOC]]

设备信息：

- 运行环境：Atlas 200DK
- 开发环境：Ubuntu 18.04
- Atlas Root 密码：sjtu1005SJTU
- 登录密码：登录名：HwHiAiUser 密码：Mind@123



## 总览

首先，总览Atlas 200DK结构

![](https://pic4.zhimg.com/80/v2-e743e6074acf1443a27f268d1d80bfdb.png)

## 制卡

### 官方教程

https://support.huaweicloud.com/Atlas200DK202/

### 步骤

#### 下载软件包

1. 下载Arm64 Server版本的Ubuntu18.04.5：https://repo.huaweicloud.com/ubuntu-cdimage/releases/18.04.5/release/ubuntu-18.04.5-server-arm64.iso 

   > **注意**：是ARM64不是AMD64\
   > 本例下载ubuntu-18.04.5-server-arm64.iso

2. 下载1.0.9.alpha版NPU驱动：https://www.hiascend.com/zh/hardware/firmware-drivers?tag=community
   > 本例下载 A200dk-npu-driver-20.2.0-ubuntu18.04-aarch64-minirc.tar.gz
   
3. 下载CANN软件包(ARM平台推理引擎软件包)：https://www.hiascend.com/zh/software/cann/community

   > 仅支持 20.2 和 3.3.0 开头的CANN\
   > 本例下载 Ascend-cann-nnrt_3.3.0.alpha006_linux-aarch64.run

#### 制卡
1. 将格式化后的SD卡放入读卡器，与Ubuntu连接
2. 切换`root`用户：
   ```
   $ su root
   ```
3. 安装依赖：
   ```
   $ apt-get install -y qemu-user-static binfmt-support gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
   $ apt-get update
   ```
4. 使用Ubuntu自带的python3安装依赖（不用anaconda是避免麻烦）：
   ```
   $ pip3 install pyyaml
   ```
5. 新建一个文件夹（例，`/home/skylark/make_sd`），并将下载的三个软件包拷贝进去。
6. 下载制卡脚本
   ```
   $ cd /home/skylark/make_sd

   # 下载制卡入口脚本 make_sd_card.py
   $ wget https://gitee.com/ascend/tools/raw/master/makesd/for_1.0.9.alpha/make_sd_card.py  # from gitee
   $ wget https://raw.githubusercontent.com/Ascend/tools/master/makesd/for_1.0.9.alpha/make_sd_card.py  # from github

   # 下载制作SD卡操作系统的脚本 make_ubuntu_sd.sh
   $ wget https://gitee.com/ascend/tools/raw/master/makesd/for_1.0.9.alpha/make_ubuntu_sd.sh  # from gitee
   $ wget https://raw.githubusercontent.com/Ascend/tools/master/makesd/for_1.0.9.alpha/make_ubuntu_sd.sh  # from github
   ```
7. root用户下查找SD卡的设备名称：
   ```
   $ fdisk -l
   ```
   根据SD卡的大小来判断，可能会有类似`/dev/sdc，/dev/sdc1`的两个名字，选用不带数字那个。

8. 运行制卡脚本
   ```
   $ python3 make_sd_card.py local /dev/sdc
   ```
   - `local`表示使用本地方式制作SD卡。
   - `/dev/sdc`为SD卡所在的USB设备名称。 
  
   ![](https://pic4.zhimg.com/80/v2-8e68c34f7091271d69260d8e5e437454.png)
   **制卡成功!**
9. 插入Atlas 200DK并上电。

## 连接
Atlas默认IP配置如下:
- USB 网卡：192.168.1.2
- NIC(NETWORK_CARD_DEFAULT_IP)网卡：192.168.0.2

官方推荐的配置流程是：
1. 先用USB将Atlas与Ubuntu服务器相连，修改USB虚拟网卡IP，并使其能够通过USB进行SSH连接；
2. USB连接并在Ubuntu服务器中进入Atlas后，修改eth设定，使其能够通过网络连接。

### USB连接
首先，修改连接Atlas对应USB的Ubuntu服务器的虚拟USB网卡IP
1. 创建新文件夹m，例如:
   ```
   $ mkdir /home/skylark/config_usb_ip/
   ```
2. 下载configure_usb_ethernet.sh脚本
   ```
   $ cd /home/skylark/config_usb_ip/
   $ wget https://gitee.com/ascend/tools/raw/master/configure_usb_ethernet/for_20.1/configure_usb_ethernet.sh  # from gitee
   $ wget https://raw.githubusercontent.com/Huawei-Ascend/tools/master/configure_usb_ethernet/for_20.1/configure_usb_ethernet.sh  # from github
   ```
3. 切到`root`用户：
   ```
   $ su root
   $ bash configure_usb_ethernet.sh -s <ip_address>
   # 或指定网卡名称，通过ifconfig以及插拔atlas来查询对应USB网卡名称
   $ bash configure_usb_ethernet.sh -s <usb_nic_name> <ip_address>
   ```
   输出如下，即为修改成功。
   ![](https://pic4.zhimg.com/80/v2-b62fbf5ade8da0fd565004aeb80670c4.png)
然后，即可用USB连接`192.168.1.2`：
```
$ ssh HwHiAiUser@192.168.1.2
```
默认密码为`Mind@123`


### 网线连接

1. 切换`root`用户
2. 编辑网络配置文件
   ```
   vi /etc/netplan/01-netcfg.yaml
   ```
3. 修改eth0网卡的IP地址获取方式为DHCP
   ```
      eth0:
         dhcp4: true
         addresses: []
         optional: true
   ```
   `:wq`保存退出


- 内网：192.168.1.143  
- 外网：202.120.48.24:30200

插网线，直接 ssh 连接

```
ssh HwHiAiUser@192.168.1.143
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

