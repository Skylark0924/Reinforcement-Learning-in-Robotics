# Docker Ⅲ：Nvidia Docker安装与测试指南

[[TOC]]

> 工欲善其事，必先利其器

**设备信息**

- Ubuntu 18.04
- Nvidia RTX2080 Ti 显卡
- Nvidia Driver Version: 440.33.01
- CUDA Version: 10.2

## 安装
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

```
$ sudo apt-get update
```

```
$ sudo apt-get install -y nvidia-docker2
```

```
$ sudo systemctl restart docker
```

## 测试
```
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
查看版本
```
$ sudo nvidia-docker version 

NVIDIA Docker: 2.6.0
Client: Docker Engine - Community
 Version:           19.03.12
 API version:       1.40
 Go version:        go1.13.10
 Git commit:        48a66213fe
 Built:             Mon Jun 22 15:45:49 2020
 OS/Arch:           linux/amd64
 Experimental:      false

Server: Docker Engine - Community
 Engine:
  Version:          19.03.12
  API version:      1.40 (minimum version 1.12)
  Go version:       go1.13.10
  Git commit:       48a66213fe
  Built:            Mon Jun 22 15:44:20 2020
  OS/Arch:          linux/amd64
  Experimental:     false
 containerd:
  Version:          1.2.13
  GitCommit:        7ad184331fa3e55e52b890ea95e65ba581ae3429
 runc:
  Version:          1.0.0-rc10
  GitCommit:        dc9208a3303feef5b3839f4323d9beb36df0a9dd
 docker-init:
  Version:          0.18.0
  GitCommit:        fec3683

```

## 卸载
```
$ sudo docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
$ sudo apt-get purge nvidia-docker
```