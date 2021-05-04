# Docker Ⅰ：安装与测试指南

[[TOC]]

> 工欲善其事，必先利其器

**设备信息**

- Ubuntu 18.04
- Nvidia RTX2080 Ti 显卡
- Nvidia Driver Version: 440.33.01
- CUDA Version: 10.2

## 安装



## 测试
### 启动 Docker
```
$ sudo systemctl enable docker
$ sudo systemctl start docker
```
(无输出)


### 测试安装
```
$ docker run -rm hello-world
```
输出如下:
```
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

```
如出现上述输出，即为安装正常。\
如果是第一次打开 hello-world镜像，docker会自动拉取这个库，所以在上述输出之前还会出现：
```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
b8dfde127a29: Pull complete 
Digest: sha256:f2266cbfc127c960fd30e76b7c792dc23b588c0db76233517e1891a4e357d519
Status: Downloaded newer image for hello-world:latest
```

## 卸载
```
$ sudo apt-get remove docker docker-engine docker.io
```

## Reference
1. [Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)