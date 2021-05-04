# Docker Ⅱ：管理与使用命令手册

[[TOC]]

> 工欲善其事，必先利其器

## 镜像
Docker 运行容器前需要本地存在对应的镜像，如果本地不存在该镜像，Docker 会从镜像仓库下载该镜像。
### 查看镜像
```
$ sudo docker images -a
``` 
或 
```
$ sudo docker images
```
### 下载镜像
与git类似：
```
$ sudo docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]
```

### 删除镜像
```
$ sudo docker rmi <your-own-image-id>
```
其中，`<your-own-image-id>`是你要删除的镜像ID。

**强制删除**\
上述删除可能会出现如下输出：
```
Error response from daemon: conflict: unable to delete fce289e99eb9 (must be forced) - image is being used by stopped container 3e92eacb4f6b
```
可以使用`-f`强制删除
```
$ sudo docker rmi <your-own-image-id> -f
```

**多个同时删除**\
镜像ID间空格隔开
```
$ sudo docker rmi <your-image-id-1> <your-image-id-2> ...
```

## 容器
### 查看容器
- `sudo docker ps` 仅列出在运行的容器；
- `sudo docker ps -a` 列出所有容器，包括停止运行的；
- `sudo docker ps -q` 仅列出在运行容器的ID，无其他信息；
- `sudo docker ps -q -a` 仅列出所有容器的ID，无其他信息；

### 开始运行
- 

### 停止运行
- `sudo docker stop $(docker ps -a -q)`停止所有容器；
- `sudo docker stop <your-own-container-id>`停止指定容器。



### 删除容器
```
$ sudo docker rm <your-own-container-id>
```
多个容器同时删除与镜像同理。\
删除全部容器：
```
$ sudo docker rm $(sudo docker ps -a -q)
```


## Reference
1. [Docker —— 从入门到实践](https://yeasy.gitbook.io/docker_practice/)
2. [如何删除 Docker 镜像和容器](https://chinese.freecodecamp.org/news/how-to-remove-images-in-docker/)



