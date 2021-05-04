# Docker Ⅴ：Docker与Nvidia Docker踩坑与解决方案记录

[[TOC]]

> 工欲善其事，必先利其器

## Docker


## Nvidia Docker
### GPG key Error
**踩坑日志**
当我在安装Nvidia Docker时，添加Nvidia的存储库和GPG密钥是没问题的，但是当我`sudo apt-get update`时，有以下报错：
```
W: GPG error: http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64  Release: The following signatures were invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
W: The repository 'http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64  Release' is not signed.
```

**初步分析**
很明显这是中国区的网络问题，这个[Reply](https://github.com/NVIDIA/nvidia-docker/issues/571#issuecomment-393783307)也佐证了这一观点。那么解决的思路就是换源。

### libnvidia-tls.so.* 缺失
运行官方测试镜像报错
```
$ docker run --runtime=nvidia --rm nvidia/cuda:11.0-base nvidia-smi
```
按理说，这行命令应该输出会显卡信息，说明nvidia docker成功创建并在内部正确执行了 nvidia-smi 命令。\
然而...

**踩坑日志**
```
docker: Error response from daemon: OCI runtime create failed: container_linux.go:349: starting container process caused "process_linux.go:449: container init caused \"process_linux.go:432: running prestart hook 1 caused \\\"error running hook: exit status 1, stdout: , stderr: nvidia-container-cli: detection error: open failed: /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.460.56: no such file or directory\\\\n\\\"\"": unknown.
```
**初步分析**
一开始，我天真地复制了这一堆报错去查，结果发现很多人都有这个问题，但是一顿讨论看下来谁都拿不出解决方案，详见[nvidia-docker/issues/1225](https://github.com/NVIDIA/nvidia-docker/issues/1225)。直到我看到`klueska`的这个[Reply](https://github.com/NVIDIA/nvidia-docker/issues/1225#issuecomment-694736952)，才意识到`docker: Error response from daemon: OCI runtime create failed: container_linux.go:349:...`这个前缀是Docker的报错信息，而真正的Nvidia Docker Error是`stderr`后面的那些，也就是：
```
nvidia-container-cli: detection error: open failed: /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.460.56: no such file or directory\\\\n\\\"\"": unknown.
```
那么，按照这个报错，我就找到了[nvidia-docker/issues/1404](https://github.com/NVIDIA/nvidia-docker/issues/1404)

还是参考`klueska`的[Reply](https://github.com/NVIDIA/nvidia-docker/issues/1404#issuecomment-720657408)，这是库文件与驱动版本不统一的问题。我的显卡驱动是440，`/usr/lib/`目录下搜索所有 `libnvidia*` 文件，却真的找到了几个后缀有460的文件，然后终端 `sudo rm <file_name>` 删除它们。

> 有趣的是，这个issue后面那个老哥也犯了我一样的错误，以为前缀一样就是 same issue了。

![](https://pic4.zhimg.com/80/v2-95327a38d4c9e7f94fe0d16fb82a68fa.png)

### nvidia-container-runtime: no such file or directory
**踩坑日志**
```
docker: Error response from daemon: OCI runtime create failed: unable to retrieve OCI runtime error (open /run/containerd/io.containerd.runtime.v1.linux/moby/d9f69fa38a697ffbc276caefea82f0e3262683c815f95b783ad835a19461696b/log.json: no such file or directory): fork/exec /usr/bin/nvidia-container-runtime: no such file or directory: : unknown.  
```
**解决方案**
```
sudo apt-get install nvidia-container-runtime
```
**Reference**
1. https://www.geek-share.com/detail/2794463648.html