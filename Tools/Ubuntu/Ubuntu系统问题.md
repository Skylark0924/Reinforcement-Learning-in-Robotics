# Ubuntu 系统问题集锦

[[TOC]]
### 显卡驱动安装
**解决方案**
```
sudo apt-get purge nvidia*

sudo ubuntu-drivers autoinstall
```
重启机器
```
nvidia-smi
```



### 执行 apt-get install 报错Errors were encountered while processing:

**踩坑记录**
![](https://pic4.zhimg.com/80/v2-db28c41899af24ae18d7d794dd8ee44e.png)

**解决方案**
```
# 将info文件夹更名
sudo mv /var/lib/dpkg/info /var/lib/dpkg/info.bk  
# 新建一个新的info文件夹
sudo mkdir /var/lib/dpkg/info  

# 安装修复
sudo apt-get update  
sudo apt-get install -f 

# 上一步操作在info文件夹下生成一些文件，现将这些文件全部移到info.bk文件夹下
sudo mv /var/lib/dpkg/info/* /var/lib/dpkg/info.bk  
# 把新建的info文件夹删掉
sudo rm -rf /var/lib/dpkg/info  
# 恢复原有info文件夹，修改名字
sudo mv /var/lib/dpkg/info.bk /var/lib/dpkg/info  

sudo apt-get -o Dpkg::Options::="--force-overwrite" install locales
```

**Reference**\
https://blog.csdn.net/qq_42103502/article/details/105808323