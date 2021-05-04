# Habitat Challenge提交指南
[[TOC]]
### Clone the challenge repository
```
$ git clone https://github.com/facebookresearch/habitat-challenge.git
$ cd habitat-challenge
```

### 安装 Nvidia Docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### 配置环境依赖
编辑`Dockerfile`，在名为`habitat`的conda环境下，添加依赖安装命令。以自定义安装`pytorch`为例：

```
FROM fairembodied/habitat-challenge:2021

# install dependencies in the habitat conda environment
RUN /bin/bash -c ". activate habitat; pip install torch"

ADD agent.py /agent.py
ADD submission.sh /submission.sh
```

### 将程序封装为Docker容器
```
$ sudo docker build . --file Objectnav.Dockerfile -t objectnav_submission
```

### 准备数据集
建立软链接
```
$ ln -f -s  /home/skylark/datasets/habitat/data/scene_datasets/mp3d \
   habitat-challenge-data/data/scene_datasets/mp3d
```


### Docker容器本地评估
```
$ sudo ./test_locally_objectnav_rgbd.sh --docker-name objectnav_submission
```
输出如下：
```
2021-05-04 14:51:40,617 Initializing dataset ObjectNav-v1
2021-05-04 14:51:40,638 initializing sim Sim-v0
2021-05-04 14:51:41,753 Initializing task ObjectNav-v1
2021-05-04 14:51:42,593 distance_to_goal: 6.3394588788350426
2021-05-04 14:51:42,593 success: 0.0
2021-05-04 14:51:42,593 spl: 0.0
2021-05-04 14:51:42,593 softspl: 0.01344610551192538
```

### 在线提交
安装 `EvalAI`：
```
# Installing EvalAI Command Line Interface
$ pip3h install "evalai>=1.3.5"

# Set EvalAI account token
$ evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
# Objectnav
$ evalai push objectnav_submission:latest --phase <phase-name>
```

`<phase-name>` 可选:
- **Minival phase**：与`./test_locally_{pointnav, objectnav}_rgbd.sh`一样，目的是进行完整性检查-确认我们的远程评估报告的结果与您在本地看到的结果相同。每个阶段每天最多允许100个团队提交。
我们将阻止和取消垃圾邮件服务器团队的资格。
- **Test Standard phase**：目的是充当公共排行榜，以建立最先进的技术；这就是用来报告论文结果的方法。每个小组在此阶段每天最多可以提交10份意见书。
- **Test Challenge phase**：用于确定挑战获胜者。到挑战赛提交阶段结束之前，每个团队总共可以提交5份报告。在CVPR的Embeded AI研讨会上宣布最终结果之前，不会公开此次拆分的结果。


## ObjectNav Baselines and DD-PPO Training Starter Code
### 安装 Habitat-sim 和Habitat-lab

### 下载数据集
Dataset: https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip\
> you should have the train and val splits at habitat-challenge/habitat-challenge-data/data/datasets/objectnav/mp3d/v1/train/ and habitat-challenge/habitat-challenge-data/data/datasets/objectnav/mp3d/v1/val/ respectively.


Scene_dataset: 软链接

### 下载Habitat2021 DDPPO预训练模型
下载模型至项目目录
```
$ wget https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo_objectnav_habitat2021_challenge_baseline_v1.pth
```

### 构建Docker容器
```
$ sudo docker build . --file Objectnav_DDPPO_baseline.Dockerfile -t objectnav_submission
```

### 本地评估
```
$ sudo ./test_locally_objectnav_rgbd.sh
```