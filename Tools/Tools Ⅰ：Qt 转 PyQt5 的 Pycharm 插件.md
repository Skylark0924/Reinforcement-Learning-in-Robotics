#! https://zhuanlan.zhihu.com/p/259564109
![Image](https://pic4.zhimg.com/80/v2-8ed5a3b27f24ae0a423da00918581c30.jpg)

# Tools Ⅰ：如何用 PyQt5 和 Qt Designer 在 Pycharm 中愉快地开发软件

转载自 [两个轮子](https://me.csdn.net/qq_40666028) 的 CSDN  https://blog.csdn.net/qq_40666028/article/details/81069878

封面图自 [Link](https://www.google.com/url?sa=i&url=https%3A%2F%2Fsteamcommunity.com%2Fsharedfiles%2Ffiledetails%2F%3Fid%3D1779931748&psig=AOvVaw0ELHQO3iVoOS3hnQrxBAlv&ust=1601188068117000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCJDhjsOYhuwCFQAAAAAdAAAAABAk)

> 本系列是用作备忘的，毕竟有些开发小工具的设置方式我可能反复找了不下十遍了，不如自己记录一下。

## PyQt5和Qt Designer安装

先把PyQt5和Qt Designer安装了才行。

**PyQt5**

```
pip install PyQt5
```
**Qt Designer**

[这里](https://build-system.fman.io/qt-designer-download) 下载


## PyCharm中添加Qt Designer工具
- 在“File—>Settings—>Tools—>External Tools”中点击“+”号，添加外部工具；
- Program中填入“designer.exe”的路径， 
eg. `D:\ProgramData\Anaconda2\Library\bin\designer.exe`；
- Working directory中填入`$FileDir$`。

## PyCharm中添加Pyuic工具
- 在“File—>Settings—>Tools—>External Tools”中点击“+”号，添加外部工具；
- Program中填入“python.exe”的路径， 
eg. `D:\ProgramData\Anaconda2\python.exe`
- Arguments中填入`-m PyQt5.uic.pyuic 
\$FileName$ -o \$FileNameWithoutExtension$.py`；
- Working directory中填入`$ProjectFileDir$`。

![Image](https://pic4.zhimg.com/80/v2-a9ddc3f261c73ece6e3df617db9d6448.jpg)


## .ui 文件转 .py
对于从 Qt Designer 生成的 .ui 窗体文件，直接在pycharm中右键

![Image](https://pic4.zhimg.com/80/v2-446b10bb8daae6faef84059441f2aa80.jpg)

点击pyuic即可生成相应的窗体.py文件 Ui_Windows 类。

接下来就可以愉快地用 python 开发软件啦！