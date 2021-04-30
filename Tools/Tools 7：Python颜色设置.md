#! https://zhuanlan.zhihu.com/p/368668837
# Tools 7：Python字体、背景、绘图颜色设置以及强迫症中文对齐
> 工欲善其事，必先利其器

本文是为了备忘，整理一下常用的python颜色显示小工具。

## Python字体、背景颜色
字体、背景颜色**转发自：https://blog.csdn.net/ever_peng/article/details/91492491**

试了上面链接之后发现，输出的颜色背景不对齐。作为一个强迫症，这是不被允许的。试过了`%.10s`和`%-10s`以及`str({}).ljust(10)`都一样，最后发现问题出在中文字符上，补齐时一个中文算一个字符，打印出来却成了两个字符。所以，我又找到了下面的专治中文对齐的妙招：

https://blog.csdn.net/weixin_42280517/article/details/80814677

顺便，把代码改成了方便复制的形式。

```python
print("\033[1;30m{}\033[0m".format(' 字体颜色：白色'))
print("\033[1;31m{}\033[0m".format(' 字体颜色：红色'))
print("\033[1;32m{}\033[0m".format(' 字体颜色：深黄色'))
print("\033[1;33m{}\033[0m".format(' 字体颜色：浅黄色'))
print("\033[1;34m{}\033[0m".format(' 字体颜色：蓝色'))
print("\033[1;35m{}\033[0m".format(' 字体颜色：淡紫色'))
print("\033[1;36m{}\033[0m".format(' 字体颜色：青色'))
print("\033[1;37m{}\033[0m".format(' 字体颜色：灰色'))
print("\033[1;38m{}\033[0m".format(' 字体颜色：浅灰色'))
print('\n')
print('{0:{1}<9} \033[1;40m{2}\033[0m\n'.format('背景颜色：白色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;41m{2}\033[0m\n'.format('背景颜色：红色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;42m{2}\033[0m\n'.format('背景颜色：深黄色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;43m{2}\033[0m\n'.format('背景颜色：浅黄色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;44m{2}\033[0m\n'.format('背景颜色：蓝色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;45m{2}\033[0m\n'.format('背景颜色：淡紫色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;46m{2}\033[0m\n'.format('背景颜色：青色', chr(12288), '    '), end='')
print('{0:{1}<9} \033[1;47m{2}\033[0m\n'.format('背景颜色：灰色', chr(12288), '    '), end='')
```

运行结果：

![](https://pic4.zhimg.com/80/v2-84523348c41ffa8d32fea942e4dc1653.png)


## Matplotlib颜色表

**转发自：https://finthon.com/matplotlib-color-list/**

![](https://pic4.zhimg.com/80/v2-6e8318e268b833a9f7564c76cc536b13.png)