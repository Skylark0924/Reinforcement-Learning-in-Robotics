#! https://zhuanlan.zhihu.com/p/273508904
![Image](https://pic4.zhimg.com/80/v2-a60e43faa3dbb62c1ab5db2604be27c0.jpg)
# Tools 5：Python三行转并行后续——多进程全局变量

## 感谢原作者 [houyanhua1 - CSDN](https://blog.csdn.net/houyanhua1/article/details/78244288)

自从上次学会了CPU并行，我就打开了新世界的大门，什么代码都得并行一下，直到我遇到了需要多进程维护同一个字典变量的情况。

当我自信满满地写完了一个处理并保存 `dict` 到 `pickle` 的程序。结果后续的程序告诉我不能使用空的 `pickle` 的时候，我一脸问号？？？我可是加了 `global` 的啊？？？

![图自：黑人问号脸](https://pic4.zhimg.com/80/v2-d3ca87f25b52907c1f69b76dbafc8443.png)

实际上，用 `global` 的方式让多进程共用一个变量，在一个进程中修改后，在另外的进程中并没有产生修改。

那么，就需要 `multiprocessing` 库来正式地定义一个全局变量了：

在定义线程池之前，声明全局变量
```python
num=multiprocessing.Value("d",10.0)  # 共享数值：d表示数值,
num=multiprocessing.Array("i",[1,2,3,4,5])  # 共享数组
mydict=multiprocessing.Manager().dict()  # 共享字典
mylist=multiprocessing.Manager().list(range(5))  # 共享 list
```

完整版如下：
## 进程之间共享数据(数值型)：

```python
import multiprocessing
 
def  func(num):
    num.value=10.78  #子进程改变数值的值，主进程跟着改变
 
if  __name__=="__main__":
    num=multiprocessing.Value("d",10.0) # d表示数值,主进程与子进程共享这个value。（主进程与子进程都是用的同一个value）
    print(num.value)
 
    p=multiprocessing.Process(target=func,args=(num,))
    p.start()
    p.join()
 
    print(num.value)
```
## 进程之间共享数据(数组型)： 
```python
import multiprocessing
 
def  func(num):
    num[2]=9999   #子进程改变数组，主进程跟着改变
 
if  __name__=="__main__":
    num=multiprocessing.Array("i",[1,2,3,4,5])   #主进程与子进程共享这个数组
    print(num[:])
 
    p=multiprocessing.Process(target=func,args=(num,))
    p.start() 
    p.join()
 
    print(num[:])
```
## 进程之间共享数据(dict, list)：
```python
import multiprocessing
 
def func(mydict,mylist):
    mydict["index1"]="aaaaaa"   #子进程改变dict,主进程跟着改变
    mydict["index2"]="bbbbbb"
    mylist.append(11)        #子进程改变List,主进程跟着改变
    mylist.append(22)
    mylist.append(33)
 
if __name__=="__main__":
    with multiprocessing.Manager() as MG:   #重命名
        mydict=multiprocessing.Manager().dict()   #主进程与子进程共享这个字典
        mylist=multiprocessing.Manager().list(range(5))   #主进程与子进程共享这个List
 
        p=multiprocessing.Process(target=func,args=(mydict,mylist))
        p.start()
        p.join()
 
        print(mylist)
        print(mydict)
```

不过，我后来想想，如果是在 `class` 里，应该可以用 `self.dict` 来更改吧。（未验证）

