#! https://zhuanlan.zhihu.com/p/269623666
![Image](https://pic4.zhimg.com/80/v2-ba8349bf6c1a51498ab00ba6d7838ac5.jpg)
# Tools 4：Python三行转并行——真香！

俗话说，**二十核就是二十倍的快乐**。让我们先来感受一下这份加倍的快乐：

![](https://pic4.zhimg.com/80/v2-913300b0c19c12bddbeaab5da0a6bb6f.png)

```python
import time
import multiprocessing


def job(x, y):
	"""
	:param x:
	:param y:
	:return:
	"""
	return x * y

def parallel(z):
	"""
    处理多参数传参问题（实际上把参数写成元组，在job函数内再拆成
     x, y = param 也一样
	:param z:
	:return:
	"""
	return job(z[0], z[1])


if __name__ == "__main__":
	time1=time.time()
	pool = multiprocessing.Pool(2) # 参数缺省的话就是cpu全员上阵
	# 把本来要写成循环的参数, 做成一个list
	data_list=[(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10)]
	# 重点就只有下面这一行！！！
	# 用map函数代替循环
	res = pool.map(parallel, data_list)
	time2=time.time()
	print(res)
	pool.close()
	pool.join()
	print('总耗时：' + str(time2 - time1) + 's')
```


当我最终用上了并行之后，除了感叹真香，还流下了不学无术的眼泪。

## Reference
**感谢以下两位的分享**：
1. [教你用一行Python代码实现并行](https://www.cnblogs.com/wumingxiaoyao/archive/2004/01/13/8241869.html)
2. [【python 多进程传参】pool.map() 函数传多参数](https://blog.csdn.net/u013421629/article/details/100284962)