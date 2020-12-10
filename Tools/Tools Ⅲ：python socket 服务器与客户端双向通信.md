#! https://zhuanlan.zhihu.com/p/263630359
![Image](https://pic4.zhimg.com/80/v2-ba8349bf6c1a51498ab00ba6d7838ac5.jpg)
# Tools 3：Python socket 服务器与客户端双向通信（服务器NAT，文件传输）

## Task
1. Client -> Server 发送诊断图像标题 + 图像数据
2. Server 接收诊断图像标题 + 图像数据并存储
3. Server -> Client 处理诊断图像并发送诊断结果标题 + 结果数据
4. Client 接收诊断结果标题 + 结果数据并存储

## Key
1. 服务器端口要在与其连接的路由器中设置端口映射，例：6666（服务器自身端口）-> 30666 (外网端口)
2. 服务器外网ip查询：
   ```
   curl cip.cc
   curl ifconfig.me
   curl ifconfig.me/all
   curl www.pubyun.com/dyndns/getip
   curl members.3322.org/dyndns/getip
   ```

## Server 端
```python
###服务器端server.py
import socket
import os
import sys
import struct


def socket_service_image(ui=None):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # s.bind(('127.0.0.1', 6666))
        s.bind(('192.xxx.xxx.xxx', 6666))
        # Client 的ip和端口
        s.listen(10)
    except socket.error as msg:
        print(msg)
        sys.exit(1)

    print("Wait for Connection.....................")

    while True:
        sock, addr = s.accept()  # addr是一个元组(ip,port)
        deal_image(sock, addr, ui)

def deal_image(sock, addr, ui=None):
    print("Accept connection from {0}".format(addr))  # 查看发送端的ip和端口

    while True:
        fileinfo_size = struct.calcsize('128sq')
        buf = sock.recv(fileinfo_size)  # 接收图片名
        if buf:
            filename, filesize = struct.unpack('128sq', buf)
            fn = filename.decode().strip('\x00')
            new_filename = os.path.join('/home/xxx/xxxx/xxx/Server/', 'new_' + fn)  # 在服务器端新建图片名（可以不用新建的，直接用原来的也行，只要客户端和服务器不是同一个系统或接收到的图片和原图片不在一个文件夹下）

            recvd_size = 0
            fp = open(new_filename, 'wb')

            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = sock.recv(1024)
                    recvd_size += len(data)
                else:
                    data = sock.recv(1024)
                    recvd_size = filesize
                fp.write(data)  # 写入图片数据
            print('{} saved.'.format(new_filename))
            fp.close()

            # -------- 处理图像，并发送结果到client -------
            '''result_path = imageprocess(data)'''

            result_path = '/home/xxx/xxxx/xxx/xxx.pdf'
            # 
            fhead = struct.pack(b'128sq', bytes(os.path.basename(result_path), encoding='utf-8'),
                                os.stat(result_path).st_size)  # 将xxx.jpg以128sq的格式打包
            sock.send(fhead)

            fp = open(result_path, 'rb')  # 打开要传输的结果文件
            while True:
                result_data = fp.read(1024)  # 读入结果文件数据
                if not result_data:
                    print('{0} send over...'.format(result_path))
                    break
                sock.send(result_data)  # 以二进制格式发送结果文件数据
        sock.close()
        break


if __name__ == '__main__':
    # socket_service_image(ui)  # 因为我是把这个函数加载到 pyqt 的 UI 中，所以会使用主ui窗口的图像处理程序。
    socket_service_image()
```

## Client 端
```python
'''
Fuction：客户端发送图片和数据
Date：2018.9.8
Author：snowking
'''
###客户端client.py
import socket
import os
import sys
import struct


def sock_client_image():
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('202.xxx.xxx.xxx', 30666))  # 服务器和客户端在不同的系统或不同的主机下时使用的ip和端口，首先要查看服务器所在的系统网卡的ip
            # s.connect(('127.0.0.1', 6666))  #服务器和客户端都在一个系统下时使用的ip和端口
        except socket.error as msg:
            print(msg)
            print(sys.exit(1))
        filepath = input('input the file: ')  # 输入当前目录下的图片名 xxx.jpg
        fhead = struct.pack(b'128sq', bytes(os.path.basename(filepath), encoding='utf-8'),
                            os.stat(filepath).st_size)  # 将xxx.jpg以128sq的格式打包
        s.send(fhead)

        fp = open(filepath, 'rb')  # 打开要传输的图片
        while True:
            data = fp.read(1024)  # 读入图片数据
            if not data:
                print('{0} send over...'.format(filepath))
                break
            s.send(data)  # 以二进制格式发送图片数据

        # -------- 接收server端发送的结果文件数据 --------
        while True:
            fileinfo_size = struct.calcsize('128sq')
            buf = s.recv(fileinfo_size)  # 接收图片名
            if buf:
                filename, filesize = struct.unpack('128sq', buf)
                fn = filename.decode().strip('\x00')
                new_filename = os.path.join('D:\\xxx\\xxx\\Client',
                                            'new_' + fn)

                recvd_size = 0
                fp = open(new_filename, 'wb')

                while not recvd_size == filesize:
                    if filesize - recvd_size > 1024:
                        data = s.recv(1024)
                        recvd_size += len(data)
                    else:
                        data = s.recv(1024)
                        recvd_size = filesize
                    fp.write(data)  # 写入结果文件数据
                print('{} saved.'.format(new_filename))
                fp.close()
                break

        s.close()
        # break    #循环发送


if __name__ == '__main__':
    sock_client_image()

```

## Reference
[Python，用简单代码上传内存中的图片到远程服务器进行处理 - 曾伊言](https://zhuanlan.zhihu.com/p/64534116 'card')\
[python socket编程（传输字符、文件、图片）](https://blog.csdn.net/luckytanggu/article/details/53491892 'card')


