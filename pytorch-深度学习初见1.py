#pytorch使用GPU加速前后运行时间对比
import torch
import time
print(torch.__version__)
print(torch.cuda.is_available())  #查看显卡加速是否可用
print("hello world！")
a=torch.rand(10000,1000)
b=torch.rand(1000,2000)

t0=time.time()
c=torch.matmul(a,b)
t1=time.time()
print(a.device,t1-t0,c.norm(2))  #输出使用的平台名称和整体的计算运行时间

device=torch.device("cuda")
t0=time.time()
c=torch.matmul(a,b)
t2=time.time()
print(a.device,t2-t0,c.norm(2))

t0=time.time()
c=torch.matmul(a,b)
t2=time.time()
print(a.device,t2-t0,c.norm(2))
print(c)

#pytorch自动求导函数调用
import torch
from torch import autograd

x=torch.tensor(1.)
a=torch.tensor(1.,requires_grad=True)
b=torch.tensor(2.,requires_grad=True)
c=torch.tensor(3.,requires_grad=True)

y=a**2*x+b*x+c
print("before:",a.grad,b.grad,c.grad)
grads=autograd.grad(y,[a,b,c])
print("after:",grads[0],grads[1],grads[2])



