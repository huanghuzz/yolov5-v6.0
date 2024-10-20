#import torch

#print(torch.cuda.is_available())


import torch
print(torch.cuda.is_available())  # 应返回True，如果CUDA可用的话
print(torch.cuda.device_count())  # 显示可用的CUDA设备数量

print(torch.cuda.device_count())
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
import torch
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.cuda_version)
print(torch.backends.cudnn.version())

import torch
print(torch.__version__) #查阅pytorch版本

import sys
print(sys.version) #查询python版本