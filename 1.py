'''检查环境是否可以调用cuda和一些安装版本查询是否正常'''


import torch

print('显示可用的CUDA设备数量:',torch.cuda.device_count())  # 显示可用的CUDA设备数量
print("CUDA available（如果CUDA可用的话,应返回True:）:", torch.cuda.is_available())
print('检查是否能调用cudnn，Ture表示可以调用：',torch.backends.cudnn.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print('cuda版本：',torch.cuda_version)
print('cudnn版本:',torch.backends.cudnn.version())
print('pytorch版本:',torch.__version__) #查阅pytorch版本
import sys
print('python版本:',sys.version) #查询python版本