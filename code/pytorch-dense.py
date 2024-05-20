import torch
from time import time
 
# 定义一个简单的单层卷积层
class SimpleConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
 
    def forward(self, x):
        return self.conv(x)
 
# 实例化一个卷积层
conv_layer = SimpleConv2d(in_channels=128, out_channels=128, kernel_size=3)
 
# 输入一个随机的batch数据
batch_size = 1
input_data = torch.randn(batch_size, 128, 112, 112)  # 假设输入尺寸为224x224
 
# 测试运行时间
start_time = time()
output = conv_layer(input_data)
end_time = time()
elapsed_time = end_time - start_time
 
print(f"Conv2d layer forward pass took: {elapsed_time * 1e3} ms")