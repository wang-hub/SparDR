import torch
import time
import os
import numpy as np

# 定义输入数据和卷积核

def test_torch(input_file=None,
         kernel_file=None,
         padding=(1,1),
         stride=(1,1)):
    
        # batch_size = 1,
        # input_channels = 512,
        # input_height = 14,
        # input_width = 14,
        # kernel_size = 3,
        # output_channels = 512:
    # 设置设备为GPU
    # print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input_data = torch.randn(batch_size, input_channels, input_height, input_width).to(device)
    # conv_weight = torch.randn(output_channels, input_channels, kernel_size, kernel_size).to(device)

    in_file = os.path.dirname(input_file)
    
    input_data = np.load(input_file)
    batch_size, input_channels, input_height, input_width = input_data.shape

    conv_weight = np.load(kernel_file)
    output_channels,_,KH,KW = conv_weight.shape
    kernel_size = KH
    
    stride = stride[0]
    padding = padding[0]

    input_tensor = torch.from_numpy(input_data).to(device)

    # 定义单层卷积
    conv_layer = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size).to(device)

    conv_layer.eval()
    # 测试单层卷积的时间
    num_iterations = 100
    total_time = 0.0

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            output = conv_layer(input_tensor)
            torch.cuda.synchronize()  # 等待GPU计算完成
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time

    average_time = total_time / num_iterations *1e3
    print("Average convolution time on GPU-Torch: {:.6f} ms".format(average_time))
    
    return average_time
# test()
