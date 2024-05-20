import numpy as np
import time
from scipy.signal import convolve2d
import pyculib.sparse as sparse
import pyculib.blas as blas

# 生成随机输入数据和卷积核
input_shape = (64, 64)
kernel_size = (3, 3)
num_channels = 3
num_kernels = 64

input_data = np.random.randn(*input_shape, num_channels).astype(np.float32)
kernel_data = np.random.randn(*kernel_size, num_channels, num_kernels).astype(np.float32)

# 定义 im2col 函数
def im2col(input_data, kernel_size):
    k_h, k_w = kernel_size
    p_h, p_w = k_h // 2, k_w // 2  # 使用相同填充大小
    padded_data = np.pad(input_data, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), mode='constant')
    batch_size, in_h, in_w, num_channels = padded_data.shape
    col_data = np.zeros((batch_size, in_h * in_w, k_h * k_w * num_channels), dtype=input_data.dtype)
    for i in range(in_h):
        for j in range(in_w):
            col_data[:, i * in_w + j, :] = padded_data[:, i:i + k_h, j:j + k_w, :].reshape(batch_size, -1)
    return col_data

# 定义卷积操作函数
def convolve_im2col(input_data, kernel_data):
    batch_size, input_height, input_width, _ = input_data.shape
    kernel_height, kernel_width, _, num_kernels = kernel_data.shape
    col_data = im2col(input_data, (kernel_height, kernel_width))
    
    # 将 col_data 转换为 CSR 格式
    col_data_flatten = col_data.reshape(batch_size, -1)
    col_data_csr = sparse.dn2csr(col_data_flatten)
    
    # 将 kernel_data 转换为 CSR 格式
    kernel_data_flatten = kernel_data.reshape(-1, num_kernels)
    kernel_data_csr = sparse.dn2csr(kernel_data_flatten.T)
    
    # 执行稀疏矩阵-稀疏向量乘法（SpMV）
    result = sparse.csrgemm('N', 'N', 1.0, col_data_csr, kernel_data_csr, transa='N')
    return result.reshape(batch_size, input_height, input_width, num_kernels)

# 测试卷积操作的时间
num_iterations = 100
total_time = 0.0
for _ in range(num_iterations):
    start_time = time.time()
    
    # 执行卷积操作
    _ = convolve_im2col(input_data, kernel_data)
    
    end_time = time.time()
    iteration_time = end_time - start_time
    total_time += iteration_time

# 计算平均时间
average_time = total_time / num_iterations
print("Average convolution time using IM2COL+GEMM and CuSparse via pyculib: {:.6f} seconds".format(average_time))
