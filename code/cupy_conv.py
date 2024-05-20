import cupy as cp
import cupyx.scipy.sparse as sparse
import time
import numpy as np
from tvm.topi.sparse.utils import random_bsr_matrix
import os

def im2col_NCHW(input_data, kernel_size, stride, padding):
    # 获取输入数据的维度信息
    batch_size, num_channels, input_height, input_width = input_data.shape
    # 获取卷积核的尺寸信息
    kernel_height, kernel_width, _, _ = kernel_size

    # 计算输出图像的尺寸
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    # 对输入数据进行填充
    input_padded = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # 初始化图像块矩阵
    cols = np.zeros((batch_size, num_channels * kernel_height * kernel_width, output_height, output_width))

    # 循环遍历输出图像的每一个像素位置
    for y in range(output_height):
        for x in range(output_width):
            # 计算当前位置在原始图像上的位置
            y_start, y_end = y * stride, y * stride + kernel_height
            x_start, x_end = x * stride, x * stride + kernel_width

            # 提取当前位置的图像块，并将其展平为列向量
            cols[:, :, y, x] = input_padded[:, :, y_start:y_end, x_start:x_end].reshape(batch_size, -1)

    return cols

# 定义 im2col 函数
def im2col(input_data, kernel_size, stride, padding):
    n, h, w, c = input_data.shape
    kh, kw, _, _ = kernel_size
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1

    input_padded = cp.pad(input_data, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    cols = cp.zeros((n, oh, ow, kh * kw * c), dtype=input_data.dtype)
    
    for y in range(oh):
        for x in range(ow):
            y_start, y_end = y * stride, y * stride + kh
            x_start, x_end = x * stride, x * stride + kw
            cols[:, y, x, :] = input_padded[:, y_start:y_end, x_start:x_end, :].reshape(n, -1)
    
    return cols

# 将卷积核展平为二维矩阵形式
def flatten_kernel(kernel):
    # return kernel.reshape(-1, kernel.shape[0])
    return kernel.reshape(kernel.shape[0], -1)

# 生成随机图像数据和稀疏卷积核
def test_cupy(input_file=None,
         kernel_file=None,
         padding=(1,1),
         stride=(1,1)):
    
    in_file = os.path.dirname(input_file)
    
    image = np.load(input_file)
    batch_size, num_channels, image_height, image_width = image.shape
    print(image.shape)

    kernel = np.load(kernel_file)
    num_filters,_,KH,KW = kernel.shape
    kernel_size = KH
    print(kernel.shape)
    
    stride = stride[0]
    padding = padding[0]

    # image_shape = (batch_size, image_height, image_width, num_channels)  # 图像大小为 256x256，3通道
    # 随机生成图像和稀疏卷积核
    # image = cp.random.rand(*image_shape).astype(cp.float32)
    # sparse_kernel = cp.random.randint(2, size=(kernel_size, kernel_size, num_channels, num_filters)).astype(cp.float32)
    # kernel = cp.array(random_bsr_matrix(num_filters, num_channels*kernel_size*kernel_size, 1, 1, 0.1, "float32")
                    # .todense()).reshape(num_filters, kernel_size, kernel_size, num_channels)
    # csr = utils.deal_sp_kernel(kernel)

    # 转换图像为 im2col 格式
    # cols = im2col(image, (kernel_size, kernel_size, num_channels, num_filters), stride, padding)
    cols = im2col_NCHW(image, (kernel_size, kernel_size, num_channels, num_filters), stride, padding)

    # 将 im2col 格式的数据转换为二维数组
    # cols_reshaped = cp.reshape(cols, (-1, num_channels))
    cols_reshaped = cp.reshape(cols, (cols.shape[1],-1))
    # cols_reshaped = cp.reshape(cols, (-1, batch_size * cols.shape[2] * cols.shape[3]))
    # print(cols_reshaped.shape)

    # 展平卷积核
    sparse_kernel = flatten_kernel(kernel)
    # print(sparse_kernel.shape)
    
    cols_reshaped = cp.array(cols_reshaped)
    sparse_kernel = cp.array(sparse_kernel)
    
    # 转换稀疏卷积核为稀疏矩阵
    ress = cp.nonzero(sparse_kernel)
    # print(ress)
    rows = ress[0]
    cols = ress[1]
    # print(rows.shape)
    # print(cols.shape)
    values = sparse_kernel[rows, cols]
    print('sparity:',values.shape[0]/(num_filters*num_channels*kernel_size*kernel_size))
    # print(values.shape)
    # input()
    # rows, cols, values = cp.nonzero(sparse_kernel)
    sparse_matrix_kernel = sparse.csr_matrix((values, (rows, cols)), shape=sparse_kernel.shape)

    # 执行稀疏矩阵和密集矩阵的乘法
    num_iterations = 100
    total_time = 0.0

    print("sparse:",sparse_matrix_kernel.shape)
    print("data:",sparse_matrix_kernel.data)
    print("dense:",cols_reshaped.shape)
    for _ in range(num_iterations):
        # start_time = time.time()
        start_time = time.perf_counter()
        
        # 执行稀疏矩阵和密集矩阵的乘法
        result = sparse_matrix_kernel @ cols_reshaped
        # result = cupyx.CuSparse   sparse_matrix_kernel @ cols_reshaped
        
        cp.cuda.stream.get_current_stream().synchronize()
        # end_time = time.time()
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        total_time += iteration_time

    # 计算平均时间
    average_time = total_time / num_iterations *1e3
    print("Average convolution time using sparse matrix and dense matrix (im2col+GEMM): {:.6f} ms".format(average_time))
    
    return average_time

def test(
    M = 10000,
    N = 10000,
    K = 10000):
    A = cp.array(random_bsr_matrix(M, K, 1, 1, 0.1, "float32")
                    .todense()).reshape(M, K)
    B = cp.random.rand(K, N).astype(cp.float32)
    # 转换稀疏卷积核为稀疏矩阵
    ress = cp.nonzero(A)
    rows = ress[0]
    cols = ress[1]
    values = A[rows, cols]
    print('sparity:',values.shape[0]/(M*K))
    # print(values.shape)
    # input()
    # rows, cols, values = cp.nonzero(sparse_kernel)
    sparse_matrix = sparse.csr_matrix((values, (rows, cols)), shape=A.shape)

    # 执行稀疏矩阵和密集矩阵的乘法
    num_iterations = 100
    total_time = 0.0

    # print("sparse:",sparse_matrix_kernel.shape)
    # print("data:",sparse_matrix_kernel.data)
    # print("dense:",cols_reshaped.shape)
    for _ in range(num_iterations):
        # start_time = time.time()
        start_time = time.perf_counter()
        # 执行稀疏矩阵和密集矩阵的乘法
        # result = sparse_matrix @ B
        result = sparse_matrix @ sparse_matrix
        cp.cuda.stream.get_current_stream().synchronize()
        # end_time = time.time()
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        total_time += iteration_time

    # 计算平均时间
    average_time = total_time / num_iterations *1e3
    print("Average convolution time using sparse matrix and dense matrix (im2col+GEMM): {:.6f} ms".format(average_time))

    A = np.array(random_bsr_matrix(M, K, 1, 1, 0.1, "float32")
                    .todense()).reshape(M, K)
    B = np.random.rand(K, N).astype(cp.float32)
    # 转换稀疏卷积核为稀疏矩阵
    ress = np.nonzero(A)
    rows = ress[0]
    cols = ress[1]
    values = A[rows, cols]
    print('sparity:',values.shape[0]/(M*K))
    import scipy
    sparse_matrix = scipy.sparse.csr_matrix((values, (rows, cols)), shape=A.shape)

    # 执行稀疏矩阵和密集矩阵的乘法
    num_iterations = 100
    total_time = 0.0

    # print("sparse:",sparse_matrix_kernel.shape)
    # print("data:",sparse_matrix_kernel.data)
    # print("dense:",cols_reshaped.shape)
    for _ in range(num_iterations):
        # start_time = time.time()
        start_time = time.perf_counter()
        # 执行稀疏矩阵和密集矩阵的乘法
        # result = sparse_matrix @ B
        result = sparse_matrix @ sparse_matrix
        # cp.cuda.stream.get_current_stream().synchronize()
        # end_time = time.time()
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        total_time += iteration_time

    # 计算平均时间
    average_time = total_time / num_iterations *1e3
    print("Average convolution time using sparse matrix and dense matrix (im2col+GEMM)_cpu: {:.6f} ms".format(average_time))

# test()





# import numpy as np
# import cupy as cp
# import cupyx.scipy.sparse as sparse
# import cupyx.scipy.sparse.linalg as splinalg
# import time

# # 定义 im2col 函数
# def im2col(input_data, kernel_size, stride, padding):
#     n, h, w, c = input_data.shape
#     kh, kw, _, _ = kernel_size
#     oh = (h + 2 * padding - kh) // stride + 1
#     ow = (w + 2 * padding - kw) // stride + 1

#     input_padded = cp.pad(input_data, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
#     cols = cp.zeros((n, oh, ow, kh * kw * c), dtype=input_data.dtype)
    
#     for y in range(oh):
#         for x in range(ow):
#             y_start, y_end = y * stride, y * stride + kh
#             x_start, x_end = x * stride, x * stride + kw
#             cols[:, y, x, :] = input_padded[:, y_start:y_end, x_start:x_end, :].reshape(n, -1)
    
#     return cols

# # 生成随机图像数据和卷积核
# batch_size = 1
# image_height = 256
# image_width = 256
# num_channels = 3
# image_shape = (batch_size, image_height, image_width, num_channels)  # 图像大小为 256x256，3通道
# kernel_size = 3
# num_filters = 64
# stride = 1
# padding = 1

# # 随机生成图像和卷积核
# image = np.random.rand(*image_shape).astype(np.float32)
# kernel = np.random.rand(kernel_size, kernel_size, num_channels, num_filters).astype(np.float32)

# # 将图像和卷积核移动到 GPU
# image_gpu = cp.array(image)
# kernel_gpu = cp.array(kernel)

# # 转换图像为 im2col 格式
# cols = im2col(image_gpu, (kernel_size, kernel_size, num_channels, num_filters), stride, padding)

# # 将 im2col 格式的数据转换为二维数组
# cols_reshaped = cp.reshape(cols, (batch_size * cols.shape[1] * cols.shape[2], -1))
# print(cols_reshaped.shape)
# # input()
# idx = 0
# for i in range(cols_reshaped.shape[0]):
#     for j in range(cols_reshaped.shape[1]):
#         idx+=1
#         if(idx%9 != 0):
#             cols_reshaped[i][j] = 0
# print(cols_reshaped[0][0])
# # 构建稀疏矩阵
# # nonzero_indices = cp.nonzero(cp.any(cols_reshaped != 0, axis=1))  # 非零元素的行和列索引
# nonzero_indices = cp.nonzero(cols_reshaped)  # 非零元素的行和列索引
# print(nonzero_indices)
# input()
# rows = nonzero_indices[0]  # 非零元素的行索引
# cols = nonzero_indices[1]  # 非零元素的列索引
# values = cp.ones((cols.size,))  # 非零元素的值

# # 创建 CSR 格式稀疏矩阵
# sparse_matrix_csr = sparse.csr_matrix((values, (rows, cols)), shape=(batch_size * image_height * image_width, num_filters))

# # 将稀疏矩阵移动到 GPU
# sparse_matrix_csr_gpu = sparse_matrix_csr.tocoo().astype(cp.float32).tocsr()

# # 构建右侧矩阵
# rhs_matrix = cp.ones(num_filters, dtype=np.float32)

# # 执行稀疏矩阵乘法
# num_iterations = 100
# total_time = 0.0

# for _ in range(num_iterations):
#     start_time = time.time()
    
#     # 调用 CuPy 的稀疏矩阵乘法函数
#     result = splinalg.spsolve(sparse_matrix_csr_gpu, rhs_matrix)
    
#     end_time = time.time()
#     iteration_time = end_time - start_time
#     total_time += iteration_time

# # 计算平均时间
# average_time = total_time / num_iterations
# print("Average convolution time on GPU using im2col+GEMM+CuSparse (CuPyX): {:.6f} seconds".format(average_time))
