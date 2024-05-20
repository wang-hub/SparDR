import itertools

import numpy as np
import scipy.sparse as sp


import tvm
from tvm.ir import IRModule
from tvm import relay
from tvm.topi.sparse.utils import random_bsr_matrix
from tvm.relay.build_module import bind_params_by_name

#可视化relay
from typing import (
    Dict,
    Union,
    Tuple,
    List,
)
import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)


def run_func(func, params, x):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, new_params = relay.build(func, "llvm", params=params)

    from tvm.contrib import graph_executor

    dev = tvm.cpu(0)
    dtype = "float32"
    m = graph_executor.create(graph, lib, dev)
    # set inputs
    m.set_input("data", tvm.nd.array(x.astype(dtype)))
    m.set_input(**new_params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    return tvm_output.numpy()


def test_bsr_sparse_conv2d_nchw():
    data = relay.var("data", shape=(1, 64, 32, 32), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(128, 64, 1, 1), dtype="float32")
    y = relay.nn.conv2d(x, w, channels=128, kernel_size=1, data_layout="NCHW", kernel_layout="OIHW")
    # print("dir:",dir(y))
    print("type:",type(y))
    # print("conv->data:",y2.args[0])
    # print("conv->data.type:",type(y2.args[0]))
    # print("conv->data.dir:",dir(y2.args[0]))
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)
    #转换前
    main_func = func
    main_gvar = relay.GlobalVar("main")
    mod = tvm.IRModule({main_gvar: main_func,})
    #推理
    mod = relay.transform.InferType()(mod)
    # print("conv->trans->data:",y2.args[0])
    func = mod['main']
    viz = relay_viz.RelayVisualizer(mod)
    viz.render()

    params = {
        "weight": tvm.nd.array(
            np.array(random_bsr_matrix(128, 64, 8, 1, 0.1, "float32").todense()).reshape(
                128, 64, 1, 1
            )
        )
    }

    x_np = np.random.randn(1, 64, 32, 32).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    # sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert(
    #     func, params, (8, 1), 0.2, "NCHW"
    # )
    sparse_func = relay.data_dep_optimization.utils._run_opt_pass(
        mod['main'],
        relay.transform._ffi_api.Conv2dToSparseMEC(0.1,'NCHW',1)
    )
    #转换后
    main_func = sparse_func
    mod = tvm.IRModule({main_gvar: main_func,})
    viz = relay_viz.RelayVisualizer(mod)
    viz.render()
    
    print(mod["main"].ret_type)
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)


# test_bsr_sparse_conv2d_nchw()

kernel_size = 3
in_channel = 256
inSize = 58
padding = 0
outSize = inSize - kernel_size + 1
row = kernel_size * in_channel
col = outSize*(inSize+2*padding)
ki = in_channel * kernel_size

output = np.zeros((row,col),dtype='float32') 
# def trans(dense_data, kernel_size, padding):
#直接转换
def trans():
    # out_size = tvm.te.indexdiv(in_size - kernel_size + 2 * padding, 1) + 1
    for i in range(row):
        c = i % in_channel
        off_size = i % ki //in_channel
        for j in range(col):
            h = j // outSize
            w = j % outSize + off_size
            output[i][j] = dense_data[c][h][w]
    return output

# def trans_adv(dense_data, kernel_size, padding):
#按照原始特征图循环来重排
output_adv = np.zeros((row,col),dtype='float32') 
def trans_adv():
    # out_size = tvm.te.indexdiv(in_size - kernel_size + 2 * padding, 1) + 1
    row_s = 0
    col_s = 0 
    for w in range(kernel_size):
        for c in range(in_channel):
            col_s = 0
            for h in range(inSize):
                for ww in range(outSize):
                    output_adv[row_s][col_s] = dense_data[c][h][w+ww]
                    col_s = col_s + 1
            row_s = row_s + 1
    return output

#使用多线程转换
output_mu = np.zeros((row,col),dtype='float32') 
def trans_adv_process(id):
    # out_size = tvm.te.indexdiv(in_size - kernel_size + 2 * padding, 1) + 1
    # col_s = 0 
    # c = id // kernel_size
    # w = id % kernel_size
    # with lock:
    #     for h in range(inSize):
    #         for ww in range(outSize):
    #             # output[id][col_s] = 0
    #             output_mu[id,col_s] = dense_data[c][h][w+ww]
    #             col_s = col_s + 1
    # channel_s = 0
    # col_s = 0
    # idx = 0
    # row = kernel_size * in_channel
    # col = outSize*(inSize+2*padding)

    for col_s in range(kernel_size):
        # for channel_s in range(in_channel):
            for i in range(inSize):
                for j in range(outSize):
                    output_avoid[idx] = dense_data[channel_s,i,col_s+j]
                    idx = idx + 1
    output_avoid.reshape((row,col))
    return


#直接索引++
output_avoid = np.zeros((row*col),dtype='float32')
def trans_avoidCompute():
    idx = 0
    # row = kernel_size * in_channel
    # col = outSize*(inSize+2*padding)
    for col_s in range(kernel_size):
        for channel_s in range(in_channel):
            for i in range(inSize):
                for j in range(outSize):
                    output_avoid[idx] = dense_data[channel_s,i,col_s+j]
                    idx = idx + 1
    output_avoid.reshape((row,col))
    return


import threading
import timeit
dense_data = np.random.randn(in_channel, inSize, inSize).astype("float32")
# arrMU = np.zeros((kernel_size*in_channel, outSize*inSize),dtype='float32')
lock = threading.Lock()  # 创建一个锁以确保同步
def mul():
    # 创建多个进程
    num_threads = kernel_size * in_channel
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=trans_adv_process, args=(i,))
        threads.append(thread)
        thread.start()
    # 等待所有进程完成
    for thread in threads:
        thread.join()
    # return arrMU

output_test = np.zeros((row, col),dtype='float32')
def test():
    for i in range(row):
        # c = i % in_channel
        # off_size = i % ki //in_channel
        for j in range(col):
            # h = j // outSize
            # w = j % outSize + off_size
            output_test[i,j] = 0.0


testNum = 10
timer = timeit.timeit(lambda: test(), number=testNum)
print(f"test Average execution time: {timer / testNum:.5f} seconds")

# trans(x_np,3,0)
timer = timeit.timeit(lambda: trans(), number=testNum)
print(f"trans Average execution time: {timer / testNum:.5f} seconds")
timer = timeit.timeit(lambda: trans_adv(), number=testNum)
print(f"trans_adv Average execution time: {timer / testNum:.5f} seconds")
timer = timeit.timeit(lambda: trans_avoidCompute(), number=testNum)
print(f"trans_avoid Average execution time: {timer / testNum:.5f} seconds")
timer = timeit.timeit(lambda: mul(), number=testNum)
print(f"trans_mul Average execution time: {timer / testNum:.5f} seconds")

# arr1 = trans()
# arr2 = trans_adv()
# print(np.array_equiv(arr1,arr2))
# print(np.array_equiv(arr1,arrMU))
# # element_wise_equal = arr1 == arrMU
# print(arrMU)
# print(element_wise_equal)
