import tvm
from tvm import topi,testing,te
import numpy as np
import os
import utils


def test_conv(N,CI,H,W, w_data_shape, w_indices_shape, w_indptr_shape, dtype,stride,padding):
    X = te.placeholder((N,CI,H,W), dtype=dtype,name='input')
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype,name='w_data')
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32",name='w_indices')
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32",name='w_indptr')
    with tvm.target.Target('llvm'):
        out = topi.x86.sparse.spconv2d_3x3_nchw(X,W_data,W_indices,W_indptr)
        s = topi.x86.sparse.schedule_spconv2d_3x3_nchw(out)
        func = tvm.build(s,[X,W_data,W_indices,W_indptr,out])
    return func
    # return [X, W_data, W_indices, W_indptr, out]


def test(input_file=None,
         kernel_file=None,
         padding=0,
         stride=1):
    
    in_file = os.path.dirname(input_file)
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    x = np.load(input_file)
    N,CI,H,W = x.shape
    kernel = np.load(kernel_file)
    CO,_,kernel_size,_ = kernel.shape
    #只支持kernel大小为1
    # assert kernel_size==1
    if(kernel_size != 3):
        print("kernel_size is not 3, skip sparse_x86_k3")
        return 0
    bsr = utils.deal_sp_kernel_bsr(kernel)
    # args = (N,CI,H,W,bsr.data.shape,bsr.indices.shape,bsr.indptr.shape,'float32',stride,padding)
    func = test_conv(N,CI,H,W,bsr.data.shape,bsr.indices.shape,bsr.indptr.shape,'float32',stride,padding)
    timer = func.time_evaluator(func.entry_name, dev=dev, number=10)
    out_size = utils.conv_out_size(H,kernel_size,padding,stride)
    out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
    x,weight_indptr,weight_indices,weight_data,y = (tvm.nd.array(i,device=dev) for i in(x,bsr.indptr,bsr.indices,bsr.data,out))
    timer = func.time_evaluator(func.entry_name, dev=dev, number=10)
    result = (
        timer(x, weight_data, weight_indices, weight_indptr, y
                ).mean
            * 1e3
        )
    print(
        "sparse_x86_k3 Convolution: %f ms" 
        % result
    )