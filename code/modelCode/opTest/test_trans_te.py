##测试sparseMEC算子的te转tir
from functools import partial, reduce

import numpy as np
import tvm
from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
from tvm.topi.nn.pad import pad
from tvm.topi.transform import reshape 
from tvm.topi.nn.utils import get_pad_tuple
from tvm import auto_scheduler
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
from tvm import te, auto_scheduler, runtime
from tvm.topi.sparse.utils import random_bsr_matrix
from tvm import IRModule
from collections import namedtuple
import sys
sys.path.insert(0,sys.path[0]+'/../..')
import utils
import os


#重排不加pad
@auto_scheduler.register_workload
def featureTrans(data,kernel_size,padding,stride):
    N,C,H,W = get_const_tuple(data.shape)
    kernel_size = get_const_int(kernel_size)
    stride = get_const_int(stride)
    out_size = (H - kernel_size) // stride + 1
    
    @partial(te.compute, (kernel_size,C,H,out_size), name="trans_Y")
    def fea(col_axis,channel_s,H_s,out_s):
        return data[0,channel_s,H_s,col_axis+out_s]

    # Y= te.compute(
    #     (kernel_size,C,H,out_size),
    #     lambda col_axis,channel_s,H_s,out_s:
    #        data[0,channel_s,H_s,col_axis+out_s],
    #     name="trans_Y" 
    # )
    # return reshape(Y,[kernel_size*C,out_size*H])
    return te.compute(
        (kernel_size*C,out_size*H),
        lambda row,col:
            fea[row // (kernel_size*C),row % (kernel_size*C),col//(H*out_size),col%(H*out_size)],
        name='trans_YY'
    )

@auto_scheduler.register_workload
def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,kernel_size,stride,padding,dilation,out_dtype = None):
    if out_dtype is None:
        out_dtype = data.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    batch, in_channel, in_height, in_width = data.shape
    # compute the output shape
    dilated_kernel_h = (kernel_size - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_size - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    # out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp")
    tranData = featureTrans(temp,kernel_size,stride)
    out_channel = get_const_int(weight_indptr.shape[0]) - 1
    # (n-k+2*p)//s+1
    oshape = (1,out_channel,out_height, out_width)
    def f(n,row,h,w):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        #这里可能是影响速度的地方
        #可以预处理
        d_row = weight_indices_row[elem]
        d_col = weight_indices_col[elem]
        weight_val = tranData[d_row, d_col + h*out_height + w]
        return te.sum(a_val * weight_val, axis=elem_idx,)
    
    return te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')


N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
kernel_size = KH
sparity = 0.8
x = np.random.randn(N,CI, H, W).astype("float32")
kernel = np.array(random_bsr_matrix(CO, CI*kernel_size*kernel_size, 1, 1, 0.1, "float32")
                  .todense()).reshape(CO, CI, kernel_size, kernel_size)

out_size = utils.conv_out_size(H,kernel_size,0,1)

target = tvm.target.Target("llvm")
dev = tvm.cpu()
#  (kernel_size,C,H,out_size)
row = kernel_size*CI
col = H*out_size

def trans_avoidCompute():
    output_avoid = np.zeros((row*col),dtype='float32')
    idx = 0
    # row = kernel_size * in_channel
    # col = outSize*(inSize+2*padding)
    for col_s in range(kernel_size):
        for channel_s in range(CI):
            for i in range(H):
                for j in range(out_size):
                    output_avoid[idx] = x[0,channel_s,i,col_s+j]
                    idx = idx + 1
    out = output_avoid.reshape((row,col))
    return out
# print(trans_avoidCompute().shape)

# out = tvm.nd.empty((kernel_size,CI,H,out_size), device=dev)
out = tvm.nd.empty((kernel_size*CI,out_size*H), device=dev)

#预处理
print('data:',x.shape)
print('输出尺寸：',out.shape)
x_tvm,y = (tvm.nd.array(i,device=dev) for i in(x,out))
data_shape = x.shape

@auto_scheduler.register_workload
def fea_tr(N,C,H,W,kernel_size,padding,stride,dilation=1,out_dtype=None):
    data = te.placeholder((N,C,H,W),dtype='float32',name='feature')
    if out_dtype is None:
        out_dtype = data.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride = stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
        stride = stride_h
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    batch, in_channel, in_height, in_width = data.shape
    # compute the output shape
    dilated_kernel_h = (kernel_size - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_size - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    # out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp")
    # tranData = featureTrans(temp,kernel_size,stride)
    trans = featureTrans(temp, kernel_size, padding, stride)
    return [data,trans]

task = auto_scheduler.SearchTask(
    func=fea_tr, args=(N,CI,H,W,kernel_size,0,1), target=target
)

log_file = "./test_trans_te/test_trans_te_224-224-179-179-3-1-0.log"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=100,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# 运行自动调优（搜索）
task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
# 终止测试过程
del measure_ctx

print(tvm.lower(sch,args))

func = tvm.build(sch,args,target)

# 评估执行时间
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(x_tvm, y).results) * 1000)
)



"""
func = te.create_prim_func([input_mec,trans])   
ir_module_main = IRModule({"main": func})
print(type(ir_module_main))
# print(ir_module_main)

#te创建调度
s = te.create_schedule(trans.op)
print("s type:",type(s))

#TE调度
print("trans.op.axis:", type(trans.op.axis), trans.op.axis)
# row_s, h_s, w_s = trans.op.axis
print("conv.op.reduce_axis:", type(trans.op.reduce_axis), trans.op.reduce_axis)
#reorder
# s[trans].reorder(row_s,elem_idx_s,h_s,w_s)
#parallel
# s[trans].parallel(row_s)
#vectorized
# w_o_s,w_i_s = s[conv].split(w_s, factor=28)
# hw = s[conv].fuse(h_s,w_o_s)
# s[conv].reorder(row_s,elem_idx_s,h_s,w_o_s,w_i_s)
# s[conv].vectorize(w_i_s)

#unroll
# s[conv].unroll(w_o_s)
# s[conv].unroll(h_s)

#TEsch结束后需要lower
m = tvm.lower(s, [input_mec,trans], name = 'test_mec')
print(m)
mod = tvm.build(m, target=target)

#评估时延
timer = mod.time_evaluator(mod.entry_name, dev=dev, number=20)
result = (
    timer(x_tvm, y
            ).mean
        * 1e3
    )
print(
    "our Convolution: %f ms" 
    % result
)
"""