##测试sparseMEC算子的te转tir
from functools import partial, reduce
import numpy as np
import tvm
from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm import auto_scheduler
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
from tvm import te, auto_scheduler, runtime
from tvm.topi.sparse.utils import random_bsr_matrix
from tvm import IRModule
from tvm.contrib import tedd
from collections import namedtuple
import sys
sys.path.insert(0,sys.path[0]+'/../..')
import utils
import os
import numpy as np

@auto_scheduler.register_workload
def featureTrans(data,kernel_size,stride):
    _,C,H,_ = get_const_tuple(data.shape)
    out_size = (H - kernel_size) // stride + 1
    
    # @partial(te.compute, (kernel_size,C,H,out_size), name="trans_Y")
    # def fea(col_axis,channel_s,H_s,out_s):
    #     return data[0,channel_s,H_s,col_axis+out_s]
    fea = te.compute(
        (kernel_size,C,H,out_size),
        lambda col_axis,channel_s,H_s,out_s:
            data[0,channel_s,H_s,col_axis+out_s],
        name="trans_Y"
    )
    return te.compute(
        (kernel_size*C,out_size*H),
        lambda row,col:
            fea[row // C,row % C,col // out_size,col % out_size],
        name='trans_reshape'
    )

# @auto_scheduler.register_workload
# def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,kernel_size,stride,padding,dilation=1,out_dtype = None):
#     if out_dtype is None:
#         out_dtype = data.dtype
#     assert isinstance(stride, int) or len(stride) == 2
#     assert isinstance(dilation, int) or len(dilation) == 2
#     if isinstance(stride, int):
#         stride = stride_h = stride_w = stride
#     else:
#         stride_h, stride_w = stride
#         stride = stride_h
#     if isinstance(dilation, int):
#         dilation_h = dilation_w = dilation
#     else:
#         dilation_h, dilation_w = dilation
#     batch, in_channel, in_height, in_width = data.shape
#     # compute the output shape
#     dilated_kernel_h = (kernel_size - 1) * dilation_h + 1
#     dilated_kernel_w = (kernel_size - 1) * dilation_w + 1
#     pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
#         padding, (dilated_kernel_h, dilated_kernel_w)
#     )
#     # out_channel = num_filter
#     out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
#     out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
#     # compute graph
#     pad_before = [0, 0, pad_top, pad_left]
#     pad_after = [0, 0, pad_down, pad_right]
#     temp = pad(data, pad_before, pad_after, name="pad_temp")
#     tranData = featureTrans(temp,kernel_size,stride)
#     out_channel = get_const_int(weight_indptr.shape[0]) - 1
#     # (n-k+2*p)//s+1
#     oshape = (batch,out_channel,out_height, out_width)
#     def f(n,row,h,w):
#         row_start = weight_indptr[row]
#         row_end = weight_indptr[row + 1]
#         row_elems = row_end - row_start
#         elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
#         elem = row_start + elem_idx
#         a_val = weight_data[elem]
#         #这里可能是影响速度的地方
#         #可以预处理
#         d_row = weight_indices_row[elem]
#         d_col = weight_indices_col[elem]
#         weight_val = tranData[d_row, d_col + h*out_height + w]
#         return te.sum(a_val * weight_val, axis=elem_idx,)
    
#     conv = te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')
#     return [temp, tranData, conv]


@auto_scheduler.register_workload
def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,resIdx,reshapeIdx,kernel_size,stride,padding,dilation=1,out_dtype = None):
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
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp")
    print("temp.shape:",temp.shape)
    tranData = featureTrans(temp,kernel_size,stride)
    print("trandata.shape:",tranData.shape)
    out_channel = get_const_int(weight_indptr.shape[0]) - 1
    # (n-k+2*p)//s+1
    oshape = (batch,out_channel,out_height, out_width)
    def f(n,row,h,w):
        # print("row:",row,end=' ')
        row = resIdx[row]
        # print(row)
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
    
    conv = te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')
    output = te.compute(oshape,
                       lambda n,c,h,w:
                        conv[n,reshapeIdx[c],h,w],
                       name='conv_r',
                    )
    return [output,temp,tranData]


#将n个数加和成m个，使m个数尽可能相等
def alor(L,m):
    n = len(L)
    # print(L)
    sorted_id = sorted(range(len(L)), key=lambda k: L[k], reverse=True)
    L = sorted(L, reverse=True)
    resNum = [0] * m
    resIdx = [[] for _ in range(m)]
    res = [[] for _ in range(m)]
    # print(L)
    # print(sorted_id)
    for i in range(n):
        min_index = resNum.index(min(resNum))
        # print(min_index)
        res[min_index].append(L[i])
        resNum[min_index] += L[i]
        resIdx[min_index].append(sorted_id[i])
    # print("resNum:",resNum)
    # print("res:",res)
    print('resIdx:',resIdx)
    return [resNum,res,resIdx]

def dealKernel(csr,para):
    # print(csr.indptr)
    #每个通道有多少数
    L = []
    for i in range(len(csr.indptr)-1):
        L.append((csr.indptr)[i+1] - (csr.indptr)[i])
    print(L)
    _,_,Idx = alor(L,para)
    resIdx = []
    for li in Idx:
        li.sort()
        resIdx += li
    #升序排索引
    reshapeIdx = sorted(range(len(resIdx)), key=lambda k: resIdx[k], )
    resIdx = np.array(resIdx).astype("int32")
    reshapeIdx = np.array(reshapeIdx).astype("int32")
    print("resIdx:",resIdx)
    print("reshapeIdx:",reshapeIdx)
    return resIdx,reshapeIdx


# def mec_csrmm_k1(data,weight_data,weight_indices,weight_indptr):
#     H = get_const_int(data.shape[1])
#     out_channel = get_const_int(weight_indptr.shape[0]) - 1
#     # (n-k+2*p)//s+1
#     out_size = H
#     oshape = (out_channel,out_size, out_size)
#     def f(row,h,w):
#         # print(type(weight_indptr[row]))
#         row_start = weight_indptr[row]
#         row_end = weight_indptr[row + 1]
#         row_elems = row_end - row_start
#         elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
#         elem = row_start + elem_idx
#         # elem = te.reduce_axis((row_start, row_end), name="elem")
#         a_val = weight_data[elem]
#         idx = weight_indices[elem]
#         #这里可能是影响速度的地方
#         #可以预处理
#         # weight_val = data[idx, h*out_size + w]
#         weight_val = data[idx, h, w]
#         # return te.sum(a_val * weight_val, axis=elem,)
#         return te.sum(a_val * weight_val, axis=elem_idx,)
#     return te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')


N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
kernel_size = KH
sparity = 0.8
para = 16
# for para in [8,16,32,64]:
x = np.random.randn(N, CI, H, W).astype("float32")
kernel = np.array(random_bsr_matrix(CO, CI*kernel_size*kernel_size, 1, 1, 0.1, "float32")
                .todense()).reshape(CO, CI, kernel_size, kernel_size)

#测试正确性
conv_np = conv2d_nchw_python(x, kernel, strides, padding)

csr = utils.deal_sp_kernel(kernel)
resIdx,reshapeIdx = dealKernel(csr,para)

out_size = utils.conv_out_size(H,kernel_size,padding[0],strides[0])

target = tvm.target.Target("llvm")
dev = tvm.cpu()
out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
pad_out = tvm.nd.empty((N,CI,H+2*padding[0],W+2*padding[1]), device=dev)
trans_out = tvm.nd.empty((kernel_size*CI,out_size*(H+2*padding[0])), device=dev)

#预处理
x_row = CI * kernel_size
weight_row = csr.indices % x_row
weight_col = csr.indices // x_row * out_size

print("#####################参数输出############################")
print('weight_data:',csr.data.shape)
print('weight_row:',weight_row.shape)
print('weight_col:',weight_col.shape)
print('weight_indptr:',csr.indptr.shape)
print('稀疏度：', csr.data.shape[0]/(float)(CO*CI*kernel_size*kernel_size))
print('输出尺寸：',out.shape)


#数据形状获取
data_shape = x.shape
weight_data_shape = csr.data.shape
weight_indices_shape = csr.indices.shape
weight_indptr_shape = csr.indptr.shape
resIdx_shape = resIdx.shape
reshapeIdx_shape = reshapeIdx.shape
#算子设计
input_mec = te.placeholder(data_shape,dtype='float32')
weight_data = te.placeholder(weight_data_shape,dtype='float32',name='w_data')
weight_indices_row = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_row')
weight_indices_col = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_col')
weight_indptr = te.placeholder(weight_indptr_shape,dtype='int32',name='w_indptr')
weight_resIdx = te.placeholder(resIdx_shape,dtype='int32',name='w_resIdx')
weight_reshapeIdx = te.placeholder(reshapeIdx_shape,dtype='int32',name='w_reshapeIdx')
#函数
output,padData,transData = mec_csrmm(input_mec,
                                     weight_data,
                                     weight_indices_row,
                                     weight_indices_col,
                                     weight_indptr,
                                     weight_resIdx,
                                     weight_reshapeIdx,
                                     kernel_size,
                                     strides,
                                     padding,
                                     dilation=1,)

#mod定义
# func = te.create_prim_func([input_mec,
#                             weight_data,
#                             weight_indices_row,
#                             weight_indices_col,
#                             weight_indptr,
#                             weight_resIdx,
#                             weight_reshapeIdx,
#                             output,
#                             padData,
#                             transData])   
# ir_module_main = IRModule({"main": func})
# print(ir_module_main)

print("#######################tensor关系输出##########################")
print("output.op.inputTensor:",output.op.input_tensors)
WreshapeIdx, conv = output.op.input_tensors
print("conv.op.inputTensor:",conv.op.input_tensors)
WresIdx,Windptr,Wdata,WindicesRow,WindicesCol,tran = conv.op.input_tensors
print("transreshape.op.inputTensor:",tran.op.input_tensors)
transY = tran.op.input_tensors[0]
print("transY.op.inputTensor:",transY.op.input_tensors)
padd = transY.op.input_tensors[0]
print("########################调度输出#########################")

#te创建调度
s = te.create_schedule(output.op)
#写缓存
Cachedconv = s.cache_write(conv, 'local')
print("cacheconv.op.inputTensor:",Cachedconv.op.input_tensors)
#TE下降到TIR
# m = tvm.lower(s, [input_mec,weight_data,weight_indices,weight_indptr,conv], name = 'test_mec')
#TIR创建sch
# sch = tvm.tir.Schedule(m)
# print(sch.mod)

#调度pad
print("########################padd调度#########################")
print("pad input tensor:",padd.op.input_tensors)
# PP = s.cache_read(padd.op.input_tensors[0],"local",[padd])
print("padd.op.axis:", padd.op.axis)
p_n,p_c,p_h,p_w = padd.op.axis
p_n_c = s[padd].fuse(p_n,p_c)
s[padd].parallel(p_n_c)
p_h_w = s[padd].fuse(p_h,p_w)
s[padd].unroll(p_h_w)
# s[PP].compute_at(s[padd],p_n_c)
s[padd].set_scope("global")

#调度trans
#在split、fuse之前读缓存
print("########################transY调度#########################")
print("transY input tensor:",transY.op.input_tensors)
print("transY.op.axis:",transY.op.axis)
# s[transY].compute_inline()
# DD = s.cache_read(padd,"local",[transY])
t_n,t_c,t_h,t_w = s[transY].op.axis
t_n_c = s[transY].fuse(t_n,t_c)
s[transY].parallel(t_n_c)
t_h_w = s[transY].fuse(t_h,t_w)
s[transY].unroll(t_h_w)
# s[DD].compute_at(s[transY],transY.op.axis[0])

print("########################tran调度#########################")
print("tran input tensor:",tran.op.input_tensors)
print("tran.op.axis:",tran.op.axis)
s[tran].parallel(tran.op.axis[0])
s[tran].unroll(tran.op.axis[1])

#调度conv
print("########################CONV调度#########################")
print("conv input tensor:",conv.op.input_tensors)
print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)
# DD = s.cache_read(tran,"local",[transY])

# n_s, row_s, h_s, w_s = conv.op.axis
# row_o_s,row_i_s = s[conv].split(row_s,factor = 16)
# w_o_s,w_i_s = s[conv].split(w_s, factor= 14)
# h_o_s,h_i_s = s[conv].split(h_s, factor = 14)
# s[conv].reorder(n_s,row_o_s,h_o_s,w_o_s,row_i_s,h_i_s,w_i_s)
# row_s = s[conv].fuse(n_s,row_o_s,h_o_s,w_o_s)
# s[conv].unroll(h_i_s)
# s[conv].parallel(row_s)
# s[conv].vectorize(w_i_s)

n_s, row_s, h_s, w_s = conv.op.axis
h_w_s = s[conv].fuse(h_s, w_s)
row_s = s[conv].fuse(n_s, row_s)
# s[conv].reorder(nr_s,h_w_s)
s[conv].unroll(h_w_s)
s[conv].parallel(row_s)
# s[conv].vectorize(w_i_s)

# s[Cachedconv].compute_at(s[conv],row_o_s)
s[Cachedconv].compute_at(s[conv],row_s)

#调度cacheWrite
print("########################cacheWrite调度#########################")
print("cacheConv input tensor:",Cachedconv.op.input_tensors)
print("cacheConv.op.axis:", type(Cachedconv.op.axis), Cachedconv.op.axis)
print("cacheConv.op.reduce_axis:", type(Cachedconv.op.reduce_axis), Cachedconv.op.reduce_axis)

# WresIdx,Windptr,Wdata,WindicesRow,WindicesCol,tran
# WI = s.cache_read(Windptr,"global",[Cachedconv])
# WD = s.cache_read(Wdata,"global",[Cachedconv])
# WR = s.cache_read(WindicesRow,"global",[Cachedconv])
# WC = s.cache_read(WindicesCol,"global",[Cachedconv])
# RR = s.cache_read(tran,"global",[Cachedconv])

# print("conv.op.axis:",conv.op.axis)
n,r_s,hh_s,ww_s = Cachedconv.op.axis
# r_o_s,r_i_s = s[Cachedconv].split(r_s,nparts = 4)
# print("Cachedconv.op.reduce_axis:",Cachedconv.op.reduce_axis)
elem_idx_s = Cachedconv.op.reduce_axis[0]
#拆轴会损害时间
# e_o_s,e_i_s = s[Cachedconv].split(elem_idx_s,factor=2)
# s[Cachedconv].reorder(n,r_s,e_o_s,e_i_s,hh_s,ww_s)
s[Cachedconv].reorder(n,r_s,elem_idx_s,hh_s,ww_s)
hw = s[Cachedconv].fuse(hh_s,ww_s)
s[Cachedconv].unroll(hw)
# s[Cachedconv].vectorize(w_i_s)

# s[WI].compute_at(s[Cachedconv],n)
# s[WD].compute_at(s[Cachedconv],e_o_s)
# s[WR].compute_at(s[Cachedconv],n)
# s[WC].compute_at(s[Cachedconv],n)
# s[RR].compute_at(s[Cachedconv],r_s)

#调度output
print("########################output调度#########################")
print("output input tensor:",output.op.input_tensors)
print("output.op.axis:", type(output.op.axis), output.op.axis)
p_c = s[output].fuse(output.op.axis[0],output.op.axis[1])
s[output].parallel(p_c)
u_c = s[output].fuse(output.op.axis[2],output.op.axis[3])
s[output].unroll(u_c)

# 获取 GPU 线程索引
block_x = te.thread_axis("blockIdx.x")
s[output].bind(s[output].op.axis[0],block_x)

#保存调度图
sch = s
prefix = (str)(H) +'-'+(str)(CO)+'-'+(str)(CI)+'-'+(str)(KH)
tedd.viz_schedule_tree(sch, dot_file_path="./test_te_tir_schtree/"+prefix+"/scheduletree.dot")
os.system("dot -Tpng "+"./test_te_tir_schtree/"+prefix+"/scheduletree.dot -o "+"./test_te_tir_schtree/"+prefix+"/scheduletree.png")
sch = sch.normalize()
tedd.viz_schedule_tree(sch, dot_file_path="./test_te_tir_schtree/"+prefix+"/scheduletree2.dot")
os.system("dot -Tpng "+"./test_te_tir_schtree/"+prefix+"/scheduletree2.dot -o "+"./test_te_tir_schtree/"+prefix+"/scheduletree2.png")

#TEsch结束后需要lower
m = tvm.lower(s, [input_mec,
                  weight_data,
                  weight_indices_row,
                  weight_indices_col,
                  weight_indptr,
                  weight_resIdx,
                  weight_reshapeIdx,
                  output,
                  padData,
                  transData], name = 'test_mec')
# print(m)
mod = tvm.build(m, target=target)

#TIRsch结束仍是IRModule
# mod = tvm.build(m,target=target)
# mod(x_tvm,weight_data_tvm,weight_indices_tvm,weight_indptr_tvm,y)
# print(y)

#数据转换
x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(x,csr.indptr,weight_row,weight_col,csr.data,out))
weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))
resIdx_tvm = tvm.nd.array(resIdx,device=dev) 
reshapeIdx_tvm = tvm.nd.array(reshapeIdx,device=dev) 
pad_out_tvm = tvm.nd.array(pad_out,device=dev) 
trans_out_tvm = tvm.nd.array(trans_out,device=dev) 

#结果测试
print("dataNum:", csr.data.shape)
print("线程：",para)
#评估时延
timer = mod.time_evaluator(mod.entry_name, dev=dev, number=1000)
result = (
    timer(x_tvm, 
          weight_data_tvm, 
          weight_indices_row_tvm,
          weight_indices_col_tvm, 
          weight_indptr_tvm,
          resIdx_tvm,
          reshapeIdx_tvm, 
          y,
          pad_out_tvm,
          trans_out_tvm
            ).mean
        * 1e3
    )
print(
    "our Convolution: %f ms" 
    % result
)


import psutil

# 获取当前进程的信息
current_process = psutil.Process()

# 获取当前进程的线程数
thread_count = current_process.num_threads()

print("当前进程的线程数:", thread_count)

# 检查结果
mod(x_tvm, weight_data_tvm, weight_indices_row_tvm,weight_indices_col_tvm, weight_indptr_tvm,resIdx_tvm,reshapeIdx_tvm,y,pad_out_tvm,trans_out_tvm)
# print(x[0][0][0][0])
# print(pad_out_tvm.numpy()[0][0][1][1])
# print("x 数据:")
# print(x_tvm.numpy())
# # print(x_tvm.numpy()[0][0][0])
# print("pad 数据：")
# print(pad_out_tvm.numpy())
# # print(pad_out_tvm.numpy()[0][0][1])
# print("转换数据：")
# print(trans_out_tvm.numpy())
# print(conv_np[0][0][0][0])
# print(y.numpy()[0][0][0][0])
# np.testing.assert_allclose(conv_np, y.numpy(), rtol=1e-3)

# data = pad_out_tvm.numpy()
# n,c,h,w = data.shape
# out_size = (h - kernel_size) // strides[0] + 1
# fea = np.zeros((kernel_size,c,h,out_size)).astype('float32')
# for col_axis in range(kernel_size):
#     for channel_s in range(c):
#         for H_s in range(h):
#             for out_s in range(out_size):
#                 t = data[0][channel_s][H_s][col_axis+out_s]
#                 fea[col_axis][channel_s][H_s][out_s] = t
# ftt = np.zeros((kernel_size*c,h*out_size)).astype('float32')
# for row in range(kernel_size*c):
#     for col in range(out_size*h):
#         # print(row // c, row % c, col // out_size, col % out_size)
#         t = fea[row // c][row % c][col // out_size][col % out_size]
#         ftt[row][col] = t 

# fea = te.compute(
#     (kernel_size,C,H,out_size),
#     lambda col_axis,channel_s,H_s,out_s:
#         data[0,channel_s,H_s,col_axis+out_s],
#     name="trans_Y"
# )
# return te.compute(
#     (kernel_size*C,out_size*H),
#     lambda row,col:
#         fea[row // (kernel_size*C),row % (kernel_size*C),col//(H*out_size),col%(H*out_size)],
#     name='trans_reshape'
# )

# from tvm import relay
# from tvm.contrib import graph_executor
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build_module.build(m, params={}, target=target)
#     # file = open('sp_json.json','w')
#     # file.write(json.dumps(lib.graph_json))
#     # file.close()
#     print("write file over!")

# #执行稀疏模型
# module = graph_executor.GraphModule(lib["default"](dev))
# # module = graph_executor.create(lib.graph_json, lib.module, dev)
# # module.set_input("input",data)
# # Evaluate
# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

