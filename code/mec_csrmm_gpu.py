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
# from tvm.contrib import tedd
from collections import namedtuple
import sys
sys.path.insert(0,sys.path[0]+'/../..')
import utils
import os
import numpy as np


@auto_scheduler.register_workload
# def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,resIdx,reshapeIdx,kernel_size,stride,padding,dilation=1,out_dtype = None):
def mec_csrmm(data_shape,weight_data_shape,weight_indices_shape,weight_indptr_shape,resIdx_shape,reshapeIdx_shape,kernel_size,stride,padding,dilation=1,out_dtype = None):
    # data = te.placeholder(data_shape,dtype='float32',name = 'input')
    weight_data = te.placeholder(weight_data_shape,dtype='float32',name='w_data')
    weight_indices_row = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_row')
    weight_indices_col = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_col')
    weight_indptr = te.placeholder(weight_indptr_shape,dtype='int32',name='w_indptr')
    resIdx = te.placeholder(resIdx_shape,dtype='int32',name='w_resIdx')
    reshapeIdx = te.placeholder(reshapeIdx_shape,dtype='int32',name='w_reshapeIdx')

    if out_dtype is None:
        out_dtype = weight_data.dtype
        # out_dtype = tranData.dtype
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
    batch, in_channel, in_height, in_width = data_shape
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
    # temp = pad(data, pad_before, pad_after, name="pad_temp")
    # print("temp.shape:",temp.shape)
    tranData = te.placeholder((kernel_size*in_channel,out_height*(in_height+2*padding[0])),dtype='float32',name = 'input')
    # tranData = featureTrans(temp,kernel_size,stride)
    # print("trandata.shape:",tranData.shape)
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
    # output = te.compute(oshape,
    #                    lambda n,c,h,w:
    #                     conv[n,reshapeIdx[c],h,w],
    #                    name='conv_r',
    #                 )
    # return [output,temp,tranData]
    # return [data,weight_data, weight_indices_row, weight_indices_col, weight_indptr, resIdx, reshapeIdx,output,temp,tranData,conv]
    return [tranData,weight_data, weight_indices_row, weight_indices_col, weight_indptr, resIdx, reshapeIdx,conv]


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


#返回运行时间
def conv(input_file=None,
         kernel_file=None,
         padding=(1,1),
         strides=(1,1),
         kISone = False):
    x = np.load(input_file)
    # print("x shape:",x.shape)
    N,CI,H,W = x.shape
    print("input shape:",x.shape)
    kernel = np.load(kernel_file)
    print("kernel shape:",kernel.shape)
    CO,CI,KH,KW = kernel.shape
    # N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 358, 358, 3, 3, (1, 1), (1, 1)
    kernel_size = KH
    # sparity = 0.9
    # para = 16
    para = 64
    # kernel_size*C,out_size*H
    # for para in [8,16,32,64]:
    csr = utils.deal_sp_kernel(kernel)
    resIdx,reshapeIdx = dealKernel(csr,para)
    out_size = utils.conv_out_size(H,kernel_size,padding[0],strides[0])
    Xtrans = np.random.randn(kernel_size*CI,out_size*(H+2*padding[0])).astype("float32")
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)
    out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
    # pad_out = tvm.nd.empty((N,CI,H+2*padding[0],W+2*padding[1]), device=dev)
    # trans_out = tvm.nd.empty((kernel_size*CI,out_size*(H+2*padding[0])), device=dev)
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
    #函数
    # input_mec, weight_data, weight_indices_row, weight_indices_col, weight_indptr,weight_resIdx, weight_reshapeIdx,output,padData,transData,con = mec_csrmm(data_shape,
    input_mec, weight_data, weight_indices_row, weight_indices_col, weight_indptr,weight_resIdx, weight_reshapeIdx,conv = mec_csrmm(data_shape,
                                        weight_data_shape,
                                        weight_indices_shape,
                                        weight_indptr_shape,
                                        resIdx_shape,
                                        reshapeIdx_shape,
                                        kernel_size,
                                        strides,
                                        padding,
                                        dilation=1,)

    #调度conv
    # print("########################CONV调度#########################")
    # print("conv input tensor:",conv.op.input_tensors)
    # print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
    # print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)

    output = conv
    # 获取 GPU 线程索引
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")

    #te创建调度
    s = te.create_schedule(output.op)


    nt = para
    outOP = conv
    x, y, z, zz = s[outOP].op.axis
    z = s[outOP].fuse(z, zz)
    y = s[outOP].fuse(y, z)
    fused = s[outOP].fuse(x, y)
    bx, tx = s[outOP].split(fused, factor=nt)
    s[outOP].bind(bx,block_x)
    s[outOP].bind(tx,thread_x)

    m = tvm.lower(s, [input_mec,
                    weight_data,
                    weight_indices_row,
                    weight_indices_col,
                    weight_indptr,
                    weight_resIdx,
                    weight_reshapeIdx,
                    #   output,
                    conv], name = 'test_mec')
    # print(m)


    #调度conv
    # print("########################CONV调度#########################")
    # print("conv input tensor:",conv.op.input_tensors)
    # print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
    # print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)

    # print("线程绑定成功")

    mod = tvm.build(m, target=target)

    #数据转换
    # x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(x,csr.indptr,weight_row,weight_col,csr.data,out))
    x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(Xtrans,csr.indptr,weight_row,weight_col,csr.data,out))
    weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))

    resIdx_tvm = tvm.nd.array(resIdx,device=dev)
    reshapeIdx_tvm = tvm.nd.array(reshapeIdx,device=dev)
    # pad_out_tvm = tvm.nd.array(pad_out,device=dev)
    # trans_out_tvm = tvm.nd.array(trans_out,device=dev)
    conv_out_tvm = tvm.nd.array(out.shape,device=dev)

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
            #   pad_out_tvm,
            #   trans_out_tvm
            #   conv_out_tvm
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
    return result



# Save to the d2ltvm package.
def split_axis(factors, sch, op, axis):
        """Splitting an axis into factors

        Parameters
        ----------
        factors: array of integers
            The factors that the split applies
        sch: tvm.te.schedule.Schedule
            The tvm schedule
        op: tvm.te.tensor.Operation
            The stage to be applied
        axis: tvm.te.schedule.IterVar
            axis to split

        Returns
        -------
        axes : list of Axis
            The transformed axes.
        """
        ret = []
        for i in range(0, len(factors)):
            ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
            ret.append(ax0)
            axis = ax1
        return ret + [axis]




# def tiling(data_shape,weight_data_shape,weight_indices_shape,weight_indptr_shape,resIdx_shape,reshapeIdx_shape,kernel_size,strides,padding):
def tiling():
    # N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 512, 512, 3, 3, (1, 1), (1, 1)
    kernel_size = KH
    sparity = 0.9
    para = 16
    # kernel_size*C,out_size*H

    tile_c = [1,4, 8]
    tile_h = [1,2, 2]
    tile_w = [2,16, 2]
    tile_rc = [1, 1]
    tile_rh = [1, 3]
    tile_rw = [1, 1]
    # for para in [8,16,32,64]:
    x = np.random.randn(N, CI, H, W).astype("float32")

    kernel = np.array(random_bsr_matrix(CO, CI*kernel_size*kernel_size, 1, 1, 0.1, "float32")
                    .todense()).reshape(CO, CI, kernel_size, kernel_size)

    #测试正确性
    # conv_np = conv2d_nchw_python(x, kernel, strides, padding)

    csr = utils.deal_sp_kernel(kernel)
    resIdx,reshapeIdx = dealKernel(csr,para)

    out_size = utils.conv_out_size(H,kernel_size,padding[0],strides[0])

    Xtrans = np.random.randn(kernel_size*CI,out_size*(H+2*padding[0])).astype("float32")

    # target = tvm.target.Target("llvm")
    target = tvm.target.Target("cuda")
    # dev = tvm.cpu()
    dev = tvm.cuda(0)
    # dev = tvm.gpu()
    out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
    # pad_out = tvm.nd.empty((N,CI,H+2*padding[0],W+2*padding[1]), device=dev)
    # trans_out = tvm.nd.empty((kernel_size*CI,out_size*(H+2*padding[0])), device=dev)

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

    input_mec, weight_data, weight_indices_row, weight_indices_col, weight_indptr,weight_resIdx, weight_reshapeIdx,Y = mec_csrmm(data_shape,
                                     weight_data_shape,
                                     weight_indices_shape,
                                     weight_indptr_shape,
                                     resIdx_shape,
                                     reshapeIdx_shape,
                                     kernel_size,
                                     strides,
                                     padding,
                                     dilation=1,)
    sch = te.create_schedule(Y.op)
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    # sch[PaddedX].compute_inline()
    n,c, h, w = sch[Y].op.axis
    # Yrc, = sch[Y].op.reduce_axis
    # print(Yrc)
    # input()
    # sch[Y].reorder(n,c,Yrc,h,w)
    
    nt = 64
    s = sch
    outOP = Y
    x, y, z, zz = s[outOP].op.axis
    
    # tx = s[outOP].fuse(z, zz)
    # bx = s[outOP].fuse(x, y)
    
    z = s[outOP].fuse(z, zz)
    y = s[outOP].fuse(y, z)
    fused = s[outOP].fuse(x, y)
    bx, tx = s[outOP].split(fused, factor=nt)

    # elem = s[output].op.reduce_axis
    # print("output reduce_axis:",output.op.reduce_axis)
    # s[output].reorder(bx,tx,elem)

    # s[output].bind(s[output].op.axis[2],block_x)
    s[outOP].bind(bx,block_x)
    # s[output].bind(s[output].op.axis[3],thread_x)
    s[outOP].bind(tx,thread_x)


    # outOP = output
    # x, y, z, zz = s[outOP].op.axis
    # z = s[outOP].fuse(z, zz)
    # y = s[outOP].fuse(y, z)
    # fused = s[outOP].fuse(x, y)
    # bx, tx = s[outOP].split(fused, factor=nt)
    # s[outOP].bind(bx,block_x)
    # s[outOP].bind(tx,thread_x)
    # sch[Y].bind(n, te.thread_axis("threadIdx.x"))
    

    # YL = sch.cache_write(Y, 'local')

    # # create cache stage
    # # XX = sch.cache_read(input_mec, 'shared', [YL])
    # KK = sch.cache_read(weight_data, 'shared', [YL])
    # # XL = sch.cache_read(XX, 'local', [YL])
    # KL = sch.cache_read(KK, 'local', [YL])

    # bc, vc, tc, ic = split_axis(tile_c, sch, Y, c)
    # bh, vh, th, ih = split_axis(tile_h, sch, Y, h)
    # bw, vw, tw, iw = split_axis(tile_w, sch, Y, w)

    # sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
    # sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
    # sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
    # sch[Y].bind(vc, te.thread_axis("vthread"))
    # sch[Y].bind(vh, te.thread_axis("vthread"))
    # sch[Y].bind(vw, te.thread_axis("vthread"))
    # sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
    # sch[Y].bind(th, te.thread_axis("threadIdx.y"))
    # sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
    # sch[Y].reorder(bc, bh, bw, vc, vh, vw, tc, th, tw, ic, ih, iw)

    # sch[YL].compute_at(sch[Y], tw)

    # # tile reduction axes
    # n,c, h, w = sch[YL].op.axis
    # rc, = sch[YL].op.reduce_axis
    # tk = 32
    # ko, ki = sch[YL].split(rc, 32)
    # # print(rc)
    # # input()
    # # rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
    # # rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
    # # rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
    # sch[YL].reorder(c, ko, ki, h, w)

    # # sch[XX].compute_at(sch[YL], ko)
    # sch[KK].compute_at(sch[YL], ko)
    # # sch[XL].compute_at(sch[YL], ki)
    # sch[KL].compute_at(sch[YL], ki)

    # # cooperative fetching
    # # for load in [XX, KK]:
    # # for load in [XX,]:
    # for load in [KK,]:
    #     args = sch[load].op.axis
    #     fused = sch[load].fuse(*args)
    #     # align thread layout
    #     tz, fused = sch[load].split(fused, nparts=tile_c[0])
    #     ty, fused = sch[load].split(fused, nparts=tile_h[0])
    #     tx, _ = sch[load].split(fused, nparts=tile_w[0])
    #     sch[load].bind(tz, te.thread_axis("threadIdx.z"))
    #     sch[load].bind(ty, te.thread_axis("threadIdx.y"))
    #     sch[load].bind(tx, te.thread_axis("threadIdx.x"))

    m = tvm.lower(sch, [input_mec,
                  weight_data,
                  weight_indices_row,
                  weight_indices_col,
                  weight_indptr,
                  weight_resIdx,
                  weight_reshapeIdx,
                #   output,
                  Y], name = 'test_mec')
    print(m)
    mod = tvm.build(m, target=target)
    x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(Xtrans,csr.indptr,weight_row,weight_col,csr.data,out))
    weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))
    resIdx_tvm = tvm.nd.array(resIdx,device=dev)
    reshapeIdx_tvm = tvm.nd.array(reshapeIdx,device=dev)
    conv_out_tvm = tvm.nd.array(out.shape,device=dev)
    print('#'*50)
    print("x_tvm:",x_tvm.shape)
    print("y:",y.shape)
    print("conv_out:",conv_out_tvm.shape)
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
                ).mean
            * 1e3
        )
    print(
        "our Convolution: %f ms"
        % result
    )
# import psutil
# 获取当前系统上的所有共享内存段
# shms = [x for x in psutil.process_iter() if hasattr(x, 'numa')]

# for shm in shms:
    # print("共享内存 ID:", shm.pid)
    # print("共享内存大小:", shm.memory_info().rss / (1024 * 1024), "MB")

# test
# input()
# m = tiling()

# #函数
# # input_mec, weight_data, weight_indices_row, weight_indices_col, weight_indptr,weight_resIdx, weight_reshapeIdx,output,padData,transData,con = mec_csrmm(data_shape,
# input_mec, weight_data, weight_indices_row, weight_indices_col, weight_indptr,weight_resIdx, weight_reshapeIdx,conv = mec_csrmm(data_shape,
#                                      weight_data_shape,
#                                      weight_indices_shape,
#                                      weight_indptr_shape,
#                                      resIdx_shape,
#                                      reshapeIdx_shape,
#                                      kernel_size,
#                                      strides,
#                                      padding,
#                                      dilation=1,)


# print("########################调度输出#########################")
# #调度conv
# print("########################CONV调度#########################")
# print("conv input tensor:",conv.op.input_tensors)
# print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
# print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)

# output = conv
# # 获取 GPU 线程索引
# block_x = te.thread_axis("blockIdx.x")
# thread_x = te.thread_axis("threadIdx.x")

# #te创建调度
# s = te.create_schedule(output.op)


# # elem = s[conv].op.reduce_axis
# # s[conv].compute_inline()
# # s[conv].compute_root()
# # s[output].compute_inline()

# # s[con].compute_at(s[output],output.op.axis[0])


# # YL = s.cache_write(transData, 'shared')

# # XX = s.cache_read(input_mec,"shared",[YL])
# # XL = s.cache_read(XX, 'local', [YL])

# nt = 64
# outOP = conv
# x, y, z, zz = s[outOP].op.axis
# z = s[outOP].fuse(z, zz)
# y = s[outOP].fuse(y, z)
# fused = s[outOP].fuse(x, y)
# bx, tx = s[outOP].split(fused, factor=nt)

# # elem = s[output].op.reduce_axis
# # print("output reduce_axis:",output.op.reduce_axis)
# # s[output].reorder(bx,tx,elem)



# # s[output].bind(s[output].op.axis[2],block_x)
# s[outOP].bind(bx,block_x)
# # s[output].bind(s[output].op.axis[3],thread_x)
# s[outOP].bind(tx,thread_x)


# # outOP = output
# # x, y, z, zz = s[outOP].op.axis
# # z = s[outOP].fuse(z, zz)
# # y = s[outOP].fuse(y, z)
# # fused = s[outOP].fuse(x, y)
# # bx, tx = s[outOP].split(fused, factor=nt)
# # s[outOP].bind(bx,block_x)
# # s[outOP].bind(tx,thread_x)


# #TEsch结束后需要lower
# m = tvm.lower(s, [input_mec,
#                   weight_data,
#                   weight_indices_row,
#                   weight_indices_col,
#                   weight_indptr,
#                   weight_resIdx,
#                   weight_reshapeIdx,
#                 #   output,
#                   conv], name = 'test_mec')
# print(m)

# #写缓存
# # Cachedconv = s.cache_write(conv, 'local')
# # print("cacheconv.op.inputTensor:",Cachedconv.op.input_tensors)


# #调度conv
# print("########################CONV调度#########################")
# print("conv input tensor:",conv.op.input_tensors)
# print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
# print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)

# #调度output
# # print("########################output调度#########################")
# # print("output input tensor:",output.op.input_tensors)
# # print("output.op.axis:", type(output.op.axis), output.op.axis)
# # p_c = s[output].fuse(output.op.axis[0],output.op.axis[1])
# # s[output].parallel(p_c)
# # u_c = s[output].fuse(output.op.axis[2],output.op.axis[3])
# # s[output].unroll(u_c)


# # s[output].bind(s[output].op.axis[2],block_x)
# # s[output].bind(s[output].op.axis[3],thread_x)

# # s[padd].bind(s[padd].op.axis[3],thread_x)
# # s[output].bind(u_c,block_x)


# mod = tvm.build(m, target=target)

# #TIRsch结束仍是IRModule
# # mod = tvm.build(m,target=target)
# # mod(x_tvm,weight_data_tvm,weight_indices_tvm,weight_indptr_tvm,y)
# # print(y)

# #数据转换

# # x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(x,csr.indptr,weight_row,weight_col,csr.data,out))
# x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(Xtrans,csr.indptr,weight_row,weight_col,csr.data,out))
# weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))

# resIdx_tvm = tvm.nd.array(resIdx,device=dev)
# reshapeIdx_tvm = tvm.nd.array(reshapeIdx,device=dev)
# # pad_out_tvm = tvm.nd.array(pad_out,device=dev)
# # trans_out_tvm = tvm.nd.array(trans_out,device=dev)
# conv_out_tvm = tvm.nd.array(out.shape,device=dev)

# print('#'*50)
# print("x_tvm:",x_tvm.shape)
# print("y:",y.shape)
# print("conv_out:",conv_out_tvm.shape)
# #结果测试
# print("dataNum:", csr.data.shape)
# print("线程：",para)
# #评估时延
# timer = mod.time_evaluator(mod.entry_name, dev=dev, number=1000)
# result = (
#     timer(x_tvm,
#           weight_data_tvm,
#           weight_indices_row_tvm,
#           weight_indices_col_tvm,
#           weight_indptr_tvm,
#           resIdx_tvm,
#           reshapeIdx_tvm,
#           y,
#         #   pad_out_tvm,
#         #   trans_out_tvm
#         #   conv_out_tvm
#             ).mean
#         * 1e3
#     )
# print(
#     "our Convolution: %f ms"
#     % result
# )
# # input("over")

# import psutil

# # 获取当前进程的信息
# current_process = psutil.Process()

# # 获取当前进程的线程数
# thread_count = current_process.num_threads()

# print("当前进程的线程数:", thread_count)






# ##测试sparseMEC算子的te转tir
# from functools import partial, reduce
# import numpy as np
# import tvm
# from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
# from tvm.topi.nn.pad import pad
# from tvm.topi.nn.utils import get_pad_tuple
# from tvm import auto_scheduler
# from tvm.topi.testing import conv2d_nchw_python
# import tvm.testing
# from tvm import te, auto_scheduler, runtime
# from tvm.topi.sparse.utils import random_bsr_matrix
# from tvm import IRModule
# from tvm.contrib import tedd
# from collections import namedtuple
# import sys
# sys.path.insert(0,sys.path[0]+'/../..')
# import utils
# import os
# import numpy as np
#
# @auto_scheduler.register_workload
# def featureTrans(data,kernel_size,stride):
#     _,C,H,_ = get_const_tuple(data.shape)
#     out_size = (H - kernel_size) // stride + 1
#
#     # @partial(te.compute, (kernel_size,C,H,out_size), name="trans_Y")
#     # def fea(col_axis,channel_s,H_s,out_s):
#     #     return data[0,channel_s,H_s,col_axis+out_s]
#     fea = te.compute(
#         (kernel_size,C,H,out_size),
#         lambda col_axis,channel_s,H_s,out_s:
#             data[0,channel_s,H_s,col_axis+out_s],
#         name="trans_Y"
#     )
#     return te.compute(
#         (kernel_size*C,out_size*H),
#         lambda row,col:
#             fea[row // C,row % C,col // out_size,col % out_size],
#         name='trans_reshape'
#     )
#
# # @auto_scheduler.register_workload
# # def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,kernel_size,stride,padding,dilation=1,out_dtype = None):
# #     if out_dtype is None:
# #         out_dtype = data.dtype
# #     assert isinstance(stride, int) or len(stride) == 2
# #     assert isinstance(dilation, int) or len(dilation) == 2
# #     if isinstance(stride, int):
# #         stride = stride_h = stride_w = stride
# #     else:
# #         stride_h, stride_w = stride
# #         stride = stride_h
# #     if isinstance(dilation, int):
# #         dilation_h = dilation_w = dilation
# #     else:
# #         dilation_h, dilation_w = dilation
# #     batch, in_channel, in_height, in_width = data.shape
# #     # compute the output shape
# #     dilated_kernel_h = (kernel_size - 1) * dilation_h + 1
# #     dilated_kernel_w = (kernel_size - 1) * dilation_w + 1
# #     pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
# #         padding, (dilated_kernel_h, dilated_kernel_w)
# #     )
# #     # out_channel = num_filter
# #     out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
# #     out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
# #     # compute graph
# #     pad_before = [0, 0, pad_top, pad_left]
# #     pad_after = [0, 0, pad_down, pad_right]
# #     temp = pad(data, pad_before, pad_after, name="pad_temp")
# #     tranData = featureTrans(temp,kernel_size,stride)
# #     out_channel = get_const_int(weight_indptr.shape[0]) - 1
# #     # (n-k+2*p)//s+1
# #     oshape = (batch,out_channel,out_height, out_width)
# #     def f(n,row,h,w):
# #         row_start = weight_indptr[row]
# #         row_end = weight_indptr[row + 1]
# #         row_elems = row_end - row_start
# #         elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
# #         elem = row_start + elem_idx
# #         a_val = weight_data[elem]
# #         #这里可能是影响速度的地方
# #         #可以预处理
# #         d_row = weight_indices_row[elem]
# #         d_col = weight_indices_col[elem]
# #         weight_val = tranData[d_row, d_col + h*out_height + w]
# #         return te.sum(a_val * weight_val, axis=elem_idx,)
#
# #     conv = te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')
# #     return [temp, tranData, conv]
#
#
# @auto_scheduler.register_workload
# def mec_csrmm(data,weight_data,weight_indices_row,weight_indices_col,weight_indptr,resIdx,reshapeIdx,kernel_size,stride,padding,dilation=1,out_dtype = None):
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
#     out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
#     out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
#     # compute graph
#     pad_before = [0, 0, pad_top, pad_left]
#     pad_after = [0, 0, pad_down, pad_right]
#     temp = pad(data, pad_before, pad_after, name="pad_temp")
#     print("temp.shape:",temp.shape)
#     tranData = featureTrans(temp,kernel_size,stride)
#     print("trandata.shape:",tranData.shape)
#     out_channel = get_const_int(weight_indptr.shape[0]) - 1
#     # (n-k+2*p)//s+1
#     oshape = (batch,out_channel,out_height, out_width)
#     def f(n,row,h,w):
#         # print("row:",row,end=' ')
#         row = resIdx[row]
#         # print(row)
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
#
#     conv = te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')
#     output = te.compute(oshape,
#                        lambda n,c,h,w:
#                         conv[n,reshapeIdx[c],h,w],
#                        name='conv_r',
#                     )
#     return [output,temp,tranData]
#
#
# #将n个数加和成m个，使m个数尽可能相等
# def alor(L,m):
#     n = len(L)
#     # print(L)
#     sorted_id = sorted(range(len(L)), key=lambda k: L[k], reverse=True)
#     L = sorted(L, reverse=True)
#     resNum = [0] * m
#     resIdx = [[] for _ in range(m)]
#     res = [[] for _ in range(m)]
#     # print(L)
#     # print(sorted_id)
#     for i in range(n):
#         min_index = resNum.index(min(resNum))
#         # print(min_index)
#         res[min_index].append(L[i])
#         resNum[min_index] += L[i]
#         resIdx[min_index].append(sorted_id[i])
#     # print("resNum:",resNum)
#     # print("res:",res)
#     print('resIdx:',resIdx)
#     return [resNum,res,resIdx]
#
# def dealKernel(csr,para):
#     # print(csr.indptr)
#     #每个通道有多少数
#     L = []
#     for i in range(len(csr.indptr)-1):
#         L.append((csr.indptr)[i+1] - (csr.indptr)[i])
#     print(L)
#     _,_,Idx = alor(L,para)
#     resIdx = []
#     for li in Idx:
#         li.sort()
#         resIdx += li
#     #升序排索引
#     reshapeIdx = sorted(range(len(resIdx)), key=lambda k: resIdx[k], )
#     resIdx = np.array(resIdx).astype("int32")
#     reshapeIdx = np.array(reshapeIdx).astype("int32")
#     print("resIdx:",resIdx)
#     print("reshapeIdx:",reshapeIdx)
#     return resIdx,reshapeIdx
#
#
# # def mec_csrmm_k1(data,weight_data,weight_indices,weight_indptr):
# #     H = get_const_int(data.shape[1])
# #     out_channel = get_const_int(weight_indptr.shape[0]) - 1
# #     # (n-k+2*p)//s+1
# #     out_size = H
# #     oshape = (out_channel,out_size, out_size)
# #     def f(row,h,w):
# #         # print(type(weight_indptr[row]))
# #         row_start = weight_indptr[row]
# #         row_end = weight_indptr[row + 1]
# #         row_elems = row_end - row_start
# #         elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
# #         elem = row_start + elem_idx
# #         # elem = te.reduce_axis((row_start, row_end), name="elem")
# #         a_val = weight_data[elem]
# #         idx = weight_indices[elem]
# #         #这里可能是影响速度的地方
# #         #可以预处理
# #         # weight_val = data[idx, h*out_size + w]
# #         weight_val = data[idx, h, w]
# #         # return te.sum(a_val * weight_val, axis=elem,)
# #         return te.sum(a_val * weight_val, axis=elem_idx,)
# #     return te.compute(oshape, f, tag="mec_csrmm", name='mec_csrmm_k1')
#
#
# N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
# kernel_size = KH
# sparity = 0.8
# para = 16
# # for para in [8,16,32,64]:
# x = np.random.randn(N, CI, H, W).astype("float32")
# kernel = np.array(random_bsr_matrix(CO, CI*kernel_size*kernel_size, 1, 1, 0.1, "float32")
#                 .todense()).reshape(CO, CI, kernel_size, kernel_size)
#
# #测试正确性
# conv_np = conv2d_nchw_python(x, kernel, strides, padding)
#
# csr = utils.deal_sp_kernel(kernel)
# resIdx,reshapeIdx = dealKernel(csr,para)
#
# out_size = utils.conv_out_size(H,kernel_size,padding[0],strides[0])
#
# # target = tvm.target.Target("llvm")
# target = tvm.target.Target("cuda")
# # dev = tvm.cpu()
# dev = tvm.cuda(0)
# out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
# pad_out = tvm.nd.empty((N,CI,H+2*padding[0],W+2*padding[1]), device=dev)
# trans_out = tvm.nd.empty((kernel_size*CI,out_size*(H+2*padding[0])), device=dev)
#
# #预处理
# x_row = CI * kernel_size
# weight_row = csr.indices % x_row
# weight_col = csr.indices // x_row * out_size
#
# print("#####################参数输出############################")
# print('weight_data:',csr.data.shape)
# print('weight_row:',weight_row.shape)
# print('weight_col:',weight_col.shape)
# print('weight_indptr:',csr.indptr.shape)
# print('稀疏度：', csr.data.shape[0]/(float)(CO*CI*kernel_size*kernel_size))
# print('输出尺寸：',out.shape)
#
#
# #数据形状获取
# data_shape = x.shape
# weight_data_shape = csr.data.shape
# weight_indices_shape = csr.indices.shape
# weight_indptr_shape = csr.indptr.shape
# resIdx_shape = resIdx.shape
# reshapeIdx_shape = reshapeIdx.shape
# #算子设计
# input_mec = te.placeholder(data_shape,dtype='float32')
# weight_data = te.placeholder(weight_data_shape,dtype='float32',name='w_data')
# weight_indices_row = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_row')
# weight_indices_col = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices_col')
# weight_indptr = te.placeholder(weight_indptr_shape,dtype='int32',name='w_indptr')
# weight_resIdx = te.placeholder(resIdx_shape,dtype='int32',name='w_resIdx')
# weight_reshapeIdx = te.placeholder(reshapeIdx_shape,dtype='int32',name='w_reshapeIdx')
# #函数
# output,padData,transData = mec_csrmm(input_mec,
#                                      weight_data,
#                                      weight_indices_row,
#                                      weight_indices_col,
#                                      weight_indptr,
#                                      weight_resIdx,
#                                      weight_reshapeIdx,
#                                      kernel_size,
#                                      strides,
#                                      padding,
#                                      dilation=1,)
#
# #mod定义
# # func = te.create_prim_func([input_mec,
# #                             weight_data,
# #                             weight_indices_row,
# #                             weight_indices_col,
# #                             weight_indptr,
# #                             weight_resIdx,
# #                             weight_reshapeIdx,
# #                             output,
# #                             padData,
# #                             transData])
# # ir_module_main = IRModule({"main": func})
# # print(ir_module_main)
#
# print("#######################tensor关系输出##########################")
# print("output.op.inputTensor:",output.op.input_tensors)
# WreshapeIdx, conv = output.op.input_tensors
# print("conv.op.inputTensor:",conv.op.input_tensors)
# WresIdx,Windptr,Wdata,WindicesRow,WindicesCol,tran = conv.op.input_tensors
# print("transreshape.op.inputTensor:",tran.op.input_tensors)
# transY = tran.op.input_tensors[0]
# print("transY.op.inputTensor:",transY.op.input_tensors)
# padd = transY.op.input_tensors[0]
# print("########################调度输出#########################")
#
# #te创建调度
# s = te.create_schedule(output.op)
# #写缓存
# Cachedconv = s.cache_write(conv, 'local')
# print("cacheconv.op.inputTensor:",Cachedconv.op.input_tensors)
# #TE下降到TIR
# # m = tvm.lower(s, [input_mec,weight_data,weight_indices,weight_indptr,conv], name = 'test_mec')
# #TIR创建sch
# # sch = tvm.tir.Schedule(m)
# # print(sch.mod)
#
# #调度pad
# print("########################padd调度#########################")
# print("pad input tensor:",padd.op.input_tensors)
# # PP = s.cache_read(padd.op.input_tensors[0],"local",[padd])
# print("padd.op.axis:", padd.op.axis)
# p_n,p_c,p_h,p_w = padd.op.axis
# p_n_c = s[padd].fuse(p_n,p_c)
# s[padd].parallel(p_n_c)
# p_h_w = s[padd].fuse(p_h,p_w)
# s[padd].unroll(p_h_w)
# # s[PP].compute_at(s[padd],p_n_c)
# s[padd].set_scope("global")
#
# #调度trans
# #在split、fuse之前读缓存
# print("########################transY调度#########################")
# print("transY input tensor:",transY.op.input_tensors)
# print("transY.op.axis:",transY.op.axis)
# # s[transY].compute_inline()
# # DD = s.cache_read(padd,"local",[transY])
# t_n,t_c,t_h,t_w = s[transY].op.axis
# t_n_c = s[transY].fuse(t_n,t_c)
# s[transY].parallel(t_n_c)
# t_h_w = s[transY].fuse(t_h,t_w)
# s[transY].unroll(t_h_w)
# # s[DD].compute_at(s[transY],transY.op.axis[0])
#
# print("########################tran调度#########################")
# print("tran input tensor:",tran.op.input_tensors)
# print("tran.op.axis:",tran.op.axis)
# s[tran].parallel(tran.op.axis[0])
# s[tran].unroll(tran.op.axis[1])
#
# #调度conv
# print("########################CONV调度#########################")
# print("conv input tensor:",conv.op.input_tensors)
# print("conv.op.axis:", type(conv.op.axis), conv.op.axis)
# print("conv.op.reduce_axis:", type(conv.op.reduce_axis), conv.op.reduce_axis)
# # DD = s.cache_read(tran,"local",[transY])
#
# # n_s, row_s, h_s, w_s = conv.op.axis
# # row_o_s,row_i_s = s[conv].split(row_s,factor = 16)
# # w_o_s,w_i_s = s[conv].split(w_s, factor= 14)
# # h_o_s,h_i_s = s[conv].split(h_s, factor = 14)
# # s[conv].reorder(n_s,row_o_s,h_o_s,w_o_s,row_i_s,h_i_s,w_i_s)
# # row_s = s[conv].fuse(n_s,row_o_s,h_o_s,w_o_s)
# # s[conv].unroll(h_i_s)
# # s[conv].parallel(row_s)
# # s[conv].vectorize(w_i_s)
#
# n_s, row_s, h_s, w_s = conv.op.axis
# h_w_s = s[conv].fuse(h_s, w_s)
# row_s = s[conv].fuse(n_s, row_s)
# # s[conv].reorder(nr_s,h_w_s)
# s[conv].unroll(h_w_s)
# s[conv].parallel(row_s)
# # s[conv].vectorize(w_i_s)
#
# # s[Cachedconv].compute_at(s[conv],row_o_s)
# s[Cachedconv].compute_at(s[conv],row_s)
#
# #调度cacheWrite
# print("########################cacheWrite调度#########################")
# print("cacheConv input tensor:",Cachedconv.op.input_tensors)
# print("cacheConv.op.axis:", type(Cachedconv.op.axis), Cachedconv.op.axis)
# print("cacheConv.op.reduce_axis:", type(Cachedconv.op.reduce_axis), Cachedconv.op.reduce_axis)
#
# # WresIdx,Windptr,Wdata,WindicesRow,WindicesCol,tran
# # WI = s.cache_read(Windptr,"global",[Cachedconv])
# # WD = s.cache_read(Wdata,"global",[Cachedconv])
# # WR = s.cache_read(WindicesRow,"global",[Cachedconv])
# # WC = s.cache_read(WindicesCol,"global",[Cachedconv])
# # RR = s.cache_read(tran,"global",[Cachedconv])
#
# # print("conv.op.axis:",conv.op.axis)
# n,r_s,hh_s,ww_s = Cachedconv.op.axis
# # r_o_s,r_i_s = s[Cachedconv].split(r_s,nparts = 4)
# # print("Cachedconv.op.reduce_axis:",Cachedconv.op.reduce_axis)
# elem_idx_s = Cachedconv.op.reduce_axis[0]
# #拆轴会损害时间
# # e_o_s,e_i_s = s[Cachedconv].split(elem_idx_s,factor=2)
# # s[Cachedconv].reorder(n,r_s,e_o_s,e_i_s,hh_s,ww_s)
# s[Cachedconv].reorder(n,r_s,elem_idx_s,hh_s,ww_s)
# hw = s[Cachedconv].fuse(hh_s,ww_s)
# s[Cachedconv].unroll(hw)
# # s[Cachedconv].vectorize(w_i_s)
#
# # s[WI].compute_at(s[Cachedconv],n)
# # s[WD].compute_at(s[Cachedconv],e_o_s)
# # s[WR].compute_at(s[Cachedconv],n)
# # s[WC].compute_at(s[Cachedconv],n)
# # s[RR].compute_at(s[Cachedconv],r_s)
#
# #调度output
# print("########################output调度#########################")
# print("output input tensor:",output.op.input_tensors)
# print("output.op.axis:", type(output.op.axis), output.op.axis)
# p_c = s[output].fuse(output.op.axis[0],output.op.axis[1])
# s[output].parallel(p_c)
# u_c = s[output].fuse(output.op.axis[2],output.op.axis[3])
# s[output].unroll(u_c)
#
# # 获取 GPU 线程索引
# block_x = te.thread_axis("blockIdx.x")
# s[output].bind(s[output].op.axis[0],block_x)
# print("线程绑定成功")
#
# #保存调度图
# sch = s
# prefix = (str)(H) +'-'+(str)(CO)+'-'+(str)(CI)+'-'+(str)(KH)
# tedd.viz_schedule_tree(sch, dot_file_path="./test_te_tir_schtree/"+prefix+"/scheduletree.dot")
# os.system("dot -Tpng "+"./test_te_tir_schtree/"+prefix+"/scheduletree.dot -o "+"./test_te_tir_schtree/"+prefix+"/scheduletree.png")
# sch = sch.normalize()
# tedd.viz_schedule_tree(sch, dot_file_path="./test_te_tir_schtree/"+prefix+"/scheduletree2.dot")
# os.system("dot -Tpng "+"./test_te_tir_schtree/"+prefix+"/scheduletree2.dot -o "+"./test_te_tir_schtree/"+prefix+"/scheduletree2.png")
#
# #TEsch结束后需要lower
# m = tvm.lower(s, [input_mec,
#                   weight_data,
#                   weight_indices_row,
#                   weight_indices_col,
#                   weight_indptr,
#                   weight_resIdx,
#                   weight_reshapeIdx,
#                   output,
#                   padData,
#                   transData], name = 'test_mec')
# # print(m)
# mod = tvm.build(m, target=target)
#
# #TIRsch结束仍是IRModule
# # mod = tvm.build(m,target=target)
# # mod(x_tvm,weight_data_tvm,weight_indices_tvm,weight_indptr_tvm,y)
# # print(y)
#
# #数据转换
# x_tvm,weight_indptr_tvm,weight_indices_row_tvm,weight_indices_col_tvm,weight_data_tvm,y = (tvm.nd.array(i,device=dev) for i in(x,csr.indptr,weight_row,weight_col,csr.data,out))
# weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))
# resIdx_tvm = tvm.nd.array(resIdx,device=dev)
# reshapeIdx_tvm = tvm.nd.array(reshapeIdx,device=dev)
# pad_out_tvm = tvm.nd.array(pad_out,device=dev)
# trans_out_tvm = tvm.nd.array(trans_out,device=dev)
#
# #结果测试
# print("dataNum:", csr.data.shape)
# print("线程：",para)
# #评估时延
# timer = mod.time_evaluator(mod.entry_name, dev=dev, number=1000)
# result = (
#     timer(x_tvm,
#           weight_data_tvm,
#           weight_indices_row_tvm,
#           weight_indices_col_tvm,
#           weight_indptr_tvm,
#           resIdx_tvm,
#           reshapeIdx_tvm,
#           y,
#           pad_out_tvm,
#           trans_out_tvm
#             ).mean
#         * 1e3
#     )
# print(
#     "our Convolution: %f ms"
#     % result
# )
#
#
# import psutil
#
# # 获取当前进程的信息
# current_process = psutil.Process()
#
# # 获取当前进程的线程数
# thread_count = current_process.num_threads()
#
# print("当前进程的线程数:", thread_count)
#
# # 检查结果
# mod(x_tvm, weight_data_tvm, weight_indices_row_tvm,weight_indices_col_tvm, weight_indptr_tvm,resIdx_tvm,reshapeIdx_tvm,y,pad_out_tvm,trans_out_tvm)
# # print(x[0][0][0][0])
# # print(pad_out_tvm.numpy()[0][0][1][1])
# # print("x 数据:")
# # print(x_tvm.numpy())
# # # print(x_tvm.numpy()[0][0][0])
# # print("pad 数据：")
# # print(pad_out_tvm.numpy())
# # # print(pad_out_tvm.numpy()[0][0][1])
# # print("转换数据：")
# # print(trans_out_tvm.numpy())
# # print(conv_np[0][0][0][0])
# # print(y.numpy()[0][0][0][0])
# # np.testing.assert_allclose(conv_np, y.numpy(), rtol=1e-3)
#
# # data = pad_out_tvm.numpy()
# # n,c,h,w = data.shape
# # out_size = (h - kernel_size) // strides[0] + 1
# # fea = np.zeros((kernel_size,c,h,out_size)).astype('float32')
# # for col_axis in range(kernel_size):
# #     for channel_s in range(c):
# #         for H_s in range(h):
# #             for out_s in range(out_size):
# #                 t = data[0][channel_s][H_s][col_axis+out_s]
# #                 fea[col_axis][channel_s][H_s][out_s] = t
# # ftt = np.zeros((kernel_size*c,h*out_size)).astype('float32')
# # for row in range(kernel_size*c):
# #     for col in range(out_size*h):
# #         # print(row // c, row % c, col // out_size, col % out_size)
# #         t = fea[row // c][row % c][col // out_size][col % out_size]
# #         ftt[row][col] = t
#
# # fea = te.compute(
# #     (kernel_size,C,H,out_size),
# #     lambda col_axis,channel_s,H_s,out_s:
# #         data[0,channel_s,H_s,col_axis+out_s],
# #     name="trans_Y"
# # )
# # return te.compute(
# #     (kernel_size*C,out_size*H),
# #     lambda row,col:
# #         fea[row // (kernel_size*C),row % (kernel_size*C),col//(H*out_size),col%(H*out_size)],
# #     name='trans_reshape'
# # )
#
# # from tvm import relay
# # from tvm.contrib import graph_executor
# # with tvm.transform.PassContext(opt_level=3):
# #     lib = relay.build_module.build(m, params={}, target=target)
# #     # file = open('sp_json.json','w')
# #     # file.write(json.dumps(lib.graph_json))
# #     # file.close()
# #     print("write file over!")
#
# # #执行稀疏模型
# # module = graph_executor.GraphModule(lib["default"](dev))
# # # module = graph_executor.create(lib.graph_json, lib.module, dev)
# # # module.set_input("input",data)
# # # Evaluate
# # print("Evaluate inference time cost...")
# # print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
#
