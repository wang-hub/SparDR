import numpy as np
import tvm
from collections import namedtuple
from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
import utils
from tvm.topi.testing import conv2d_nchw_python
import os
import numpy as np
import tvm.testing
from tvm import te


def mec_trans(input,kernel_size,padding,stride):
    batch,in_channel,in_size,_ = get_const_tuple(input.shape)
    out_size = (in_size - kernel_size + 2*padding) // stride + 1
    oshape = (batch*kernel_size*in_channel, out_size*(in_size+2*padding))
    ki = kernel_size * in_channel
    def trans(row,col):
        n = row // ki
        c = row % in_channel
        h = col // out_size
        off_size = row % ki //in_channel
        w = col % out_size + off_size
        # h = h - padding
        # w = w - padding
        return tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < in_size, w >= padding, w - padding < in_size),
            input[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, "float32"),
        )
    # print("mec_oshape:",oshape)
    return te.compute(oshape,trans,name='mec_new_trans')


def use_feature_trans(input_file=None,
                  kernel_size=3,
                  padding=1,
                  stride=1):
    #返回转换后的特征图
    #大小为(N*kernel_size*CI,out_size*(H+2*padding))
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    x = np.load(input_file)
    x = tvm.nd.array(x,device=dev)
    N,CI,H,W = x.shape
    out_size = utils.conv_out_size(H,kernel_size,padding,stride)

    # 调试mec_trans
    def test_mec(ishape,kernel_size,padding,stride):
        input_mec = te.placeholder(ishape,dtype='float32')
        op_mec = mec_trans(input_mec,kernel_size,padding,stride)
        return [input_mec,op_mec]


    A,B = test_mec(x.shape,kernel_size,padding,stride) 
    s = te.create_schedule(B.op)
    # print(tvm.lower(s, [A, B], simple_mode=True))
    out_mec = tvm.nd.empty((N*kernel_size*CI,out_size*(H+2*padding)), device=dev)
    mod = tvm.build(s,[A, B],target)
    timer = mod.time_evaluator(mod.entry_name,dev,number=10)
    timer(x, out_mec)
    # print(
    #     "feature_trans: %f ms" 
    #     % (
    #     timer(x, out_mec
    #             ).mean
    #         * 1e3
    #     )
    # )
    #save mec output
    in_file = os.path.dirname(input_file)
    mec_file = in_file + "/input_trans.npy"
    np.save(mec_file,out_mec.numpy())
    # print("save path:" + mec_file)
    # print("save successfully!")
    return out_mec.numpy()