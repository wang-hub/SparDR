import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
import os


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    # bias = te.placeholder((1, CO, 1, 1), name="bias")
    in_size = H
    batch, in_channel, in_height, in_width = N,CI,H,W
    num_filter, channel, kernel_h, kernel_w = CO,CI,KH,KW
    out_size = (in_height - KH + 2 * padding) // stride + 1
    out_dtype = 'float32'
    Apad = te.compute(
        (batch, in_channel, in_size + 2 * padding, in_size + 2 * padding,),
        lambda nn, cc ,yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= padding, yy - padding < in_size, xx >= padding, xx - padding < in_size),
            data[nn, cc, yy - padding, xx - padding],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    conv = te.compute(
        (batch, CO, out_size, out_size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy * stride + ry, xx * stride].astype(
                out_dtype
            )
            * kernel[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_nchw",
    )
    # conv = topi.nn.conv2d(data, kernel, stride, padding, dilation=1,layout='NCHW', out_dtype="float32",)
    # out = topi.nn.relu(conv + bias)
    # return [data, kernel, bias, out]
    return [data, kernel, conv]


def test(input_file=None,
         kernel_file=None,
         padding=1,
         stride=1):
    in_file = os.path.dirname(input_file)
    # target = tvm.target.Target("llvm")
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    dev = tvm.cpu()
    # log_file = in_file +"/tune/test_conv2d.json"
    log_file = in_file +"/tune/test_conv2d-avx2.json"
    # log_file = in_file +"/tune/test_conv2d-avx512.json"
    x = np.load(input_file)
    N,CI,H,W = x.shape
    kernel = np.load(kernel_file)
    CO,_,kernel_size,_ = kernel.shape
    import utils
    out_size = utils.conv_out_size(H,kernel_size,padding,stride)
    args=(N, H, W, CO, CI, kernel_size, kernel_size, stride, padding)
    import autotune 
    if not os.path.exists(log_file):
        print("denseConv没有调优过")
        func = autotune.autoTune(conv2d_layer,args,target,log_file,200)
    else:
        func = autotune.use_tune(conv2d_layer,args,target,log_file)
    data_np = np.load(input_file)
    weight_np = np.load(kernel_file)
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
    mod = func
    timer = mod.time_evaluator(mod.entry_name, dev, number=10)
    # print("Convolution: %f ms" % (timer(data_tvm,weight_tvm,bias_tvm,out_tvm).mean * 1e3))
    result = (timer(data_tvm,weight_tvm,out_tvm).mean * 1e3)
    print(
        "dense Convolution: %f ms" 
        % result
    )
    return result
