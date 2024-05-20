import tvm
import numpy as np
import os
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
# from tvm.contrib import tedd
import autotune 

# 定义简单的卷积层模型
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]



def test(input_file=None,
         kernel_file=None,
         padding=(1,1),
         stride=(1,1)):
    in_file = os.path.dirname(input_file)
    target = tvm.target.Target("cuda")
    dev = tvm.cuda()
    log_file = in_file +"/tune/test_conv2d.json"
    x = np.load(input_file)
    N,CI,H,W = x.shape
    kernel = np.load(kernel_file)
    CO,_,KH,KW = kernel.shape
    kernel_size = KH
    # N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 358, 358, 3, 3, (1, 1), (1, 1)
    import utils
    out_size = utils.conv_out_size(H,kernel_size,padding[0],stride[0])
    args=(N, H, W, CO, CI, kernel_size, kernel_size, stride, padding)
    task = auto_scheduler.SearchTask(
        func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, stride, padding), target=target
    )
    if not os.path.exists(log_file):
        print("denseConv没有调优过")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=100,  # change this to 1000 to achieve the best performance
            runner=measure_ctx.runner,
            # runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        # del measure_ctx
        func = tvm.build(sch, args, target)
        # func = autotune.autoTune(conv2d_layer,args,target,log_file,200)
    else:
        sch, args = task.apply_best(log_file)
        # 终止测试过程
        func = tvm.build(sch, args, target)
        # func = autotune.use_tune(conv2d_layer,args,target,log_file)
    data_np = np.load(input_file)
    weight_np = np.load(kernel_file)
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty((N,CO,out_size,out_size), device=dev)
    mod = func
    timer = mod.time_evaluator(mod.entry_name, dev, number=100)
    # print("Convolution: %f ms" % (timer(data_tvm,weight_tvm,bias_tvm,out_tvm).mean * 1e3))
    result = (timer(data_tvm,weight_tvm,out_tvm).mean * 1e3)
    print(
        "dense Convolution: %f ms" 
        % result
    )
    return result

# # 设置目标设备和目标编译
# # target = "llvm -mcpu=broadwell"
# target = "cuda"

# # 使用 ResNet-50 中的最后一层
# # N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
# task = auto_scheduler.SearchTask(
#     func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
# )

# log_file = "/root/user-data/ww/code/modelCode/opTest/test_denseConv_tune/gpu_conv2d_tuning-"+(str)(H)+"-"+(str)(CO)+"-"+(str)(CI)+"-"+(str)(KH)+".log"
# measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
# tune_option = auto_scheduler.TuningOptions(
#     num_measure_trials=100,  # change this to 1000 to achieve the best performance
#     runner=measure_ctx.runner,
#     # runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
#     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#     verbose=2,
# )

# # 运行自动调优（搜索）
# # task.tune(tune_option)
# # 应用最佳 schedule
# sch, args = task.apply_best(log_file)
# # 终止测试过程
# del measure_ctx

# func = tvm.build(sch, args, target)

# # 检查正确性
# data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
# weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
# conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)

# # dev = tvm.cpu()
# dev = tvm.cuda(0)
# data_tvm = tvm.nd.array(data_np, device=dev)
# weight_tvm = tvm.nd.array(weight_np, device=dev)
# out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
# func(data_tvm, weight_tvm, out_tvm)

# # 检查结果
# # np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# # 评估执行时间
# evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
# print(
#     "Execution time of this operator: %.3f ms"
#     % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
# )

