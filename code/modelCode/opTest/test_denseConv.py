import tvm
import numpy as np
import os
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib import tedd

# 定义简单的卷积层模型
@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

# 设置目标设备和目标编译
target = "llvm -mcpu=broadwell"

# 使用 ResNet-50 中的最后一层
N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.SearchTask(
    func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
)

log_file = "./test_denseConv_tune/conv2d_tuning-"+(str)(H)+"-"+(str)(CO)+"-"+(str)(CI)+"-"+(str)(KH)+".log"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    # runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# 运行自动调优（搜索）
task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
# 终止测试过程
del measure_ctx

tedd.viz_schedule_tree(sch, dot_file_path="./test_denseConv_tune/scheduletree.dot")
os.system("dot -Tpng "+"./test_denseConv_tune/scheduletree.dot -o "+"./test_denseConv_tune/scheduletree.png")
sch = sch.normalize()
tedd.viz_schedule_tree(sch, dot_file_path="./test_denseConv_tune/scheduletree2.dot")
os.system("dot -Tpng "+"./test_denseConv_tune/scheduletree2.dot -o "+"./test_denseConv_tune/scheduletree2.png")
file_mod = open("./test_denseConv_tune/modfile.txt", "w")
file_mod.write(tvm.lower(sch,args,simple_mode=True).astext())
file_mod.close()

func = tvm.build(sch, args, target)

# 检查正确性
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)

dev = tvm.cpu()
data_tvm = tvm.nd.array(data_np, device=dev)
weight_tvm = tvm.nd.array(weight_np, device=dev)
out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
func(data_tvm, weight_tvm, out_tvm)

# 检查结果
# np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# 评估执行时间
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
)

