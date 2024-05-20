import onnx
import tvm
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
from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor
import numpy as np


onnx_model = onnx.load("./model/resnet50/model.onnx")
# onnx_model = onnx.load("./model/vgg19/model.onnx")
# onnx_model = onnx.load("./model/yolov8/model.onnx")
#导入onnx稀疏模型

const_mod, params = relay.frontend.from_onnx(onnx_model,{'input':(1,3,224,224)}, freeze_params=True, )
# const_mod, params = relay.frontend.from_onnx(onnx_model,{'images':(1,3,640,640)}, freeze_params=True, )
#推理获取形状
const_mod = relay.transform.InferType()(const_mod)
# print(const_mod)
func = const_mod["main"]

dev = tvm.cpu()
data = tvm.nd.array(np.random.rand(1,3,224,224).astype('float32'))
print("############################")
# print(func)

# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build_module.build(const_mod, params={}, target=tvm.target.Target("llvm -mcpu=core-avx2"))

# graph_const = debug_executor.create(lib.graph_json, lib.module, dev)
# graph_const.set_input('input', data)
# graph_const.run()
# module = graph_executor.GraphModule(lib["default"](dev))
# module.set_input("input",data)


# Evaluate
# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))

#获取op
def _count_nodes(expr):
    """Count the number of occurrences of each operator in the module"""
    ret = {}

    def visit(node):
        if isinstance(node, relay.expr.Call):
            if hasattr(node.op, "name"):
                op_name = node.op.name
                # if(op_name=='nn.conv2d'):
                #     if (node.attrs['data_layout'] == 'NCHW'):
                #         if(node.attrs['kernel_size'][0] == 3):
                #             if(node.attrs['strides'][0] == 1):
                #                 print(op_name)
                #     # if (node.attrs['data_layout'] == 'NCHW' & node.attrs['strides'][0] == 1 & node.attrs['kernel_size'][0] == 1):
                #         # print(ret)
                # if(op_name=='nn.sparse_conv2d'):
                #     print("hahhahaha")
                #     print(type(node.args[0]))
                #     #获得输入大小
                #     print(node.args[0].checked_type.shape)
                #     #获得输出大小
                #     print(node.checked_type.shape)
            else:
                # Some CallNode may not have 'name' such as relay.Function
                return
            ret[op_name] = ret.get(op_name, 0) + 1

    relay.analysis.post_order_visit(expr, visit)
    return ret


ret = _count_nodes(func)
print(ret)
print('******************************************')
# sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert2(
#         const_mod['main'], params, (8, 1), 0.2, "NCHW", 1
#     )

#转换稀疏模型
sparse_func = relay.data_dep_optimization.utils._run_opt_pass(
    const_mod['main'],
    relay.transform._ffi_api.Conv2dToSparseMEC(0.8,'NCHW',1)
)
ret = _count_nodes(sparse_func)
print(ret)
# exit(0)
# sparse_func = relay.data_dep_optimization.utils._run_opt_pass(
#     const_mod['main'],
#     relay.transform._ffi_api.Conv2dToSparse2('NCHW',3,16,1,0.1)
# )

# for i in const_mod.functions.values():
#     j = i.body
#     j.ops
# sparse_func.values()
print("####################################")
# newfunc = relay.data_dep_optimization.utils._run_opt_pass(
#     const_mod["main"],
#     relay.transform._ffi_api.Conv2dToSparse('NCHW',3,16,1,0.4)
# )

# print(sparse_func)

spconst_mod = tvm.ir.IRModule.from_expr(sparse_func)

#获得回喂数据
spconv_data = {}

def fvisit(e):
    if isinstance(e,relay.Call) and e.op.name.startswith("nn.sparseMEC"):
        args_type = [i.checked_type for i in e.args]
        args_type = tuple(('TENSOR', i.shape, i.dtype) for i in args_type)
        weight = tuple(i.data.numpy() for i in e.args[1:])
        spconv_data.setdefault(args_type, []).append(weight)
        # print(weight)
        # spconv_data.setdefault()append(weight)
        # pylint: disable=import-outside-toplevel
        from tvm.auto_scheduler.search_task import (
            register_task_input_buffer,
        )  # lazily import to avoid recursive dependency
        if e.attrs['kernel_size'][0] == 1:
            prefix = "sparseMECconv2d_%d_%d_%d_" % (
            weight[0].shape[0],
            weight[1].shape[0],
            weight[3].shape[0],
            )
            register_task_input_buffer(
                "default",
                prefix+"W_data",
                tvm.runtime.ndarray.array(weight[0]),
                overwrite=True,
                # save_to_file=True
            )
            register_task_input_buffer(
                "default",
                prefix+"W_indices_row",
                tvm.runtime.ndarray.array(weight[1]),
                overwrite=True,
                # save_to_file=True
            )
            register_task_input_buffer(
                "default",
                prefix + "W_indptr",
                tvm.runtime.ndarray.array(weight[3]),
                overwrite=True,
                # save_to_file=True
            )
        else:
            prefix = "sparseMECconv2d_%d_%d_%d_%d_" % (
                weight[0].shape[0],
                weight[1].shape[0],
                weight[2].shape[0],
                weight[3].shape[0],
            )
            register_task_input_buffer(
                "default",
                prefix+"W_data",
                tvm.runtime.ndarray.array(weight[0]),
                overwrite=True,
                # save_to_file=True
            )
            register_task_input_buffer(
                "default",
                prefix+"W_indices_row",
                tvm.runtime.ndarray.array(weight[1]),
                overwrite=True,
                # save_to_file=True
            )
            register_task_input_buffer(
                "default",
                prefix + "W_indices_col",
                tvm.runtime.ndarray.array(weight[2]),
                overwrite=True,
                # save_to_file=True
            )
            register_task_input_buffer(
                "default",
                prefix + "W_indptr",
                tvm.runtime.ndarray.array(weight[3]),
                overwrite=True,
                # save_to_file=True
            )

        # print(prefix)
relay.analysis.post_order_visit(spconst_mod['main'], fvisit)
# print('spconv_data:',spconv_data)

# print("sp conv data:",spconv_data)

#调优原模型
def run_tu(mod):
    from tvm import auto_scheduler
    # target=tvm.target.Target("llvm -mcpu=core-avx2")
    target=tvm.target.Target("llvm")
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], {}, target)
    # auto_scheduler.SearchTask()
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    def run_tuning():
        print("Begin tuning orginal model...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile("./log_file.json")],
        )

        from tvm.topi.sparse.utils import sparse_sketch_rules
        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]
        tuner.tune(tune_option, search_policy=search_policy)
    run_tuning()    

#调优
# from tvm import auto_scheduler
# target=tvm.target.Target("llvm -mcpu=core-avx2")
# print("Extract tasks...")
# tasks, task_weights = auto_scheduler.extract_tasks(spconst_mod["main"], {}, target)
# # auto_scheduler.SearchTask()
# for idx, task in enumerate(tasks):
#     print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
#     print(task.compute_dag)

# def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile("./log_file.json")],
    )

    from tvm.topi.sparse.utils import sparse_sketch_rules
    search_policy = [
        auto_scheduler.SketchPolicy(
            task,
            program_cost_model=auto_scheduler.XGBModel(),
            init_search_callbacks=sparse_sketch_rules(),
        )
        for task in tasks
    ]

    import random
    from tvm import runtime
    # for tsk in tasks:
    #     # tune_option.ref_input = None
    #     if tsk.name.startswith('sparseMEC'):
    #         data,w_data,w_indices_row,w_indices_col,w_indptr,*attrs = tsk.args
    #         weight = random.choice(spconv_data[data,w_data,w_indices_row,w_indices_col,w_indptr])
    #         w_data,w_indices_row,w_indices_col,w_indptr = weight.data,weight.weight_data, weight.weight_indices_row, weight.weight_indices_col, weight.weight_indptr
    #         data = np.random.rand(*data[1]).astype(data[2])
    #         ret = np.zeros_like(data)
    #         # tune_option.ref_input = [ret, w_indptr, w_indices_col, w_indices_row, w_data, data]
    #         tsk.set_inputs({
    #             "w_data": runtime.ndarray.array(weight.data),
    #             "w_indices_row": runtime.ndarray.array(weight.weight_indices_row),
    #             "w_indices_col": runtime.ndarray.array(weight.weight_indices_col),
    #             "w_indptr": runtime.ndarray.array(weight.weight_indptr),
    #         })

    tuner.tune(tune_option, search_policy=search_policy)

# run_tu(const_mod)

# run_tu(spconst_mod)

#原模型运行
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build_module.build(const_mod, params={}, target=tvm.target.Target("llvm -mcpu=core-avx2"))

# module = graph_executor.GraphModule(lib["default"](dev))
# module.set_input("input",data)
# # Evaluate
# print("Evaluate dense inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
# graph_dense = debug_executor.create(lib.graph_json, lib.module, dev)
# graph_dense.set_input('input', data)
# graph_dense.run()

spconst_mod, _ = relay.build_module.optimize(spconst_mod, target="llvm -mcpu=broadwell")
#写算子文件
import json
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build_module.build(spconst_mod, params={}, target=tvm.target.Target("llvm -mcpu=broadwell"))
    # file = open('sp_json.json','w')
    # file.write(json.dumps(lib.graph_json))
    # file.close()
    print("write file over!")

#执行稀疏模型
module = graph_executor.GraphModule(lib["default"](dev))
# module = graph_executor.create(lib.graph_json, lib.module, dev)
module.set_input("input",data)
# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))



# graph_sparse = debug_executor.GraphModuleDebug(lib["default"](dev),dev,lib.graph_json,None)
graph_sparse = debug_executor.create(lib.graph_json, lib.module, dev)
print("true")
graph_sparse.set_input('input', data)
graph_sparse.run()

# cd /home/ww/tvm_20231023/tvm/build
# cmake ..
# make -j8
# cd /home/ww/prune_ww/TVM_MEC/easyUse/code/modelCode
# python ./baselineTest.py














#提取常量表达式
# class ExtractMetaConstants(ExprMutator):
#     # Dirty fix for unknown relay.Const[Meta] occurance.
#     def __init__(self):
#         super().__init__()
#         self.constants = []

#     def visit_constant(self, const: relay.expr.Constant):
#         np_data = const.data.numpy()
#         new_const = relay.const(np_data)
#         if "relay.Constant" in str(const):
#             self.constants.append(np_data)
#         return new_const

#     def extract_constants(self, func):
#         expr = self.visit(func)
#         return expr, self.constants


# for i in const_mod.astext().splitlines()[:100]:
#     print(i) 





# sparseMECconv2d_819_819_819_65_
# sparseMECconv2d_7373_7373_7373_65_
# sparseMECconv2d_3277_3277_3277_257_
# sparseMECconv2d_3277_3277_3277_257_
# sparseMECconv2d_3277_3277_3277_65_
# sparseMECconv2d_5529_5529_5529_65_
# sparseMECconv2d_3277_3277_3277_257_
# sparseMECconv2d_3277_3277_3277_65_
# sparseMECconv2d_5529_5529_5529_65_
# sparseMECconv2d_3277_3277_3277_257_
# sparseMECconv2d_6553_6553_6553_129_
# sparseMECconv2d_13107_13107_13107_513_
# sparseMECconv2d_9830_9830_9830_129_
# sparseMECconv2d_22118_22118_22118_129_
# sparseMECconv2d_9830_9830_9830_513_
# sparseMECconv2d_9830_9830_9830_129_
# sparseMECconv2d_22118_22118_22118_129_
# sparseMECconv2d_13107_13107_13107_513_
# sparseMECconv2d_13107_13107_13107_129_
# sparseMECconv2d_22118_22118_22118_129_
# sparseMECconv2d_9830_9830_9830_513_
# sparseMECconv2d_26214_26214_26214_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_39321_39321_39321_257_
# sparseMECconv2d_58982_58982_58982_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_39321_39321_39321_257_
# sparseMECconv2d_58982_58982_58982_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_39321_39321_39321_257_
# sparseMECconv2d_58982_58982_58982_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_39321_39321_39321_257_
# sparseMECconv2d_58982_58982_58982_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_39321_39321_39321_257_
# sparseMECconv2d_58982_58982_58982_257_
# sparseMECconv2d_39321_39321_39321_1025_
# sparseMECconv2d_104857_104857_104857_513_
# sparseMECconv2d_104857_104857_104857_2049_
# sparseMECconv2d_157286_157286_157286_513_
# sparseMECconv2d_235929_235929_235929_513_
# sparseMECconv2d_104857_104857_104857_2049_
# sparseMECconv2d_157286_157286_157286_513_
# sparseMECconv2d_235929_235929_235929_513_
# sparseMECconv2d_104857_104857_104857_2049_
