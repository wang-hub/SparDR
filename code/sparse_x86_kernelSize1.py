import tvm
from tvm import topi
import os
import numpy as np
import tvm
import tvm.testing
from tvm import te, auto_scheduler, runtime, topi, tir
from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
from tvm.topi.nn.utils import  get_pad_tuple
from tvm.topi.sparse.utils import random_bsr_matrix
import utils


@auto_scheduler.register_workload
def test_im2col(N,CI,H,W, w_data_shape, w_indices_shape, w_indptr_shape, dtype,stride,padding):
    X = te.placeholder((N,CI,H,W), dtype=dtype,name='input')
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype,name='w_data')
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32",name='w_indices')
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32",name='w_indptr')
    out = topi.nn.sparse_conv2d(X,W_data,W_indices,W_indptr,layout='NCHW',kernel_size=1)
    # out = topi.nn.sparse_dense(B, W_data, W_indices, W_indptr)
    return [X, W_data, W_indices, W_indptr, out]


def test(input_file=None,
         kernel_file=None,
         padding=0,
         stride=1):
    #该方法没有padding
    # assert padding==0
    if(padding != 0):
        return -1
    in_file = os.path.dirname(input_file)
    # target = tvm.target.Target("llvm")
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    dev = tvm.cpu()
    x = np.load(input_file)
    N,CI,H,W = x.shape
    kernel = np.load(kernel_file)
    CO,_,kernel_size,_ = kernel.shape
    #只支持kernel大小为1
    # assert kernel_size==1
    if(kernel_size != 1):
        print("kernel_size is not 1, skip sparse_x86")
        return 0
    bsr = utils.deal_sp_kernel_bsr(kernel)

    args = (N,CI,H,W,bsr.data.shape,bsr.indices.shape,bsr.indptr.shape,'float32',stride,padding)
    task = tvm.auto_scheduler.SearchTask(
        func=test_im2col,
        args=args,
        target=target,
        task_inputs={
            "w_data": runtime.ndarray.array(bsr.data),
            "w_indices": runtime.ndarray.array(bsr.indices),
            "w_indptr": runtime.ndarray.array(bsr.indptr),
            "input":runtime.ndarray.array(x),
        },
    )

    print("sparse_x86:Computational DAG:")
    print(task.compute_dag)
    # log_file = in_file + "/tune/x86tune.json"
    log_file = in_file + "/tune/x86tune-avx2.json"
    # log_file = in_file + "/tune/x86tune-avx512.json"

    def run_tuning():
        # print("Begin tuning...")
        # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            # verbose=2,
        )
        from tvm.topi.sparse.utils import sparse_sketch_rules
        search_policy = auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
        
        task.tune(tune_option, search_policy=search_policy)


    if not os.path.exists(log_file):
        print("sparseX86没有调优过")
        run_tuning()
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target)
    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))
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
        "sparse_x86 Convolution: %f ms" 
        % result
    )
    
    return result



