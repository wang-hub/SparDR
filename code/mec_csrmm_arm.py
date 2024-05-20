import numpy as np
import tvm
from collections import namedtuple
from tvm.topi.utils import get_const_int, get_const_tuple, simplify, tag
from tvm.topi.nn.pad import pad
from tvm.topi.nn.utils import get_pad_tuple
from tvm import auto_scheduler
import utils
from tvm.topi.testing import conv2d_nchw_python
import os
import numpy as np
import tvm.testing
from tvm import te, auto_scheduler, runtime


#处理kernelSize ≠ 1的情况，特征图有重用情况
def mec_csrmm(data,weight_data,weight_row,weight_col,weight_indptr,batch,out_size):
    #data:(batch, new_axis, out_size)
    assert batch==1
    #只处理batch=1的情况
    # data_row,data_col = get_const_tuple(data.shape)
    out_channel = get_const_int(weight_indptr.shape[0]) - 1
    oshape = (batch,out_channel,out_size, out_size)
    # idxd = tvm.tir.indexdiv
    # idxm = tvm.tir.indexmod
    def f(n,row,h,w):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        
        # idx = weight_indices[elem]
        #这里可能是影响速度的地方
        #可以预处理
        # d_row = idxm(idx, data_row)
        d_row = weight_row[elem]
        d_col = weight_col[elem]
        # d_col = idxd(idx, data_row) * out_size

        weight_val = data[d_row, d_col + h*out_size + w]
        return te.sum(a_val * weight_val, axis=elem_idx,)

    return te.compute(oshape, f, tag="mec_csrmm")


#处理kernelSize = 1 的情况
def mec_csrmm_k1(data,weight_data,weight_indices,weight_indptr,batch,out_size):
    #data:(batch, new_axis, out_size)
    assert batch==1
    #只处理batch=1的情况
    # data_row,data_col = get_const_tuple(data.shape)
    out_channel = get_const_int(weight_indptr.shape[0]) - 1
    oshape = (batch,out_channel,out_size, out_size)
    # idxd = tvm.tir.indexdiv
    # idxm = tvm.tir.indexmod
    def f(n,row,h,w):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = te.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        
        idx = weight_indices[elem]
        #这里可能是影响速度的地方
        #可以预处理
        # d_row = idxm(idx, data_row)
        # d_col = idxd(idx, data_row) * out_size

        weight_val = data[idx, h*out_size + w]
        return te.sum(a_val * weight_val, axis=elem_idx,)

    return te.compute(oshape, f, tag="mec_csrmm")


#调试csrmm部分
#kernel尺寸为1
@auto_scheduler.register_workload
def test_sp_csrmm_k1(data_shape,weight_data_shape,weight_indices_shape,weight_indptr_shape,batch,out_size):
    input_mec = te.placeholder(data_shape,dtype='float32')
    weight_data = te.placeholder(weight_data_shape,dtype='float32',name='w_data')
    weight_indices = te.placeholder(weight_indices_shape,dtype='int32',name='w_indices')
    weight_indptr = te.placeholder(weight_indptr_shape,dtype='int32',name='w_indptr')
    conv = mec_csrmm_k1(input_mec,weight_data,weight_indices,weight_indptr,batch,out_size)
    return [input_mec, weight_data, weight_indices, weight_indptr, conv]
    
#调试csrmm部分
#kernel尺寸不为1
@auto_scheduler.register_workload
def test_sp_csrmm(data_shape,weight_data_shape,weight_row_shape,weight_col_shape,weight_indptr_shape,batch,out_size):
    input_mec = te.placeholder(data_shape,dtype='float32')
    weight_data = te.placeholder(weight_data_shape,dtype='float32',name='w_data')
    weight_row= te.placeholder(weight_row_shape,dtype='int32',name='w_row')
    weight_col= te.placeholder(weight_col_shape,dtype='int32',name='w_col')
    weight_indptr = te.placeholder(weight_indptr_shape,dtype='int32',name='w_indptr')
    conv = mec_csrmm(input_mec,weight_data,weight_row,weight_col,weight_indptr,batch,out_size)
    return [input_mec, weight_data, weight_row, weight_col, weight_indptr, conv]

#返回运行时间
def conv(input_file=None,
         kernel_file=None,
         padding=1,
         stride=1,
         kISone = False):
    x = np.load(input_file)
    # print("x shape:",x.shape)
    N,CI,H,_ = x.shape

    kernel = np.load(kernel_file)
    CO,ki,kernel_size,_ = kernel.shape
    #做个简单判输入是否正确
    assert ki==CI
    csr = utils.deal_sp_kernel(kernel)

    import featureTrans
    x = featureTrans.use_feature_trans(input_file,kernel_size,padding,stride)

    # target = tvm.target.Target("llvm")
    target = tvm.target.Target("llvm -mcpu=core-avx2")
    # target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    dev = tvm.cpu()
    out_size = utils.conv_out_size(H,kernel_size,padding,stride)
    out = tvm.nd.empty((N,CO,out_size,out_size), device=dev)

    #预处理
    x_row,_ = x.shape
    weight_row = csr.indices % x_row
    weight_col = csr.indices // x_row * out_size
    # print("x_row:",x_row)
    # print("csr_dicies:",csr.indices)
    # print("weight_row:",weight_row)
    # print("weight_col:",weight_col)

    x_tvm,weight_indptr,weight_indices,weight_data,y = (tvm.nd.array(i,device=dev) for i in(x,csr.indptr,csr.indices,csr.data,out))
    weight_row,weight_col = (tvm.nd.array(i,device=dev) for i in(weight_row,weight_col))
   
    in_file = os.path.dirname(input_file)
    # log_file = in_file + "/tune/tune.json"
    # log_file = in_file + "/tune/tune-avx512.json"
    # log_file = in_file + "/tune/tune-avx2.json"
    log_file = in_file + "/tune/tune-arm.json"
    #卷积核为1的情况
    if (kISone):
        task = tvm.auto_scheduler.SearchTask(
            func=test_sp_csrmm_k1,
            args=(x.shape,csr.data.shape,csr.indices.shape,csr.indptr.shape,N,out_size),
            target=target,
            task_inputs={
                "w_data": runtime.ndarray.array(csr.data),
                "w_indices": runtime.ndarray.array(csr.indices),
                "w_indptr": runtime.ndarray.array(csr.indptr),
            },
        )
        # print("Computational DAG:")
        # print(task.compute_dag)
        def meet_condition_func(search_policy, state, stage_id):
            state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
            if state.stages[stage_id].op.tag in [
                "mec_csrmm",
            ]:
                print("apply this " + str(state.stages[stage_id].op.tag) + " sketchRule")
                return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
            else:
                return auto_scheduler.PreloadCustomSketchRule.PASS


        def apply_func(search_policy, state, stage_id):
            ret = []
            s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
            output = s0.stages[stage_id].op
            # print("input:",output.input_tensors)
            assert output.tag == "mec_csrmm"
            # print("mec_csrmm iters:")
            # print(s0[output].iters)

            n, c, h, w, elem_idx = s0[output].iters
            # hw1, hw2, hw3 = s0.split(output, , [None, None])
            # s0.reorder(output, [n, hw1, c, hw2 ,elem_idx, h, hw3])
            # s0.unroll(output, h)
            # s0.vectorize(output, hw3)

            # i0, i1, i2 = s0.split(output, h, [None, None])
            # j0, j1 = s0.split(output, c, [None])
            # s0.reorder(output, [n, i0, j0, i1 , j1, elem_idx, i2, w])

            # h0, h1, h2 = s0.split(output, h, [None, None])
            # w0, w1, w2 = s0.split(output, w, [None, None])
            
            # c0, c1 = s0.split(output, c, [None])
            # s0.reorder(output, [n, c0, c1, elem_idx, h, w])
            # m = s0.fuse(output,[n,c0])
            # s0.reorder(output, [m, c1, elem_idx, h, w])

            # s0.parallel(output,m)
            # s0.unroll(output, h)
            # s0.vectorize(output, w)

            i_0, i_1, i_2 = s0.split(output, n, [None, None])
            h_0, h_1, h_2 = s0.split(output, h, [None, None])
            w_0, w_1, w_2 = s0.split(output, w, [None, None])  # pylint: disable=invalid-name
            j_0, j_1 = s0.split(output, c, [None])
            s0.reorder(
                        output,
                        [i_0, j_0, h_0, w_0, i_1, j_1, h_1, w_1, elem_idx, i_2, h_2, w_2],
                    )


            # infer_bound_from_state（状态）
            # 推断并填充状态的所有迭代器的边界
            ret.append([s0.state_object, stage_id - 1])

            return ret  


        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            # early_stopping=10,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            # verbose=2,
        )

        search_policy = auto_scheduler.SketchPolicy(
            task,
            program_cost_model=auto_scheduler.XGBModel(),
            init_search_callbacks=[
                auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "mecCsrmmConv")
            ],
        )
        # Run auto-tuning (search)
        if not os.path.exists(log_file):
            print("ourMethod没有调优过")
            task.tune(tune_option, search_policy)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)
        # print("Lowered TIR:")
        # print(tvm.lower(sch, args, simple_mode=True))
        func = tvm.build(sch, args, target)
        # print("mod is ok")
        timer = func.time_evaluator(func.entry_name, dev=dev, number=10)
        result = (
            timer(x_tvm, weight_data, weight_indices, weight_indptr, y
                    ).mean
                * 1e3
            )
        print(
            "our Convolution: %f ms" 
            % result
        )
        return result
    else:
        task = tvm.auto_scheduler.SearchTask(
            func=test_sp_csrmm,
            args=(x.shape,csr.data.shape,weight_row.shape,weight_col.shape,csr.indptr.shape,N,out_size),
            target=target,
            task_inputs={
                "w_data": runtime.ndarray.array(csr.data),
                "w_row": runtime.ndarray.array(weight_row),
                "w_col": runtime.ndarray.array(weight_col),
                "w_indptr": runtime.ndarray.array(csr.indptr),
            },
        )
        # print("Computational DAG:")
        # print(task.compute_dag)
        def meet_condition_func(search_policy, state, stage_id):
            state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
            if state.stages[stage_id].op.tag in [
                "mec_csrmm",
            ]:
                print("apply this " + str(state.stages[stage_id].op.tag) + " sketchRule")
                return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
            else:
                return auto_scheduler.PreloadCustomSketchRule.PASS


        def apply_func(search_policy, state, stage_id):
            ret = []
            s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
            output = s0.stages[stage_id].op
            # print("input:",output.input_tensors)
            assert output.tag == "mec_csrmm"
            # print("mec_csrmm iters:")
            # print(s0[output].iters)

            n, c, h, w, elem_idx = s0[output].iters
            # hw1, hw2, hw3 = s0.split(output, , [None, None])
            # s0.reorder(output, [n, hw1, c, hw2 ,elem_idx, h, hw3])
            # s0.unroll(output, h)
            # s0.vectorize(output, hw3)

            # i0, i1, i2 = s0.split(output, h, [None, None])
            # j0, j1 = s0.split(output, c, [None])
            # s0.reorder(output, [n, i0, j0, i1 , j1, elem_idx, i2, w])

            # h0, h1, h2 = s0.split(output, h, [None, None])
            # w0, w1, w2 = s0.split(output, w, [None, None])
            
            # c0, c1 = s0.split(output, c, [None])
            # s0.reorder(output, [n, c0, c1, elem_idx, h, w])
            # m = s0.fuse(output,[n,c0])
            # s0.reorder(output, [m, c1, elem_idx, h, w])

            # s0.parallel(output,m)
            # s0.unroll(output, h)
            # s0.vectorize(output, w)

            i_0, i_1, i_2 = s0.split(output, n, [None, None])
            h_0, h_1, h_2 = s0.split(output, h, [None, None])
            w_0, w_1, w_2 = s0.split(output, w, [None, None])  # pylint: disable=invalid-name
            j_0, j_1 = s0.split(output, c, [None])
            s0.reorder(
                        output,
                        [i_0, j_0, h_0, w_0, i_1, j_1, h_1, w_1, elem_idx, i_2, h_2, w_2],
                    )


            # infer_bound_from_state（状态）
            # 推断并填充状态的所有迭代器的边界
            ret.append([s0.state_object, stage_id - 1])

            return ret

        

        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,
            # early_stopping=10,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            # verbose=2,
        )

        search_policy = auto_scheduler.SketchPolicy(
            task,
            program_cost_model=auto_scheduler.XGBModel(),
            init_search_callbacks=[
                auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "mecCsrmmConv")
            ],
        )
        # Run auto-tuning (search)
        if not os.path.exists(log_file):
            print("ourMethod没有调优过")
            task.tune(tune_option, search_policy)
        # Apply the best schedule
        sch, args = task.apply_best(log_file)
        # print("Lowered TIR:")
        # print(tvm.lower(sch, args, simple_mode=True))
        func = tvm.build(sch, args, target)
        # print("mod is ok")
        timer = func.time_evaluator(func.entry_name, dev=dev, number=10)
        result = (
            timer(x_tvm, weight_data, weight_row, weight_col, weight_indptr, y
                    ).mean
                * 1e3
            )
        print(
            "our Convolution: %f ms" 
            % result
        )
        return result

