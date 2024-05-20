import numpy as np
import tvm
from tvm import te, runtime
import scipy.sparse as ss
from scipy.stats import truncnorm
from tvm import auto_scheduler
from tvm import topi
# from tvm.contrib import tedd


def autoTune(func,args,target = tvm.target.Target("cuda"),log_file=None,num_measure_trials=100):
    # print("target:",target.list_kinds())
    task = auto_scheduler.SearchTask(
        func = func,
        args = args,
        target = target,
    )
    print("denseConv:Computational DAG:")
    print(task.compute_dag)
    #CPU
    runner = auto_scheduler.LocalRunner(
        timeout=10,
        number=3,
        repeat=3,
        min_repeat_ms=300,
        enable_cpu_cache_flush=True,
    )
    #GPU
    # measure_ctx =  auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=num_measure_trials,
        # verbose=2,
        # runner=measure_ctx.runner,
        runner= runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        # early_stopping=10,
    )
    search_policy = auto_scheduler.SketchPolicy(
        task=task,
        # program_cost_model=auto_scheduler.RandomModel(),
        program_cost_model=auto_scheduler.XGBModel(),
    )
    task.tune(
        tune_option,
        search_policy,
    )
    # del measure_ctx
    # task.print_best(log_file)
    sch,args = task.apply_best(log_file)
    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))
    mod = tvm.build(sch,args,target)
    return mod

import os
def use_tune(func,args,target = tvm.target.Target("cuda"),log_file="autotune_log_file.json"):
    task = auto_scheduler.SearchTask(
        func = func,
        args = args,
        target = target,
    )
    # print("Computational DAG:")
    # print(task.compute_dag)
    # print("best result:")
    # task.print_best(log_file)
    sch,args = task.apply_best(log_file)
    # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))
    # tedd_file = os.path.dirname(log_file)
    # print("tedd_file:",tedd_file)
    # tedd.viz_dataflow_graph(sch, dot_file_path=tedd_file + "/dfg.dot")
    # os.system("dot -Tpng "+tedd_file + "/dfg.dot -o "+tedd_file+"/dfg.png")
    # tedd.viz_schedule_tree(sch, dot_file_path=tedd_file + "/scheduletree.dot")
    # os.system("dot -Tpng "+tedd_file + "/scheduletree.dot -o "+tedd_file+"/scheduletree.png")
    # sch.normalize()
    # tedd.viz_schedule_tree(sch, dot_file_path=tedd_file + "/scheduletree2.dot")
    # os.system("dot -Tpng "+tedd_file + "/scheduletree2.dot -o "+tedd_file+"/scheduletree2.png")
    # # print("Lowered TIR:")
    # print(tvm.lower(sch, args, simple_mode=True))
    mod = tvm.build(sch,args,target)
    return mod