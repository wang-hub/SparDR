import tvm
from tvm.script.parser import ir_module
from tvm.ir.module import IRModule
from tvm import tir as T
import numpy as np



@ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(8):
            # block 是计算的抽象。
            with T.block("B"):
                # 定义 spatial block 迭代器，并将其绑定到值 i。
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
print(type(ir_module))
ir_module.show()