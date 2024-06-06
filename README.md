# SPT
Accelerating Convolution of CNN via Sparse Weight Pre-Indexing on TVM

## 使用说明

我们的基础环境是
  TVM
  numpy
  

我们提供了

如果你只想使用稀疏算子的调优测试，而不是整个网络部署，你可以直接运行
type the following from the code directory:
  python test.py

否则你需要对TVM的源码进行更改和重新编译
  提供TVM（）
  
  安装TVM（）地址
  更改相应的源码
  重新编译
  
如果你需要在ARM板卡上部署，你还可以选择在ARM硬件上安装TVM的runtime，使用交叉编译的形式
