import numpy as np
import get_data
from scipy import sparse
import tvm
from tvm import relay
from tvm.relay import transform


def print_add_mark(a, mark='#'):
    print("##############################begin###################################")
    print(a)
    print("###############################end#######################################")


#获取opnum
def get_op_nums(mod):
    """Count the number of occurrences of each operator in the module"""
    # 用法样例：ret = _count_nodes(func)
    ret = {}
    print('*************get op num******************')

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

    relay.analysis.post_order_visit(mod['main'], visit)
    print(ret)
    print('******************************************')
    return ret
    

#遍历relay获得op参数
#提取所有卷积算子中
#input shape
#kernel shape
#kernel padding
#kernel strides
def get_op_para(mod,params):
    #先推理模型
    mod = transform.InferType()(mod)
    res = []
    def fvisit(e):
        if isinstance(e,relay.Call):
            # print("op name:",e.op.name)
            # if e.op.name.startswith("nn.conv"):
            if e.op.name=='nn.conv2d':
                tem = []
                input_shape = (list)(e.args[0].checked_type.shape)
                # print(type(e.args[0].checked_type.shape))
                #YOLOv8不定长问题放在外面处理
                # print(type(input_shape[0]))
                # input_shape[0] = 1
                # for i in input_shape[1:]:
                #     if not isinstance(i,tvm.tir.expr.IntImm):
                #     # if not i._dtype_is_int:
                #         print(input_shape)
                #         input()
                kernel_shape = list(e.args[1].checked_type.shape)

                # print(e.args[0])
                # print(e.args[1])
                # print(e.name_hint)
                # 获取卷积操作节点的名称
                kernel_name = e.args[1].name_hint if hasattr(e.args[1], "name_hint") else "kernel_name"
                # input()
                # kernel_name = e.args[1].name_hint
                # weight = params[kernel_name]
                weight = e.args[1].data
                # print(type(weight.numpy()))
                # input()
                attrs = e.attrs

                tem.append(input_shape)
                tem.append(kernel_shape)
                tem.append(attrs['padding'])
                tem.append(attrs['strides'])
                tem.append(weight.numpy())
                tem.append(kernel_name)

                res.append(tem)
                # input()
                # print("attr")
                # attrs = e.attrs
                # print_add_mark(attrs['strides'])
                # print_add_mark(attrs['padding'])
                # print("特征图大小")
                # input_shape = e.args[0].checked_type
                # print_add_mark(input_shape)
                # print("卷积核参数：")
                # kernel_shape = e.args[1].checked_type
                # print_add_mark(kernel_shape)
                # print("卷积核")
                # kernel = e.args[1]
                # print_add_mark(kernel)
                # print("卷积核权重")
                # kernel_name = e.args[1].name_hint
                # print_add_mark(kernel_name)
                # weight = params[kernel_name]
                # print_add_mark(weight)
    relay.analysis.post_order_visit(mod['main'], fvisit)
    # print_add_mark(res)
    return res


def conv_out_size(n,k,p,s):
    return (n-k+2*p)//s+1


def deal_data(kernel,nonzeros=0):
    in_channel,out_channel,kh,kw = kernel.shape
    #计算出ker_idx和ker_val
    ker_idx = []
    ker_val = []
    idx = 0
    nonzeros = 0
    for ci in range(in_channel):
        for i in range(kh):
            for j in range(kw):
                temp = 0
                for co in range(out_channel):
                    temp = temp + kernel[ci,co,i,j]
                if(temp!=0):
                    nonzeros = nonzeros + 1
                    ker_idx.append(idx)
                    ker_val.append(temp)
                idx = idx + 1
    # 计算出ker_sig_wei
    ker_sig_wei = []
    for co in range(out_channel):
        t_list = []
        for n in range(nonzeros):
            index = ker_idx[n]
            c = index // (kh*kw)
            index = index % (kh*kw)
            h = index//kw
            index = index%kw
            w = index
            t_list.append(kernel[c,co,h,w]/ker_val[n])
        ker_sig_wei.append(t_list)
    # ker_sig_wei
    return [nonzeros,
            np.array(ker_idx,dtype='int32'),
            np.array(ker_val,dtype='float32'),
            np.array(ker_sig_wei,dtype='float32')]

def deal_sp_kernel(kernel_file,dtype='float32'):
    kernel = np.load(kernel_file)
    out_channel,in_channel,kh,kw = kernel.shape
    ker_sig_wei = []
    for co in range(out_channel):
        idx = 0
        t_list = []
        for h in range(kh):
            for w in range(kw):
                for c in range(in_channel):
                    t_list.append(kernel[co,c,h,w])
                    idx = idx + 1
        ker_sig_wei.append(t_list)
    wei = np.array(ker_sig_wei)
    return sparse.csc_matrix(wei)


def deal_sp_kernel(kernel):
    out_channel,in_channel,kh,kw = kernel.shape
    ker_sig_wei = []
    for co in range(out_channel):
        idx = 0
        t_list = []
        for h in range(kh):
            for w in range(kw):
                for c in range(in_channel):
                    t_list.append(kernel[co,c,h,w])
                    idx = idx + 1
        ker_sig_wei.append(t_list)
    wei = np.array(ker_sig_wei)
    # print(wei)
    # print(wei.shape)
    return sparse.csr_matrix(wei)

#test
# 2,3,2,3
kernel = [[[[1,0,1],
          [0,0,1]],
         [[1,0,1],
          [0,0,1]]],
         [[[1,0,1],
          [0,0,1]],
         [[1,0,1],
          [0,0,1]]],
         [[[1,0,1],
          [0,0,1]],
         [[1,0,1],
          [0,0,1]]],
         [[[1,0,1],
          [0,0,1]],
         [[1,0,1],
          [0,0,1]]]]

def tongdaozhuanhuan():
    kernel = np.array(kernel)
    print(kernel)
    print(kernel.shape)
    sker = deal_sp_kernel(kernel)
    # print(sker)
    # k = kernel.reshape((-1, kernel.shape[-1])).T

    ##通道转换
    # img = np.transpose(img, (0, 2, 3, 1))
    
    # img_hwc = np.transpose(img_chw, (1, 2, 0))
    
    # image = np.expand_dims(image, axis=0)

    k = np.transpose(kernel,(0,2,3,1)).reshape((kernel.shape[0], -1))
    print(k.shape)
    print(k)
    sk = sparse.csr_matrix(k)
    # print(sk)




def deal_sp_kernel_bsr(kernel):
    out_channel,in_channel,kh,kw = kernel.shape
    ker_sig_wei = []
    for co in range(out_channel):
        idx = 0
        t_list = []
        for h in range(kh):
            for w in range(kw):
                for c in range(in_channel):
                    t_list.append(kernel[co,c,h,w])
                    idx = idx + 1
        ker_sig_wei.append(t_list)
    wei = np.array(ker_sig_wei)
    return sparse.bsr_matrix(wei)

def deal_sp_dense_data(kernel):
    #co,ci,kh,kw
    #按通道优先排列
    out_channel,in_channel,kh,kw = kernel.shape
    #计算出ker_val
    ker_val = []
    for i in range(kh):
        for j in range(kw):
            for ci in range(in_channel):
                temp = 0
                for co in range(out_channel):
                    temp = temp + kernel[co,ci,i,j]
                ker_val.append(temp)
    # 计算出ker_sig_wei
    ker_sig_wei = []
    for co in range(out_channel):
        idx = 0
        t_list = []
        for h in range(kh):
            for w in range(kw):
                for c in range(in_channel):
                    if(ker_val[idx]!=0):
                        t_list.append(kernel[co,c,h,w]/ker_val[idx])
                    else:
                        t_list.append(0)
                    idx = idx + 1
        ker_sig_wei.append(t_list)
    # ker_sig_wei
    return [
            np.array(ker_val,dtype='float32'),
            np.array(ker_sig_wei,dtype='float32')]


def cal_ker_val(kernel):
    out_channel,in_channel,kh,kw = kernel.shape
    #计算出ker_val
    ker_val = []
    for i in range(kh):
        for j in range(kw):
            for ci in range(in_channel):
                temp = 0
                for co in range(out_channel):
                    temp = temp + kernel[co,ci,i,j]
                ker_val.append(temp)
    return np.array(ker_val,dtype='float32')


def cal_ker_wei(kernel,ker_val):
    out_channel,in_channel,kh,kw = kernel.shape
    ker_sig_wei = []
    for co in range(out_channel):
        idx = 0
        t_list = []
        for h in range(kh):
            for w in range(kw):
                for c in range(in_channel):
                    if(ker_val[idx]!=0):
                        t_list.append(kernel[co,c,h,w]/ker_val[idx])
                    else:
                        t_list.append(0)
                    idx = idx + 1
        ker_sig_wei.append(t_list)
    return  np.array(ker_sig_wei,dtype='float32')


def get_deal_data(co,ci,kh,kw,sparity=0.2):
    # ci,co,kh,kw = 32,32,3,3
    kernel = get_data.get_conv_sparse_data(co,ci,kh,kw,sparity)
    kv,wei = deal_sp_dense_data(kernel)
    return [kv,wei]

def get_deal_data_kernel(kernel_file=None):
    # ci,co,kh,kw = 32,32,3,3
    # kernel = get_data.get_conv_sparse_data(co,ci,kh,kw,sparity)
    if kernel_file==None:
        kernel_file = "./conv_test/conv_test_256_358/data/tensor.npy"
    kernel = np.load(kernel_file)
    kv,wei = deal_sp_dense_data(kernel)
    return [kernel,kv,wei]

def np_csr_ker(kernel_file):
    if kernel_file==None:
        kernel_file = "./conv_test/conv_test_256_358/data/tensor.npy"
    kernel = np.load(kernel_file)
    Csc = sparse.csc_matrix(kernel)
    return Csc

def np_csr_ker_wei(kernel_wei_file=None):
    if kernel_wei_file==None:
        kernel_wei_file = "./conv_test/conv_test_256_358/data/kernel_wei_file.npy"
        print("使用默认文件:",kernel_wei_file)
    kernel_wei = np.load(kernel_wei_file)
    print(kernel_wei.shape)
    Csr = sparse.csr_matrix(kernel_wei)
    return Csr


def get_input_ker_val_wei():
    input_file = "./conv_test/conv_test_256_358/data/input_file.npy"
    kernel_file = "./conv_test/conv_test_256_358/data/kernel_file.npy"
    kernel_val_file = "./conv_test/conv_test_256_358/data/kernel_val_file.npy"
    kernel_wei_file = "./conv_test/conv_test_256_358/data/kernel_wei_file.npy"
    input = np.load(input_file)
    kernel = np.load(kernel_file)
    k_val = np.load(kernel_val_file)
    k_wei = np.load(kernel_wei_file)
    return [input,kernel,k_val,k_wei]


def test_trans(input):
    batch,in_channel,in_size,in_size = input.shape



# kernel_file = "./conv_test/conv_test_89_3/data/tensor.npy"
# kernel = np.load(kernel_file)
# csr = deal_sp_kernel(kernel)

#test
# kernel = np.array(
#             [[
#                 [[1,0,0],
#                  [0,1,0],
#                  [0,0,0]],
#                 [[1,0,0],
#                  [0,0,0],
#                  [0,0,1]],
#                 [[1,0,0],
#                  [0,0,0],
#                  [0,0,0]]
#                  ],
#              [
#                  [[1,0,0],
#                   [0,0,2],
#                   [0,0,0]
#                   ],
#                 [[3,0,0],
#                  [0,0,0],
#                  [0,4,0]],
#                 [[5,0,0],
#                  [0,0,0],
#                  [6,0,0]]]]
# )
# csr = deal_sp_kernel(kernel)

# print(kernel.shape)
# print("kernel:",kernel)
# # kv,kw = deal_sp_dense_data(kernel)
# # print("kv:",kv)
# # print("ks:",kw)
# # print("kw shape:", kw.shape)
# # csc = sparse.csr_matrix(kw)
# # print("csc:",csc)
# print(csr.indptr)
# print(csr.indices)
# print(csr.data)




# def get_deal_data(ci,co,kh,kw,sparity=0.2):
#     # ci,co,kh,kw = 32,32,3,3
#     kernel = get_data.get_conv_sparse_data(ci,co,kh,kw,sparity)
#     # print(kernel)
#     # kernel = np.array(
#     #           [[[[1,0,0],[0,1,0],[0,0,0]],
#     #            [[1,0,0],[0,0,0],[0,0,1]],
#     #            [[1,0,0],[0,0,0],[0,0,0]]],
#     #           [[[1,0,0],[0,0,1],[0,0,0]],
#     #            [[1,0,0],[0,0,0],[0,1,0]],
#     #            [[1,0,0],[0,0,0],[1,0,0]]]]
#     # )
#     # print(kernel.shape)
#     nonzeros,ki,kv,wei = deal_data(kernel)
#     # print("nonzeros:",nonzeros)
#     # print(nonzeros/(kh*kw*ci))
#     # print("ki:",ki)
#     # print("kv:",kv)
#     # print("wei:",wei)
#     return [nonzeros,ki,kv,wei]