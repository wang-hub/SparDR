import get_data
import mec_csrmm_ansor
import sparse_x86_kernelSize1
import sparse_x86_kernelSize3
import denseConv
import os
import numpy as np
import onnx
import tvm


# sparity = 0.8
Dir = "/home/ww/prune_ww/TVM_MEC/easyUse/model/"
def testModel(modelName):
    modelDir = Dir + (str)(modelName) + "-prune/"
    modelPath = modelDir + "model-infer.onnx"
    onnx_model = onnx.load(modelPath)
    # print(modelName,"模型加载成功")

    #tvm转换onnx
    mod,params,_,_,_,_ = tvm.relay.frontend.from_onnx(onnx_model)
    # mod,params,conv_input_shapes,conv_kernel_shapes,conv_pads,conv_strides = tvm.relay.frontend.from_onnx(onnx_model)
    # print("params:",params)
    input(modelName+"加载成功，按下任意键开始")
    from utils import get_op_nums,get_op_para
    get_op_nums(mod)
    op_res = get_op_para(mod,params)

    #数据转为int
    timeRes = []
    for res in op_res:
        #res:
        #   [input shape
        #   kernel shape
        #   kernel padding
        #   kernel strides
        #   weight
        #   kernel name]
        weight = res[4]
        sparity = 1 - np.count_nonzero(weight)/float(weight.size)
        print(np.count_nonzero(weight),weight.size,sparity)
        if sparity > 0.8:
            kernel_name = res[5]
            padding = res[2][0].value
            stride = res[3][0].value
            #处理kernel shape
            kernel_s = res[1]
            for i in range(0,len(kernel_s)):
                kernel_s[i] = kernel_s[i].value
            kernel_s = (tuple)(kernel_s)
            input_s = res[0]
            #处理input shape ()
            if not isinstance(input_s[0],tvm.tir.expr.IntImm):
                input_s[0] = 1
            else:
                input_s[0] = input_s[0].value
            for i in range(1,len(input_s)):
                if not isinstance(input_s[i],tvm.tir.expr.IntImm):
                        if "model.21" in kernel_name or "model.22.cv2" in kernel_name or "model.22.cv3" in kernel_name:
                            input_s[i] = 20
                        if "model.12.m." in kernel_name or "model.18" in kernel_name or "model.19" in kernel_name:
                            input_s[i] = 40
                        if "model.15.m." in kernel_name or "model.16" in kernel_name:
                            input_s[i] = 80
                else:
                    input_s[i] = input_s[i].value
            input_s = (tuple)(input_s)
            #保存所有权重
            kernelPath = modelDir + (str)(input_s) 
            kernelPath = kernelPath + (str)(kernel_s)
            kernelPath = kernelPath +'_'+ (str)(padding)
            kernelPath = kernelPath +'_'+ (str)(stride)
            kernelPath = kernelPath +'_'+ (str)(sparity)
            kernelPath = kernelPath +'_'+ (str)(kernel_name)
            # print(kernelPath)
            if not os.path.exists(kernelPath):  #判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(kernelPath)
                N,C,H,W = input_s
                get_data.feature_data(N,C,H,W,kernelPath)
                #保存权重
                weightPath = kernelPath + "/weight.npy"
                if not os.path.exists(weightPath):
                    np.save(weightPath,weight)
                    print("权重保存成功,路径：",weightPath)
            input_file = kernelPath + "/input.npy"
            kernel_file = kernelPath + "/weight.npy"
            K = weight.shape[2]
            P = padding
            S = stride
            if(K == 1):
                # continue
                ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=True)
                if(S == 1):
                    s_x86Time = sparse_x86_kernelSize1.test(input_file,kernel_file,P,S)
                else:
                    s_x86Time = 0
                denseConvTime = denseConv.test(input_file,kernel_file,P,S)
            else:
                #x86只支持p=1
                # assert P == 1
                ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=False)
                s_x86Time = 0
                # s_x86Time = sparse_x86_kernelSize3.test(input_file,kernel_file,P,S)
                denseConvTime = denseConv.test(input_file,kernel_file,P,S)
            #保存结果
            res = [ourTime,denseConvTime,s_x86Time]
            time_file = kernelPath + "/time.txt"
            with open(time_file,'a') as f:
                f.writelines((str)(res))
                f.writelines("\n")
            res = [ourTime,denseConvTime,s_x86Time]
            print(res)
            timeRes.append(res)
    #调优结果保存
    timePath = modelDir + "/time"
    while os.path.exists(timePath+'.npy'):
        timePath = timePath + '1'
    np.save(timePath+'.npy',np.array(timeRes))
    print("##########################"+modelName+"调优完成"+"##########################")
    # print(modelName,"调优完成")
    print("结果[ourTime,denseConvTime,s_x86Time]保存在：",timePath+'.npy')
    print("########################################################################") 
    # 暂停并等待用户按下任意键继续
    # print(modelName + " is ok")
    input(modelName + " is ok,按下任意键开始下一个网络")
    # exit(0)
#调优三种网络
for modelName in [
                 "ResNet50",
                  "VGG19",
                  "YOLOv8",
                  ]:
    testModel(modelName)


# def testModel(modelName):
#     modelDir = Dir + (str)(modelName) + "-prune/"
#     modelPath = modelDir + "model-infer.onnx"
#     onnx_model = onnx.load(modelPath)
#     print(modelName,"模型加载成功")

#     #tvm转换onnx
#     mod,params,_,_,_,_ = tvm.relay.frontend.from_onnx(onnx_model)
#     # mod,params,conv_input_shapes,conv_kernel_shapes,conv_pads,conv_strides = tvm.relay.frontend.from_onnx(onnx_model)
    # print("params:",params)
    # input()    
    #处理YOLOv8中batch！=1和尺寸不确定情况
    # if modelName == 'YOLOv8':
    #     idx = 0
    #     for item in onnx_model.graph.initializer:
    #         if "conv" in (str)(item.name) and "weight" in (str)(item.name):
    #             conv_input_shapes[idx] = (list)(conv_input_shapes[idx])
    #             conv_input_shapes[idx][0] = 1
    #             if "model.21" in item.name or "model.22.cv2" in item.name or "model.22.cv3" in item.name:
    #                 conv_input_shapes[idx][2] = 20
    #                 conv_input_shapes[idx][3] = 20
    #             if "model.12.m." in item.name or "model.18" in item.name or "model.19" in item.name:
    #                 conv_input_shapes[idx][2] = 40
    #                 conv_input_shapes[idx][3] = 40
    #             if "model.15.m." in item.name or "model.16" in item.name:
    #                 conv_input_shapes[idx][2] = 80
    #                 conv_input_shapes[idx][3] = 80
    #             conv_input_shapes[idx] = (tuple)(conv_input_shapes[idx])
    #             idx = idx + 1
    # exit(0)
    #模型预处理
    #提取模型所有conv算子
    # idx = 0
    # print("conv 算子数：",len(conv_input_shapes))
    # # print(len(conv_kernel_shapes))
    # input_shapes = []
    # kernel_shapes = []
    # pads = []
    # strides = []
    # sparitys = []
    # spaIdx = 0
    # for item in onnx_model.graph.initializer:
    #     # print(item.name)
    #     if "conv" in (str)(item.name) and "weight" in (str)(item.name):
    #         print(item.name)
    #         print(onnx.numpy_helper.to_array(item).shape)
    #         # print("shape: ", item.dims)
    #         weight = np.frombuffer(item.raw_data, dtype=np.float32).reshape(*item.dims)    
    #         #处理Nan值
    #         # weight = np.nan_to_num(weight,0)
    #         print("idx:",idx)
    #         print("input.shape:",conv_input_shapes[idx])
    #         print("kernel.shape:",conv_kernel_shapes[idx])
    #         print("weight.shape(41):",item.dims)
    #         sparity = 1 - np.count_nonzero(weight)/float(weight.size)
    #         print(np.count_nonzero(weight),weight.size,sparity)
    #         if sparity > 0.8:
    #             #保存所有权重
    #             kernelPath = modelDir + (str)(conv_input_shapes[idx]) 
    #             kernelPath = kernelPath + (str)(conv_kernel_shapes[idx])
    #             kernelPath = kernelPath +'_'+ (str)(conv_pads[idx])
    #             kernelPath = kernelPath +'_'+ (str)(conv_strides[idx])
    #             kernelPath = kernelPath +'_'+ (str)(sparity)
    #             # print(kernelPath)
    #             input_shapes.append(conv_input_shapes[idx])
    #             kernel_shapes.append(conv_kernel_shapes[idx])
    #             pads.append(conv_pads[idx])
    #             strides.append(conv_strides[idx])
    #             sparitys.append(sparity)
    #             if not os.path.exists(kernelPath):  #判断是否存在文件夹如果不存在则创建为文件夹
    #                 os.makedirs(kernelPath)
    #                 N,C,H,W = conv_input_shapes[idx]
    #                 get_data.feature_data(N,C,H,W,kernelPath)
    #                 #保存权重
    #                 weightPath = kernelPath + "/weight.npy"
    #                 if not os.path.exists(weightPath):
    #                     np.save(weightPath,weight)
    #                     print("权重保存成功,路径：",weightPath)
    #             spaIdx = spaIdx + 1
    #         print("spa:",spaIdx)
    #         idx = idx + 1
    # #模型转换

    # #模型调优
    # timeRes = []
    # print("intput_shape:")
    # print(input_shapes)
    # print("kernel_shape:")
    # print(kernel_shapes)
    # input("当前网络："+modelName)
    # for i in range(0,spaIdx):
    #     print(input_shapes[i])
    #     print(kernel_shapes[i])
    #     N,C,H,W = input_shapes[i]
    #     P = pads[i]
    #     S = strides[i]
    #     K = kernel_shapes[i][2]
    #     spa = sparitys[i]
    #     kernelPath = modelDir + (str)(input_shapes[i]) 
    #     kernelPath = kernelPath + (str)(kernel_shapes[i])
    #     kernelPath = kernelPath +'_'+ (str)(pads[i])
    #     kernelPath = kernelPath +'_'+ (str)(strides[i])
    #     kernelPath = kernelPath +'_'+ (str)(spa)
    #     input_file = kernelPath + "/input.npy"
    #     kernel_file = kernelPath + "/weight.npy"
    #     if(K == 1):
    #         # continue
    #         ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=True)
    #         if(S == 1):
    #             s_x86Time = sparse_x86_kernelSize1.test(input_file,kernel_file,P,S)
    #         else:
    #             s_x86Time = 0
    #         denseConvTime = denseConv.test(input_file,kernel_file,P,S)
    #     else:
    #         #x86只支持p=1
    #         # assert P == 1
    #         ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=False)
    #         s_x86Time = 0
    #         # s_x86Time = sparse_x86_kernelSize3.test(input_file,kernel_file,P,S)
    #         denseConvTime = denseConv.test(input_file,kernel_file,P,S)
    #     #保存结果
    #     res = [ourTime,denseConvTime,s_x86Time]
    #     time_file = kernelPath + "/time.txt"
    #     with open(time_file,'a') as f:
    #         f.writelines((str)(res))
    #         f.writelines("\n")
    #     res = [ourTime,denseConvTime,s_x86Time]
    #     print(res)
    #     timeRes.append(res)
    # #调优结果保存
    # timePath = modelDir + "/time"
    # while os.path.exists(timePath+'.npy'):
    #     timePath = timePath + '1'
    # np.save(timePath+'.npy',np.array(timeRes))
    # print("##########################"+modelName+"调优完成"+"##########################")
    # # print(modelName,"调优完成")
    # print("结果[ourTime,denseConvTime,s_x86Time]保存在：",timePath+'.npy')
    # print("########################################################################") 
    # # 暂停并等待用户按下任意键继续
    # input(modelName + " is ok")







def test(N,C,H,W,CO,K,P,S):
    input_file = get_data.get_feature_data(N,C,H,W,CO,K,P,S)
    in_dir = os.path.dirname(input_file)
    kernel_file = in_dir + "/tensor.npy"

    if not os.path.exists(kernel_file):
        print(kernel_file + "kernel not import!!")
        return 
    if(K == 1):
        ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=True)
        # s_x86Time = sparse_x86_kernelSize1.test(input_file,kernel_file,P,S)
        denseConvTime = denseConv.test(input_file,kernel_file,P,S)
    elif(K == 3):
        #x86只支持p=1
        # assert P == 1
        ourTime = mec_csrmm_ansor.conv(input_file,kernel_file,P,S,kISone=False)
        s_x86Time = 0
        # s_x86Time = sparse_x86_kernelSize3.test(input_file,kernel_file,P,S)
        denseConvTime = denseConv.test(input_file,kernel_file,P,S)
    # exit(0)
    rs = np.array([ourTime,denseConvTime,s_x86Time])
    time_file = in_dir + "/time.npy"
    np.save(time_file,rs)
    return [ourTime,denseConvTime,s_x86Time]


N,H,W,P,S = 1,224,224,0,1
#resnet
# CO = [44,44,44,89,89,89,179,179,179,179,358,358,358,358,716,716,2048,2048,358]
# CI = [44,44,179,89,179,358,44,179,358,716,89,179,358,716,179,358,358,716,2048]
# KK = [1,3,1,3,1,1,1,3,1,1,1,1,3,1,1,1,1,1,1]

#vgg19
CO = [64,128,128,256,256,512,512]
CI = [64,64,128,128,256,256,512]
KK = [3,3,3,3,3,3,3]

# CO = [512,512]
# CI = [256,512]
# KK = [3,3]

# CO = [358]
# CI = [2048]
# KK = [1]
# P = 1

#调优
# for i in range(len(CO)):
#     co = CO[i]
#     ci = CI[i]
#     k = KK[i]
#     # if(k == 1):
#     #     print("K == 1，skip")
#     #     continue
#     if(k == 3):
#         P = 1
    
#     # with open("/home/ww/prune_ww/TVM_MEC/easyUse/data/time-avx2.txt",'a') as file:
#     with open("/home/ww/prune_ww/TVM_MEC/easyUse/vgg19/time-avx2.txt",'a') as file:
#         result = test(N,ci,H,W,co,k,P,S)
#         args = (N,ci,H,W,co,k,P,S)
#         file.writelines(str(args) + "\n")
#         file.write(str(result) + "\n")



