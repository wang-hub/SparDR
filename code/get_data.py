import numpy as np
import os


def get_feature_data(N,C,H,W,CO,K,P,S):
    temp = str(N)
    for i in [C,H,W,CO,K,P,S]:
        temp = temp + "_" + str(i)
    #resnet50
    # input_file_dir = "/home/ww/prune_ww/TVM_MEC/easyUse/data/" + temp 
    #yolov8-prune75
    # input_file_dir = "/home/ww/prune_ww/TVM_MEC/easyUse/yolov8-prune75/" + temp 
    #vgg19
    input_file_dir = "/home/ww/prune_ww/TVM_MEC/easyUse/vgg19/" + temp 

    input_file = input_file_dir + "/input.npy"
    if not os.path.exists(input_file_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(input_file_dir)
    
    x = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    np.save(input_file,x)
    return input_file


def feature_data(N,C,H,W,path):
    input_file = path + "/input.npy"
    if not os.path.exists(input_file):  #判断是否存在文件夹如果不存在则创建为文件夹
        x = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
        np.save(input_file,x)
    return input_file

