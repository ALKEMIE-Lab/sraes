#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/05/25 09:01:54'

import numpy as np
import matplotlib.pyplot as plt
import os


def read_mse_data(fn):
    with open(fn, 'r') as f:
        data = np.array([[float(m.split(':')[-1]) for m in i.split('|')] for i in f.readlines()])
    return data


def read_cal_predit(fn):
    with open(fn, 'r') as f:
        data = np.array([i.split() for i in f.readlines()[0:]], dtype=np.float)
    return data


def plt_simple(data, title="Train vs Test"):
    x = data[:, 0]
    ytrain = data[:, 2]
    ytest = data[:, -2]
    yvalid = data[:, -1]
    
    fig = plt.figure()
    plt.title(title)
    plt.plot(x, ytrain, color='#347FE2', label='train')
    plt.plot(x, ytest, color='r', label='test')
    # plt.plot(x, yvalid, color='green', label='valid', alpha=0.5)
    plt.legend()
    _tloc = np.where(np.abs(ytrain - ytest) == np.min(np.abs(ytrain - ytest)))
    min_arg = 'Train:%.6f Test:%.6f' %(ytrain[_tloc], ytest[_tloc])
    # min_arg = 'Train:%.6f Test:%.6f' %(np.mean(ytrain[2000:]), np.mean(ytest[2000:]))
    # rmse = np.sqrt(np.mean(np.square(pd1[:, 0] - pd1[:, 1])))
    plt.text(2000, 1, min_arg)
    # plt.show()
    plt.savefig('plt_train_%s.pdf'%title, dpi=600)
    # plt.savefig('plt_train_%s.jpg'%title, dpi=600)
    # plt.savefig('plt_train_%s.tiff'%title, dpi=600)



if __name__ == '__main__':
    fn_index = {-5: '∆GO*', -4: '∆GOH*', -3: '∆GOOH*', -2: 'ηORR', -1: 'ηOER'}
    for k, v in fn_index.items():
        fn = os.path.join('../train', v, 'running_%s.log' % v)
        data = read_mse_data(fn)
        print(data.shape)
        plt_simple(data, title=v)
    