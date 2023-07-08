#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/04/26 15:38:32'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/04/26 15:38:32'

import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sraes.train.random_forest import get_data


def symregress_m1(xdata, ydata, features=None):
    print(xdata.shape, ydata.shape)
    ydata = ydata.ravel()
    # function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min', 'sin', 'cos', 'tan', 'inv']
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs'] # , 'log', 'neg', 'inv'
    
    est_gp = SymbolicRegressor(population_size=10000,
                               generations=50,
                               tournament_size=100,
                               function_set=function_set,
                               init_depth=(20, 20),
                               # init_method='full',
                               metric='mse',
                               parsimony_coefficient=0.001,
                               stopping_criteria=0.01,
                               p_crossover=0.9,
                               max_samples=1,
                               n_jobs=-1,
                               verbose=1,
                               feature_names=features,
                               random_state=500)
    est_gp.fit(xdata, ydata)
    print(est_gp.program)
    return est_gp


def compare_est_gp_and_tree(est_gp, xdata, ydata, xtest, ytest):
    # est_gp.fit(xdata, ydata)
    est_tree = DecisionTreeRegressor()
    est_tree.fit(xdata, ydata)
    est_rf = RandomForestRegressor()
    est_rf.fit(xdata, ydata)
    
    y_gp = est_gp.predict(xtest).reshape(-1, 1)
    score_gp = est_gp.score(xtest, ytest)
    y_tree = est_tree.predict(xtest).reshape(-1, 1)
    score_tree = est_tree.score(xtest, ytest)
    y_rf = est_rf.predict(xtest).reshape(-1, 1)
    score_rf = est_rf.score(xtest, ytest)
    
    print(score_gp, score_tree, score_rf)
    _reval(ytest, y_gp)
    _reval(ytest, y_tree)
    _reval(ytest, y_rf)
    
    final_dd = [y_gp, y_tree, y_rf]
    name = ['SymbolicRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor']
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axs[i].plot(ytest, final_dd[i], 'o', label=name[i])
        
        axs[i].set_xlim(0, 3)
        axs[i].set_ylim(0, 3)
    from datetime import datetime
    time = datetime.now()
    plt.savefig('2test_all_fea_model_%s.png' % str(time))

def _reval(ytrue, ypredict):
    ytrue = ytrue.numpy()
    # ypredict = ypredict.numpy()
    print("MSE: ", np.mean((ytrue - ypredict) ** 2))
    print("MAE: ", np.mean(np.abs(ytrue - ypredict)))
    print("RMSE: ", np.sqrt(np.mean((ytrue - ypredict) ** 2)))


def sparse_str(s, feature_index_list, col_index):
    import re
    result = re.findall(r'X\d+', s)
    sym = [feature_index_list[col_index][int(i.replace('X', ''))] for i in result]
    for i in range(len(result)):
        s = s.replace(result[i], sym[i])
    print(s)


# def calculate_y(NVe, ZB, rcovTM, WB, y, NA, Ne):
def calculate_y(y, ZB, NA, Ne):
        
    import numpy as np
    from sympy import symbols, sqrt, Add, Mul
    
    # p1 = "add(div(div(0.742, sqrt(NVe)), ZB), rcovTM)"
    # program1 = 0.742 / np.sqrt(NVe) / ZB + rcovTM
    # p2 = "add(rcovTM, div(div(0.742, abs(sqrt(NVe))), WB))"
    # program2 = rcovTM + 0.742/np.abs(np.sqrt(NVe))/WB
    
    p3 = 'abs(add(sqrt(div(ZB, NA)), sub(0.915, Ne)))'
    print(NA)
    program3 = np.abs(np.sqrt(ZB / NA) + 0.915 - Ne)
    print(program3)
    _reval(y, program3.numpy())
    
    
    
if __name__ == '__main__':
    lll = {'NA': 0, 'NB': 1, 'NC': 2, 'ND': 3, 'NT': 4, 'VA': 5, 'VB': 6, 'VTM': 7, 'rcovTM': 8, 'χTM': 9,
           'ITM': 10, 'Ne': 11, 'a': 12, 'c': 13, 'α': 14, 'β': 15, 'LS': 16, 'Ecn': 17, 'Num': 18,
           'Nc(C)': 19, 'Nc(B)': 20, 'Nc(A)': 21, 'dn1-TM': 22, 'dn2-TM': 23, 'dn3-TM': 24, 'φn1-TM-n2': 25,
           'φn4-TM-n5': 26, 'hTM-n': 27}
    
    fn_index = {-5: '∆GO', -4: '∆GOH', -3: '∆GOOH', -2: 'ηORR', -1: 'ηOER'}
    col_index = -1
    feature_index_list = {
        -1: ['NA', 'NB', 'VA', 'VB', 'ZA', 'ZB', 'WA', 'WB', 'rcovA', 'rcovB', 'χA', 'χB', 'Ne',
             'NVe', 'VTM', 'ZTM', 'WTM', 'rcovTM', 'χTM', 'ITM'],
        -2: ['NA', 'NB', 'NC', 'ND', 'NT', 'VA', 'VB', 'VC', 'VD', 'VTM', 'ZA',
             'ZB', 'ZC', 'ZD', 'ZTM', 'WA', 'WB', 'WC', 'WD', 'WTM', 'rcovA', 'rcovB', 'rcovC',
             'rcovD', 'rcovTM', 'χA', 'χB', 'χC', 'χD', 'χTM', 'ITM', 'Ne', 'NVe']
    }
    train_fn = '../data/all_feature_data/5.train_all_train.xlsx'
    valid_fn = '../data/all_feature_data/5.train_all_train.xlsx'
    
    pmda, xtrain, ytrain, xtest, ytest = get_data(col_index=col_index, train_csv_fn=train_fn, valid_csv_fn=valid_fn)
    print(ytest.shape)
    
    findex = [pmda.feature_num_names[i] for i in ['ZB', "NA", "Ne"]]
    xdd = xtrain[:, findex]
    calculate_y(ytest, xdd[:, 0], xdd[:, 1], xdd[:, 2])
    exit()
    findex = [pmda.feature_num_names[i] for i in feature_index_list[col_index]]
    xdd = xtrain[:, findex]
    xtt = xtest[:, findex]
    model = symregress_m1(xdd, ytrain, features=feature_index_list[col_index])
    sparse_str(str(model.program), feature_index_list, col_index)
    compare_est_gp_and_tree(model, xdd, ytrain, xtt, ytest)
    

