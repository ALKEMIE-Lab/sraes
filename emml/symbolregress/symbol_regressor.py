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

from pprint import pprint
from emml.train.random_forest import get_data
from collections import defaultdict
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def symregress_m1(xdata, ydata):
    print(xdata.shape, ydata.shape)
    ydata = ydata.ravel()
    # function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min', 'sin', 'cos', 'tan', 'inv']
    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'abs', 'neg', 'max', 'min']
    
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20,
                               function_set=function_set,
                               # init_depth=(12, 12),
                               init_method='half and half',
                               metric='mse',
                               parsimony_coefficient=0.01,
                               stopping_criteria=0.01,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               random_state=0)
    est_gp.fit(xdata, ydata)
    print(est_gp._program)
    return est_gp


def tt(est_gp, xdata, ydata, xtest, ytest):
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
    final_dd = [y_gp, y_tree, y_rf]
    name = ['SymbolicRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor']

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axs[i].plot(ytest, final_dd[i], 'o', label=name[i])
        axs[i].set_xlim(0, 3)
        axs[i].set_ylim(0, 3)
    plt.savefig('test_3model.png')
    
    # fig = plt.figure(figsize=(12, 10))
    #
    # for i, (y, score, title) in enumerate([(y_truth, None, "Ground Truth"),
    #                                        (y_gp, score_gp, "SymbolicRegressor"),
    #                                        (y_tree, score_tree, "DecisionTreeRegressor"),
    #                                        (y_rf, score_rf, "RandomForestRegressor")]):
    #
    #     ax = fig.add_subplot(2, 2, i+1, projection='3d')
    #     ax.set_xlim(-1, 1)
    #     ax.set_ylim(-1, 1)
    #     surf = ax.plot_surface(x0, x1, y, rstride=1, cstride=1, color='green', alpha=0.5)
    #     points = ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain)
    #     if score is not None:
    #         score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'x', fontsize=14)
    #     plt.title(title)
    # plt.show()


if __name__ == '__main__':
    lll = {'NA': 0, 'NB': 1, 'NC': 2, 'ND': 3, 'NT': 4, 'VA': 5, 'VB': 6, 'VTM': 7, 'rcovTM': 8, 'χTM': 9,
           'ITM': 10, 'Ne': 11, 'a': 12, 'c': 13, 'α': 14, 'β': 15, 'LS': 16, 'Ecn': 17, 'Num': 18,
           'Nc(C)': 19, 'Nc(B)': 20, 'Nc(A)': 21, 'dn1-TM': 22, 'dn2-TM': 23, 'dn3-TM': 24, 'φn1-TM-n2': 25,
           'φn4-TM-n5': 26, 'hTM-n': 27}
    
    fn_index = {-5: '∆GO', -4: '∆GOH', -3: '∆GOOH', -2: 'ηORR', -1: 'ηOER'}
    feature_index_list = {-1: ['Ne', 'χTM', 'φn4-TM-n5', 'rcovTM', 'dn3-TM', 'ITM', 'φn1-TM-n2', 'VTM', 'dn1-TM',
                               'a', 'c', 'dn2-TM'],
                          -2: ['Ne', 'rcovTM', 'χTM', 'ITM', 'φn1-TM-n2', 'VTM', 'c', 'hTM-n', 'dn3-TM',
                               'a', 'dn2-TM', 'β']}
    
    # for k, v in fn_index.items():
    #     print(k, v)
    #     col_index = k
    col_index = -1
    pmda, xtrain, ytrain, xtest, ytest = get_data(col_index=col_index)
    print(ytest.shape)

    findex = [pmda.feature_num_names[i] for i in feature_index_list[col_index]]
    xdd = xtrain[:, findex]
    xtt = xtest[:, findex]
    model = symregress_m1(xdd, ytrain)
    tt(model, xdd, ytrain, xtt, ytest)
