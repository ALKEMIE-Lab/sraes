#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/04/26 15:10:57'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/04/26 15:10:57'

from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz


def t11():
    x0 = np.arange(-1, 1, 1/10.)
    x1 = np.arange(-1, 1, 1/10.)
    x0, x1 = np.meshgrid(x0, x1)
    y_truth = x0**2 - x1**2 + x1 - 1
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                           color='green', alpha=0.5)
    plt.show()


def generate_data():
    rng = check_random_state(0)

    # Training samples
    X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = X_train[:, 0]**2 - X_train[:, 1]**2 + X_train[:, 1] - 1
    
    # Testing samples
    X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = X_test[:, 0]**2 - X_test[:, 1]**2 + X_test[:, 1] - 1
    return X_train, y_train, X_test, y_test


def symregress_m1(xtrain ,ytrain):
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(xtrain ,ytrain)
    print(est_gp._program)
    return est_gp


def ddecisiontreegress(xtrain ,ytrain, xtest, ytest):
    
    x0 = np.arange(-1, 1, 1/10.)
    x1 = np.arange(-1, 1, 1/10.)
    x0, x1 = np.meshgrid(x0, x1)
    y_truth = x0**2 - x1**2 + x1 - 1
    
    est_gp = SymbolicRegressor(population_size=5000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(xtrain ,ytrain)
    est_tree = DecisionTreeRegressor()
    est_tree.fit(xtrain ,ytrain)
    est_rf = RandomForestRegressor()
    est_rf.fit(xtrain ,ytrain)

    y_gp = est_gp.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_gp = est_gp.score(xtest, ytest)
    y_tree = est_tree.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_tree = est_tree.score(xtest, ytest)
    y_rf = est_rf.predict(np.c_[x0.ravel(), x1.ravel()]).reshape(x0.shape)
    score_rf = est_rf.score(xtest, ytest)
    
    fig = plt.figure(figsize=(12, 10))
    
    for i, (y, score, title) in enumerate([(y_truth, None, "Ground Truth"),
                                           (y_gp, score_gp, "SymbolicRegressor"),
                                           (y_tree, score_tree, "DecisionTreeRegressor"),
                                           (y_rf, score_rf, "RandomForestRegressor")]):
        
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        surf = ax.plot_surface(x0, x1, y, rstride=1, cstride=1, color='green', alpha=0.5)
        points = ax.scatter(xtrain[:, 0], xtrain[:, 1], ytrain)
        if score is not None:
            score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'x', fontsize=14)
        plt.title(title)
    plt.show()


def visualize(est_gp):
    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render("gplearn_t1")

    print(est_gp._program.parents)

    idx = est_gp._program.parents['donor_idx']
    fade_nodes = est_gp._program.parents['donor_nodes']
    dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    graph2 = graphviz.Source(dot_data)
    graph2.render("gplearn_t2")
    
    
if __name__ == '__main__':
    # t11()
    xa, ya, xe, ye = generate_data()
    # ddecisiontreegress(xa, ya, xe, ye)
    mm = symregress_m1(xa, ya)
    visualize(mm)
    