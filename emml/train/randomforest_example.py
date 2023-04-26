#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/04/26 08:08:09'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/04/26 08:08:09'


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import graphviz


def run(is_plt=False):
    data = load_iris()
    xdagta = data['data']
    y = data['target']
    x_train, x_test, y_train, y_test = train_test_split(xdagta, y, test_size=0.2, random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    feature_importances = rf.feature_importances_
    print("Feature importances:", feature_importances)
    
    threshold = 0.1  # 设定一个阈值，用于筛选特征
    selected_features = np.where(feature_importances > threshold)[0]
    print("Selected features:", selected_features)
    
    # x_train_selected = x_train[:, selected_features]
    # x_test_selected = x_test[:, selected_features]
    if is_plt:
        tree_to_visualize = rf.estimators_[0]
    
        dot_data = tree.export_graphviz(tree_to_visualize, out_file=None,
                                        feature_names=data.feature_names,
                                        class_names=data.target_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("iris_decision_tree")  # 保存为 iris_decision_tree.pdf
    return rf, data, feature_importances


def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def tree_view(rf, data, feature_names):
    # 选择一个决策树，这里选择第一个树
    tree_to_visualize = rf.estimators_[0]
    
    # 可视化该决策树
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_visualize, feature_names=feature_names, class_names=data['target_names'], filled=True,
              rounded=True)
    plt.show()
    
    n_trees = len(rf.estimators_)
    n_cols = 2
    n_rows = (n_trees + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10), constrained_layout=True)
    
    for idx, tt in enumerate(rf.estimators_):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_tree(tt, feature_names=feature_names, class_names=data['target_names'], filled=True, rounded=True, ax=ax)
        ax.set_title(f"Tree {idx + 1}")
    
    # 如果子图数量不是偶数，隐藏最后一个空白子图
    if n_trees % n_cols != 0:
        axes[-1, -1].axis('off')
    
    # plt.show()
    plt.savefig("1.pdf")


if __name__ == '__main__':
    rfs, dd, fi = run(is_plt=False)
    # fn = dd['feature_names']
    # plot_feature_importances(fi, fn)
    # tree_view(rfs, dd, fn)
