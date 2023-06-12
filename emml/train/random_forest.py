#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/04/23 08:42:41'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/04/23 08:42:41'

import os

import numpy as np
import matplotlib.pyplot as plt
# import graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from emml.train.read_data import load_pmdata, PmEmmlData
from emml.utilities import accord_now_time_create_dir


def get_data(col_index, train_csv_fn, valid_csv_fn):
    """
    获取数据
    :param col_index:
    :return:
    """
    x_train, y_train = [], []
    output_all = False
    pmdata = PmEmmlData(train_csv_fn, output_all=output_all, output_index=col_index)
    
    train_pmdata_loader = load_pmdata(filename=train_csv_fn, shuffle=True, output_all=output_all,
                                      output_index=col_index, batch_size=82)
    # test_pmdata_loader = load_pmdata(filename=test_csv_fn, shuffle=True, output_all=output_all,
    #                                  output_index=col_index, batch_size=20)
    valid_pmdata_loader = load_pmdata(filename=valid_csv_fn, shuffle=True, output_all=output_all,
                                      output_index=col_index, batch_size=26)
    for step, (b_x, b_y) in enumerate(train_pmdata_loader):
        x_train = b_x.float()
        y_train = b_y.reshape(-1, 1).float()
    
    for _, (tx, ty) in enumerate(valid_pmdata_loader):
        x_test = tx.float()
        y_test = ty.reshape(-1, 1).float()
    
    return pmdata, x_train, y_train, x_test, y_test


def random_froest(x_train, y_train, x_test, y_test, log_name='random_forest.log'):
    """
    训练随机森林模型，评估模型精度
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param log_name:
    :return:
    """
    
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(x_train, y_train)
    
    # 使用训练好的模型进行预测
    y_pred = rf_regressor.predict(x_test)
    
    # 计算MSE、RMSE、MAE和R²分数
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    tt = ["Mean Squared Error (MSE):", mse, '\n',
          "Root Mean Squared Error (RMSE):", rmse, '\n',
          "Mean Absolute Error (MAE):", mae, '\n',
          "R-squared (R2):", r2, '\n', '\n']
    
    print(''.join(str(i) for i in tt))
    with open(log_name, 'a') as f:
        f.write(''.join(str(i) for i in tt))
    
    return rf_regressor


def plt_rf(rf, pmdata, index_name='run_random',
           head_dir='.',
           is_plt_forest_model=False,
           is_plt_feature_importance=False,
           is_plt_all_forest_model=False):
    """
    绘制随机森林模型重要特征，和模型可视化
    :param rf: 随机森林模型
    :param pmdata: 数据集
    :param index_name: 保存文件的名称
    :param head_dir: 保存文件的根目录
    :param is_plt_forest_model: 是否绘制随机森林模型
    :param is_plt_feature_importance: 是否绘制特征重要性
    :param is_plt_all_forest_model: 是否绘制所有的随机森林模型
    :return:
    """
    feature_importances = rf.feature_importances_
    print("Feature importances:", feature_importances)
    threshold = 0.1  # 设定一个阈值，用于筛选特征
    selected_features = np.where(feature_importances > threshold)[0]
    print("Selected features:", selected_features)
    
    # x_train_selected = x_train[:, selected_features]
    # x_test_selected = x_test[:, selected_features]
    feature_names = pmdata.feature_names
    label_names = pmdata.label_names
    model_index = [0, -1]
    tree_to_visualizes = [rf.estimators_[i] for i in model_index]
    
    if is_plt_feature_importance:
        indices = np.argsort(feature_importances)
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.savefig(os.path.join(head_dir, f"{index_name}_feature_importance.pdf"))
    
    if is_plt_forest_model:
        # # 可视化模型
        # dot_data = tree.export_graphviz(tree_to_visualize, out_file=None,
        #                                 feature_names=feature_names,
        #                                 class_names=label_names,
        #                                 filled=True, rounded=True,
        #                                 special_characters=True)
        # graph = graphviz.Source(dot_data)
        # graph.render("model_random_forest_graphviz_%d" % model_index)  # 保存为 iris_decision_tree.pdf
        
        # 另一种绘制模型的方法
        # 选择一个决策树，这里选择第一个树
        for index, tree_to_visualize in enumerate(tree_to_visualizes):
            plt.figure(figsize=(20, 10))
            plot_tree(tree_to_visualize, feature_names=feature_names, class_names=label_names, filled=True,
                      rounded=True)
            plt.savefig(os.path.join(head_dir,
                                     f"{index_name}_model_random_forest_%d.pdf" % model_index[index]))
    
    if is_plt_all_forest_model:
        n_trees = len(rf.estimators_)
        n_cols = 2
        n_rows = (n_trees + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 10), constrained_layout=True)
        
        for idx, tt in enumerate(rf.estimators_):
            ax = axes[idx // n_cols, idx % n_cols]
            plot_tree(tt, feature_names=feature_names, class_names=label_names, filled=True, rounded=True, ax=ax)
            ax.set_title(f"Tree {idx + 1}")
        
        # 如果子图数量不是偶数，隐藏最后一个空白子图
        if n_trees % n_cols != 0:
            axes[-1, -1].axis('off')
        
        # plt.show()
        plt.savefig(os.path.join(head_dir, f"{index_name}_model_random_forest_all.pdf"))
    return rf, feature_importances


if __name__ == '__main__':
    # train_fn = '../data/only_ele_data/5.train_82_train.xlsx'
    # # test_fn = '../data/only_ele_data/5.train_20_test.xlsx'
    # valid_fn = '../data/only_ele_data/5.26_for_check.xlsx'

    train_fn = '../data/all_feature_data/5.train_96_train.xlsx'
    valid_fn = '../data/all_feature_data/5.24_for_check.xlsx'
    dir_name = accord_now_time_create_dir('.', prefix_str='rf_feature', suffix_str='1',
                                          to_hour=True)
    
    fn_index = {-5: '∆GO', -4: '∆GOH', -3: '∆GOOH', -2: 'ηORR', -1: 'ηOER'}
    for k, v in fn_index.items():
        print(k, v)
        col_index = k
        pmda, xtrain, ytrain, xtest, ytest = get_data(col_index=col_index,
                                                      train_csv_fn=train_fn,
                                                      valid_csv_fn=valid_fn)
        rfs = random_froest(xtrain, ytrain, xtest, ytest,
                            log_name=os.path.join(dir_name, f"{fn_index[col_index]}_random_forest_log.txt"))
        plt_rf(rfs, pmda, index_name=fn_index[col_index], head_dir=dir_name,
               is_plt_forest_model=True,
               is_plt_feature_importance=True,
               is_plt_all_forest_model=False)
        # break
