#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/10 16:39:37'
#
# from ztml.tools import get_random_groupby_index, norepeat_randint
# from ztml.data.feature_normalize_data import get_normalize_data
#
# from ztml.rdata.clean_csv_data import get_clean_data, read_data
# from ztml.rdata.rename_cloumn import get_rename_column_data
# from copy import deepcopy
from emml.data.tools import get_random_from_each_groupby_index
from emml.data.tools import auto_get_corelated_group
import numpy as np
import pandas as pd

SHEET_NAME = 'Original'


def read_data(filename, is_csv=False):
    if is_csv:
        data = pd.read_csv(filename)
    else:
        data = pd.read_excel(filename, sheet_name=SHEET_NAME, header=0, index_col=0)
    return data


def normalize(data):
    dmin = np.min(data, axis=0)
    dmax = np.max(data, axis=0)
    
    scale = 1 / (dmax - dmin)
    dd = (data - dmin) * scale
    return dd


def get_normalize_data(data, gogal_column=None):
    if gogal_column is not None:
        pass
    else:
        column = data.columns.values.tolist()[1:-5]
        train_data = np.array(data.values[:, 1:-5], dtype=float)
        a = np.corrcoef(train_data, rowvar=False)
        b = np.abs(np.triu(a))
        gp = auto_get_corelated_group(data=b, coref_val=0.909, is_triu=True, get_remove=True)
        print('corelated column:', len(gp[1]))
        for i in sorted(gp[1], reverse=True):
            column.pop(i)
        gogal_column = column
    
    feature = data[gogal_column]
    ffe = normalize(feature.values)
    print(gogal_column)
    print("Input gogal column: ", len(gogal_column))
    # plt_each_dis(ffe, 'nor.pdf')
    # plt_each_dis(feature.values, '0.pdf')
    for i in range(-5, 0):
        gogal_column.append(data.columns.values.tolist()[i])
    
    label = data[gogal_column[-5:]]
    
    return pd.DataFrame(np.hstack((ffe, label)), columns=gogal_column), gogal_column[:-5]


def reconvrt_vele2num(data, write_trans_value_data2file=False, rm_max_min_equal_0=False):
    print(data.shape)
    gogal_cl = ['VA', 'VB', 'VC', 'VD', 'VTM']
    now_data = data[gogal_cl]
    npdata = now_data.to_numpy(dtype=str)
    uni_npdata = np.unique(npdata)
    if write_trans_value_data2file:
        f_data = np.array([uni_npdata.tolist(), range(uni_npdata.shape[0])])
        print(f_data)
        # pddata = pd.DataFrame(f_data.transpose(), columns=['values', 'num'])
        # pddata.to_excel('value_data2num.xlsx')
    
    f_data = dict(zip(uni_npdata.tolist(), range(uni_npdata.shape[0])))
    for i in gogal_cl:
        for n in data.index.tolist():
            data.loc[n, i] = f_data[str(data.loc[n, i])]
    print(data.shape)
    # print(f_data)
    # for i in range(uni_npdata.shape[0]):
    #     print(i, uni_npdata[i])
    if rm_max_min_equal_0:
        # 查找 列最大值-最小值=0 的列
        deal_column = data.columns.values.tolist()
        print(deal_column)
        deal_data = data[deal_column[1:-5]]
        ndata = np.array(deal_data, dtype=float)
        gogal_dd = np.where((np.max(ndata, axis=0) - np.min(ndata, axis=0)) == 0)
        rmcol = [deal_column[1:-5][i] for i in gogal_dd[0]]
        print("rm column: %s" % str(rmcol))
        for i in rmcol:
            deal_column.remove(i)
        data = data[deal_column]
    data.to_excel('1.20230201_clean.xlsx', sheet_name=SHEET_NAME)
    
    
def run_get_clean_file():
    tmp_file = '0.20230201.xlsx'
    clean_train_data = read_data(tmp_file)
    print(clean_train_data.columns)
    reconvrt_vele2num(clean_train_data, write_trans_value_data2file=True, rm_max_min_equal_0=True)


def read_rename_clean_datafile():
    tmp_file = '1.20230201_clean.xlsx'
    clean_train_data = read_data(tmp_file)
    print(clean_train_data.columns)
    
    # 获取归一化之后数据，并删除相关列
    normalized_data, _ = get_normalize_data(clean_train_data)
    normalized_data.to_excel('2.normalized_data.xlsx', sheet_name=SHEET_NAME)


def random_data():
    tmp_file = '2.normalized_data.xlsx'
    normalized_data = read_data(tmp_file)
    normalized_data['Index'] = [5] * 81 + [6] * 39 + [7] * 8
    use2train_data, use2valid_data = get_random_from_each_groupby_index(normalized_data, column_index=['Index'],
                                                                        ratio=0.8, to_data=True, train_index=None)
    print(use2train_data.shape, use2valid_data.shape)
    
    use2train_data = use2train_data.reset_index(drop=True)
    train_train_data, train_test_data = get_random_from_each_groupby_index(use2train_data, column_index=['Index'],
                                                                           ratio=0.8, to_data=True)
    print(train_train_data.shape, train_test_data.shape)
    # use2valid_data = use2valid_data.reset_index(drop=True)
    # valid_10data, valid_30data = get_random_from_each_groupby_index(use2valid_data, column_index=['Index'],
    #                                                       ratio=0.26, to_data=True)
    # print(valid_10data.shape, valid_30data.shape)

    print("Final features: %d" % (train_train_data.shape[1] - 6))
    
    __rm_point_columns(train_train_data).to_excel('5.train_%d_train.xlsx' % train_train_data.shape[0],
                                                  sheet_name=SHEET_NAME, index=False)
    __rm_point_columns(train_test_data).to_excel('5.train_%d_test.xlsx' % train_test_data.shape[0],
                                                 sheet_name=SHEET_NAME, index=False)
    __rm_point_columns(use2valid_data).to_excel('5.%d_for_check.xlsx' % use2valid_data.shape[0],
                                                sheet_name=SHEET_NAME, index=False)
    # __rm_point_columns(valid_30data).to_excel('4.28_for_predict.xlsx', sheet_name=SHEET_NAME, index=False)


def __rm_point_columns(data, index=None):
    if index is None:
        index = ["Index"]
        
    ll = data.columns.values.tolist()
    for i in index:
        ll.remove(i)
    return data[ll]


if __name__ == '__main__':
    # now_head_dir = "2_rmcoref_data"  # 包含nop数值和 zt数值
    # now_head_dir = "all_data"      # nop 被转换为01，且没有根据相关系数实施特征工程
    
    # now_head_dir = "all_rmcoref_data"  # nop 被转换为01， 根据相关系数删除相关系数大于0.9的项
    # file_name = r"simple_dataset.csv"
    # now_is_nop_to_01 = True
    # run_compounds_split(file_name, head_dir=now_head_dir, is_nop_to_01=now_is_nop_to_01)
    
    # run_get_clean_file()
    # read_rename_clean_datafile()
    random_data()
