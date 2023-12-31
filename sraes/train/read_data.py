#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/05/24 15:54:00'

from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.utils.data as Data
from sraes.constant import SHEET_NAME


class PmEmmlData(Dataset):
    
    output_index = {"∆GO*": -5,
                    "∆GOH*": -4,
                    "∆GOOH*": -3,
                    "ηORR": -2,
                    "ηOER": -1}
    
    def __init__(self, filename: str, output_index=0, output_name=None, output_all=False):
        if filename.endswith('.csv'):
            self.meta_data = pd.read_csv(filename)
        elif filename.endswith('.xlsx'):
            self.meta_data = pd.read_excel(filename, sheet_name=SHEET_NAME, header=0)

        self.data = self.meta_data.values
        self.train_data = self.data[:, :-5]
        self.output_all = output_all
        self.output_name = output_name
        self.output_index = output_index
        
        if self.output_all:
            self.label = self.data[:, -5:]
        else:
            if self.output_name:
                self.label = self.data[:, self.output_index[self.output_name]]
            else:
                self.label = self.data[:, self.output_index]
    
    @property
    def feature_names(self):
        return self.meta_data.columns.values[:-5]
    
    @property
    def feature_num_names(self):
        return {column_name: index for index, column_name in enumerate(self.meta_data.columns[:-5])}
    
    @property
    def label_names(self):
        if self.output_all:
            return self.meta_data.columns.values[-5:]
        else:
            if self.output_name:
                return self.output_name
            else:
                return self.meta_data.columns.values[self.output_index]
    
    def __getitem__(self, item):
        # train_val, label = self.data[item], self.targets[item]
        return self.train_data[item, :], self.label[item]

    def __len__(self):
        return len(self.data)


def load_pmdata(filename, batch_size=588, shuffle=True, output_all=True, output_index=-5):
    pmdata = PmEmmlData(filename, output_all=output_all, output_index=output_index)
    train_loader = Data.DataLoader(dataset=pmdata, batch_size=batch_size, shuffle=shuffle)
    return train_loader


if __name__ == '__main__':
    
    root_path = r'../data/3.train_63_train.xlsx'
    # a = PmData(root=root_path, ele='Sb')
    a = PmEmmlData(root_path, output_all=False, output_index=-5)
    print(a.train_data.shape)
    # print(a[10894])
    # print(len(a))
