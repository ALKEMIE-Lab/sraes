#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/24 16:32:53'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sraes.data.tools import method2, read_data
import os

def get_correc_column_data():
    head_dir = r"../data/1.20230201_clean.xlsx"
    data = read_data(head_dir)

    column = data.columns.values.tolist()[1:-5]
    train_data = data.values[:, 1:-5]
    train_data = np.array(train_data, dtype=float)
    a = np.corrcoef(train_data, rowvar=0)
    b = np.abs(np.triu(a))
    
    double_index = np.where(b > 0.85)
    index_final = np.array([[double_index[0][m], double_index[1][m]]
                            for m in range(len(double_index[0]))
                            if double_index[0][m] != double_index[1][m]])
    
    gp = method2(index_final)
    
    fdata, columns = [], []
    for i in gp:
        tdata = train_data[:, i]
        cr = np.corrcoef(tdata, rowvar=0)
        columns.append([column[m] for m in i])
        fdata.append(cr)
        
    return fdata, columns


def plt_cor(data, columns):
    label_font = {"fontsize": 8, 'family': 'Times New Roman'}
    # legend_font = {"fontsize": 12, 'family': 'Times New Roman'}
    # tick_font_size = 12
    index_label_font = {"fontsize": 8, 'weight': 'bold', 'family': 'Times New Roman'}
    plt.rc('font', family='Times New Roman', weight='normal')
    nrow = 15
    ncol = 15
    # index_label = "abcdefghi"
    #
    # index_posi = [(-0.02, 2.1), (-0.09, 6.2), (-0.09, 6.2), (-0.05, 5.2), (-0.05, 5.2),
    #               (-0.05, 8.4), (-0.05, 5.2), (-0.09, 6.2), (-0.05, 5.2)]
    
    plt.figure(figsize=(9, 6))
    # axes = axes.flatten()
    # cbarax = plt.subplot2grid((nrow, ncol), (4, 6), colspan=1, rowspan=3)
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    num_row = 5
    num_col = 3

    for i in range(len(data)):
        # ax = axes[i]
        xio, yio= int(i/num_col)*num_row + 1 if int(i/num_col)*num_row != 0 \
            else int(i/num_col)*num_row, int(i%num_col)*num_row
        print(xio, yio)
        ax = plt.subplot2grid((nrow, ncol),
                              (xio, yio), colspan=4, rowspan=4)
        column = []
        for nn in columns[i]:
            if str(nn).startswith('K') or str(nn).startswith('k'):
                _label = r'$\kappa_{%s}$' % nn[1:]
            elif str(nn).startswith('r') or str(nn).startswith('R') or \
                    str(nn).startswith('n') or str(nn).startswith('m'):
                _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
            elif str(nn) == 'a_b':
                _label = 'a/b'
            elif str(nn) == 'NC.1':
                nn = "NCo"
                _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
            else:
                _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
            column.append(_label)
            
        _ = sns.heatmap(data[i], vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar=False)
        
        # if i == 8:
        #     _ = sns.heatmap(data[i], vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar_ax=cbarax,
        #                     cbar_kws={"ticks": np.arange(1, -1.2, -0.2)})
        #     cbarax.tick_params(axis='x', direction='in')
        #     cbarax.tick_params(axis='y', direction='in')

        ax.set_xticks(np.array(range(0, len(column))))
        ax.set_xlim(0, len(column))
        ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)

        ax.set_yticks(np.array(range(0, len(column))))
        ax.set_ylim(0, len(column))
        ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([column[i] for i in range(len(column))],
                           fontdict=label_font,
                           minor=True,
                           rotation=85)
        ax.set_yticklabels([column[i] for i in range(len(column))],
                           fontdict=label_font,
                           minor=True)  # va='center_baseline',

        ax.grid(alpha=0.7, linewidth=0.5, color='white')
        
        ax.tick_params(axis='x', direction='in', labelrotation=85, length=0.00001)
        ax.tick_params(axis='y', direction='in', labelrotation=0, length=0.00001)

        # pp = index_posi[i]
        # ax.text(pp[0], pp[1], '(%s)' % index_label[i], fontdict=index_label_font)

    plt.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.94, wspace=1, hspace=0)
    # plt.show()
    plt.savefig('plt_coref_FigS1.pdf', dpi=600)
    plt.savefig('plt_coref_FigS1.jpg', dpi=600)
    plt.savefig('plt_coref_FigS1.tiff', dpi=600)


if __name__ == '__main__':
    data, columns = get_correc_column_data()
    print(len(data))
    columns = [[n[:5] for n in i] for i in columns]
    print(columns)
    plt_cor(data=data, columns=columns)
