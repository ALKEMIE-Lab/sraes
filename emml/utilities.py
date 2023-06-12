#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/05/12 09:46:26'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/05/12 09:46:26'

import datetime
import os


def accord_now_time_create_dir(head_dir='.',
                               prefix_str='',
                               suffix_str='',
                               pass_mkdir=False,
                               to_hour=False,
                               to_minute=False,
                               to_day=False):
    """
    根据当前时间创建文件夹
    :param head_dir: 文件夹的父目录
    :param prefix_str: 文件夹名的前缀
    :param suffix_str: 文件夹名的后缀
    :param pass_mkdir: 是否跳过创建文件夹，只返回文件夹名
    :param to_hour: 时间格式到小时
    :param to_minute: 时间格式到分钟
    :param to_day: 时间格式到天
    :return: 创建的文件夹路径
    """
    now = datetime.datetime.now()
    _fmt = "%Y%m%d_%H.%M.%S.%f"
    if to_hour:
        _fmt = "%Y%m%d_%H"
    if to_minute:
        _fmt = "%Y%m%d_%H.%M"
    if to_day:
        _fmt = "%Y%m%d"
        
    formatted_time = now.strftime(_fmt)
    filename = '_'.join([prefix_str, formatted_time, suffix_str])
    
    _p = os.path.join(head_dir, filename)
    if os.path.exists(_p):
        print("The dir has existed!")
    else:
        print("Create dir: ", _p)
        if not pass_mkdir:
            os.makedirs(_p)
    return _p
