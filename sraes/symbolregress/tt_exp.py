#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/06/13 13:14:46'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/06/13 13:14:46'
import re


def parse_brackets(s):
    # 使用正则表达式匹配括号
    matches = re.findall(r'(.*?)\({1}(.*?)\,(.*?)\)$', s)
    return matches


def run(st):
    matches = parse_brackets(st)
    for i, match in enumerate(matches):
        for nn in match:
            if '(' in match:
                return run(nn)
        print(match)
        # print(f"Match {i+1}: {match}")


if __name__ == '__main__':
    ss = 'abs(add(sqrt(div(ZB, NA)), sub(0.915, Ne)))'
    run(ss)
