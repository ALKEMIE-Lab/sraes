#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/04/26 13:15:46'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/04/26 13:15:46'


import numpy as np
from pysr import PySRRegressor


def run():
    X = 2 * np.random.randn(100, 5)
    y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    
    
    model = PySRRegressor(
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )
    
    model.fit(X, y)
    print(model)
    
    
if __name__ == '__main__':
    run()
