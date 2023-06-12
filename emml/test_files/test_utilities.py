#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 1.0
__init_date__ = '2023/05/12 09:55:34'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/05/12 09:55:34'

import os
import unittest
from emml.utilities import accord_now_time_create_dir


class Mkdir(unittest.TestCase):
    def test_dirname(self):
        dirname = accord_now_time_create_dir(head_dir='.',
                                             prefix_str='test',
                                             suffix_str='test',
                                             pass_mkdir=False)
        print(dirname)
        self.assertTrue(os.path.exists(dirname))
        os.rmdir(dirname)
        

if __name__ == "__main__":
    unittest.main()
