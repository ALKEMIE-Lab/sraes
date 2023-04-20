#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__version__ = 0.1
__init_date__ = '2022/10/10 10:52:42'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2022/10/10 10:52:42'
import os
from setuptools import find_packages, setup

NAME = 'emml'
VERSION = '0.0.1'
DESCRIPTION = 'ee by Machine Learning'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE, encoding='UTF8').read()

REQUIREMENTS = ['numpy', 'pandas', 'xlrd==1.2.0', 'openpyxl', 'seaborn', 'pyyaml']
URL = "https://gitee.com/alkemie_gjwang/emml"
AUTHOR = __author__
AUTHOR_EMAIL = __email__
LICENSE = 'MIT'
PACKAGES = find_packages()
# cmdclass = {'sdist': sdist}
PACKAGE_DATA = {}


def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        packages=find_packages(),
        package_data=PACKAGE_DATA,
        include_package_data=True,
        install_requires=REQUIREMENTS,
        cmdclass={},
        zip_safe=False,
        url=URL
    )


if __name__ == '__main__':
    setup_package()

