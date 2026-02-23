#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

install_requires = ['pandas', 'scipy', "scikit-learn"
                    'torch>=2.2.0', "optuna", 'cv2', 'matplotlib']
setup_requires = []
tests_require = []

classifiers = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
]

setup(
    name="foundwsr",
    version="0.0.1",
    author="BUPT-GAMMA LAB",
    author_email="yaoqiliu@bupt.edu.cn",
    maintainer="Yaoqi Liu",
    license="Apache-2.0 License",
    description="A stable version of the repository for deep learning models for wireless signal recognition",
    url="https://github.com/BUPT-GAMMA/FoundWSR",
    download_url="https://github.com/BUPT-GAMMA/FoundWSR",
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=classifiers
)