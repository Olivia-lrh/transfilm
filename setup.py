#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transfilm - AI视频配音系统
生产级别的AI视频翻译和配音管道
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# 读取requirements.txt
def read_requirements():
    requirements = []
    filepath = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name='transfilm',
    version='1.0.0',
    author='Olivia-lrh',
    author_email='',
    description='生产级AI视频配音系统 - 使用Qwen3-ASR、Qwen3-TTS和MiniCPM-o',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/Olivia-lrh/transfilm',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Multimedia :: Video',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'transfilm=cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
