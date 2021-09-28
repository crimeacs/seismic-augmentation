import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['seismic-augmentation']

setup(
    name='seismic-augmentation',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/IMGW-univie/seismic-augmentation',
    license='MIT License',
    author='Artemii Novoselov',
    author_email='artemii.novoselov@univie.ac.at',
    description='Pytorch library for seismic data augmentation'
)


