from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy

'''
``svd`` 动态链接库的名字
source 源文件
language 默认是c,可以改成c++
incude_dirs gcc -I 寻找头文件的目录
library_dirs gcc -L 寻找库文件的目录
libraries gcc - l 寻找动态链接库
extra_compile_args gcc 额外编译参数
extra_link_args gcc 额外链接参数
'''
setup(ext_modules = cythonize(Extension(
    'svd',
    sources=['svd.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))