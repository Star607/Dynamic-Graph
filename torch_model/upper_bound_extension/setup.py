from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='upper_bound_cpp',
      ext_modules=[cpp_extension.CppExtension('upper_bound_cpp',
                                              ['upper_bound.cpp'])],
      #     extra_compile_args=["-fopenmp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
