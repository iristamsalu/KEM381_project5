from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='md simulation cython version',
    ext_modules=cythonize("md_sim.pyx"),
    include_dirs=[np.get_include()],
)