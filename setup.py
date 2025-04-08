from setuptools import setup, Extension
from setuptools import find_packages
import pybind11

ext_modules = [
    Extension(
        'MonteCarlo.engine.monte_carlo',
        sources=['MonteCarlo/engine/monte_carlo.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
    ),
]

setup(
    name='MonteCarlo',
    version='0.1',
    author='Dein Name',
    description='Monte Carlo Library with multiple equity models',
    packages=find_packages(),
    ext_modules=ext_modules,
    zip_safe=False,
)