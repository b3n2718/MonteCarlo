from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "monte_carlo",
        ["monte_carlo.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="monte_carlo",
    ext_modules=ext_modules,
)
