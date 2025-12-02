from setuptools import setup

setup(
    name="pytorch_rl2",
    py_modules=["rl2"],
    version="2.0.0",
    install_requires=[
        'mpi4py==3.0.3',
        'torch==1.8.1'
    ]
)
