from setuptools import setup, find_packages

setup(
    name="ocr-transformer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'numpy>=1.19.0',
    ],
) 