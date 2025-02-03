from setuptools import setup, find_packages

setup(
    name="aipi540-groupa1",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'jupyter>=1.0.0',
    ],
) 