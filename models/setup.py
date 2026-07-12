from setuptools import setup, find_packages

setup(
    name='mda_aenmf_ad',
    version='1.0.0',
    description='Metabolite-Disease Association Prediction for Autoimmune Disorders',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.1.0',
        'torch-geometric>=2.4.0',
        'numpy>=1.24.4',
        'pandas>=2.1.1',
        'scipy>=1.11.3',
        'scikit-learn>=1.3.2',
        'matplotlib>=3.7.2',
        'seaborn>=0.12.2',
        'openpyxl>=3.1.2',
        'networkx>=3.1',
        'tqdm>=4.66.1',
        'pyyaml>=6.0.1',
    ],
    extras_require={
        'dev': ['pytest>=7.4', 'black', 'flake8'],
        'chem': ['rdkit>=2023.3.1'],
    },
)
