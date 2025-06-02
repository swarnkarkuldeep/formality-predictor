from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="formality-predictor",
    version="0.1.0",
    author="Kuldeep Swarnkar",
    author_email="kuldeepswarnakr14@gmail.com",
    description="A machine learning model to predict text formality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/formality-predictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy==1.24.3',
        'pandas==2.0.3',
        'scikit-learn==1.3.0',
        'torch==2.0.1',
        'transformers==4.30.2',
        'spacy==3.6.0',
        'en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0.tar.gz',
        'tqdm==4.65.0',
        'joblib==1.3.1',
        'tensorflow==2.13.0',
        'h5py==3.9.0',
        'grpcio==1.56.0',
    ],
    include_package_data=True,
)
