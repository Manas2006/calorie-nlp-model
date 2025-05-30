"""Setup script for the calorie_nlp package."""
from setuptools import setup, find_packages

setup(
    name="calorie_nlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm>=4.62.0",
        "tomli>=2.0.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.9",
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning model for predicting calories from food names",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/calorie-prediction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
) 