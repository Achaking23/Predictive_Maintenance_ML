from setuptools import setup, find_packages

setup(
    name="predictive_maintenance_ml",
    version="1.0.0",
    description="End-to-end ML pipeline for industrial predictive maintenance",
    author="ML Pipeline Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.4",
        "imbalanced-learn>=0.12",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "pyyaml>=6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)