from setuptools import setup, find_packages

setup(
    name="astra-guardian",
    version="0.1.0",
    description="AI-Powered Network Security Application",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "tensorflow",
        "jupyter",
        "kagglehub",
        "matplotlib",
        "seaborn",
        "pytest",
        "joblib",
        "h5py",
        "psutil",
    ],
    python_requires=">=3.8",
)

