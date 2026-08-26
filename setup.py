from setuptools import find_packages
from setuptools import setup


setup(
    name="customer-segmentation-training",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "google-cloud-bigquery",
        "google-cloud-storage",
        "db-dtypes",
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)