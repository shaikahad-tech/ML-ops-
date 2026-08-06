from setuptools import setup, find_packages

setup(
    name="mlops-churn",
    version="0.1.0",
    description="Production-style MLOps project for customer churn prediction.",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3,<2.0",
        "pandas>=2.0,<3.0",
        "numpy>=1.24,<3.0",
        "mlflow>=2.8,<3.0",
        "fastapi>=0.104,<1.0",
        "uvicorn[standard]>=0.24,<1.0",
        "pydantic>=2.5,<3.0",
        "pyyaml>=6.0,<7.0",
    ],
    entry_points={
        "console_scripts": [
            "mlops=run:main",
        ],
    },
)
