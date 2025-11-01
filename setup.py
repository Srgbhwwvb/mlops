from setuptools import setup, find_packages

setup(
    name="plant-classification",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "Pillow>=8.3.0",
        "tqdm>=4.60.0",
        "PyYAML>=5.4.0",
    ],
    python_requires=">=3.8",
)
