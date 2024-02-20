from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="secml2",
    version="1.0",
    description="SecML 2.0 Library",
    long_description=long_description,
    long_description_context_type="text/markdown",
    package_dir={"": "secml2"},
    packages=find_packages(where="secml2"),
    url="",
    license="MIT",
    author="Maura Pintor, Luca Demetrio",
    author_email="maura.pintor@unica.it, luca.demetrio@unige.it",
    install_requires=[],
    extras_require={
        "foolbox": ["foolbox>=3.3.0", "torch>=1.4,!=1.5.*", "torchvision>=0.5,!=0.6.*"],
    },
    python_requires=">=3.7",
)
