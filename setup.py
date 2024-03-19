import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve() / "README.md"

# Get the long description from the README file
with here.open() as f:
    long_description = f.read()

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: Implementation :: PyPy
Topic :: Software Development
Topic :: Scientific/Engineering
"""

setup(
    name="secml-torch",
    version="0.1",
    description="SecML-Torch Library",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
        ],
    ),
    include_package_data=True,
    url="https://secml-torch.readthedocs.io/en/latest/",
    license="MIT",
    author="Maura Pintor, Luca Demetrio",
    author_email="maura.pintor@unica.it, luca.demetrio@unige.it",
    install_requires=["torch>=1.4,!=1.5.*", "torchvision>=0.5,!=0.6.*"],
    extras_require={"foolbox": ["foolbox>=3.3.0"], "tensorboard": ["tensorboard"]},
    python_requires=">=3.7",
)
