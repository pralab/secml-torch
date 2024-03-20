import pathlib

from setuptools import find_packages, setup

here = pathlib.Path.cwd()
readme_path = here / "README.md"
version_path = here / "src/secmlt" / "VERSION"


# Get the long description from the README file
with readme_path.open() as f:
    long_description = f.read()

# Get the version from VERSION file
with version_path.open() as f:
    version = f.read()


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
    version=version,
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
    data_files=[("src/secmlt/VERSION", ["src/secmlt/VERSION"])],
    include_package_data=True,
    url="https://secml-torch.readthedocs.io/en/latest/",
    license="MIT",
    author="Maura Pintor, Luca Demetrio",
    author_email="maura.pintor@unica.it, luca.demetrio@unige.it",
    install_requires=["torch>=1.4,!=1.5.*", "torchvision>=0.5,!=0.6.*"],
    extras_require={"foolbox": ["foolbox>=3.3.0"], "tensorboard": ["tensorboard"]},
    python_requires=">=3.7",
)
