# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys, os

sys.path.insert(0, os.path.abspath("../src/secmlt"))

project = "SecML-Torch"
copyright = "2024, Maura Pintor, Luca Demetrio"
author = "Maura Pintor, Luca Demetrio"
release = "v0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["*tests*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Add readme and contribution guide -------------------------------------------------

import pathlib

for m in ["Readme.md", "CONTRIBUTING.md"]:
    source_path = pathlib.Path(__file__).parent.resolve().parent.parent / m
    target_path = pathlib.Path(__file__).parent / m.lower().replace(".md", ".rst")
    from m2r import convert

    with target_path.open("w") as outf:  # Change the title to "Readme"
        outf.write(convert(source_path.read_text()))
