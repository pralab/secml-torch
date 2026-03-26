import os
import sys

# Add the library path: 'conf.py' lives in 'docs/', source is in 'src/'
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

# Project information
project = "SecML-Torch"
author = "Maura Pintor, Luca Demetrio"
copyright = f"2024, {author}"

# Read version from the VERSION file
version_path = os.path.join(os.path.dirname(__file__), "..", "src", "secmlt", "VERSION")
with open(version_path) as f:
    release = f.read().strip()
version = release

# General configuration 
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

master_doc = "intro"  # Jupyter Book root document

#  BibTeX 
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# autosectionlabel 
autosectionlabel_prefix_document = True

# Options for HTML output 
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_favicon = "_static/assets/logos/logo_icon.ico"
html_logo = "_static/assets/logos/logo.png"
html_css_files = ["css/custom.css"]

html_theme_options = {
    "repository_url": "https://github.com/pralab/secml-torch",
    "path_to_docs": "docs",
    "repository_branch": "master",
    "use_issues_button": True,
    "use_repository_button": True,
    "logo_only": True,
}

# Google Analytics 
html_context = {
    "google_analytics_id": "G-JTGZVQFCJ3",
}

# Options for LaTeX / PDF output 
latex_engine = "pdflatex"

latex_documents = [
    (
        master_doc,
        "book.tex",
        "SecML-Torch",
        "Maura Pintor, Luca Demetrio",
        "manual",
    ),
]

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    # Prevent build failure from missing index file
    "printindex": "",
}

# MyST-NB 
nb_execution_mode = "force"
nb_execution_timeout = 1000
nb_output_stderr = "remove"

# Autodoc 
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

# Napoleon 
napoleon_google_docstring = True
napoleon_numpy_docstring = True