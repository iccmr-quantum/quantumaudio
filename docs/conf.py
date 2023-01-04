# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quantumaudio'
copyright = '2022, Paulo Vitor Itaboraí, ICCMR'
author = 'Paulo Vitor Itaboraí'
release = '0.0.2'
version = '0.0.2rc2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "nbsphinx", "nbsphinx_link", "sphinx.ext.mathjax", "IPython.sphinxext.ipython_console_highlighting"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Monkey-patch autosummary template context
#from sphinx.ext.autosummary.generate import AutosummaryRenderer


#def smart_fullname(fullname):
#    parts = fullname.split(".")
#    return ".".join(parts[1:])


#def fixed_init(self, app, template_dir=None):
#    AutosummaryRenderer.__old_init__(self, app, template_dir)
#    self.env.filters["smart_fullname"] = smart_fullname


#AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
#AutosummaryRenderer.__init__ = fixed_init
