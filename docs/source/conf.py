"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import math
import os

import gftool

# -- Project information -----------------------------------------------------

project = 'GfTool'
copyright = '2019, Weh Andreas'
author = 'Weh Andreas'
today_fmt = '%Y-%m-%d'
html_last_updated_fmt = '%Y-%d-%m'
master_doc = 'index'

latex_engine = 'xelatex'  # xelatex can handle Unicode
papersize = 'a4paper'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:  # RTD modifies `conf.py` file resulting in dirty git
    print("We are on RTD!")  # noqa: T201
    dirty, gtver = '.dirty', gftool.__version__
    clean_version = gtver[:-len(dirty)] if gtver.endswith(dirty) else gtver
    gftool.__version__ = clean_version
    release = clean_version
    version = clean_version
else:
    # The short X.Y version.
    version = gftool.__version__
    # The full version, including alpha/beta/rc tags.
    release = gftool.__version__


try:  # clean the version if it gets dirtied e.g. by RTD
    commits = gftool.__version__.split('+', maxsplit=1)[1].split('.', maxsplit=1)[0]
    try:
        commits = int(commits)
    except ValueError:
        pass
    else:
        if not commits:  # this is a tagged version, let's just state the tag
            version = gftool.__version__.split('+', maxsplit=1)[0]
except IndexError:
    pass  # apparently we have already only the tag

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',  # plots in examples
    'sphinx.ext.intersphinx',  # links to numpy, scipy ... docs
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx_toggleprompt',  # toggle `>>>`
    'sphinx.ext.extlinks',  # define roles for links
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
default_role = "autolink"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
# autosummary_imported_members = True

# -----------------------------------------------------------------------------
# NumPy extensions
# -----------------------------------------------------------------------------

numpydoc_use_plots = True
# numpydoc_xref_param_type = True
# numpydoc_xref_ignore = "all"  # not working...
numpydoc_show_class_members = False
# run checks, see https://numpydoc.readthedocs.io/en/latest/validation.html
numpydoc_validation_checks = {
    # "GL01",  # doesn't work with subclasses: NamedTuple.index and the like
    # "GL02",  # doesn't work with subclasses: NamedTuple.index and the like
    "GL03",
    "GL05",
    "GL06",
    "GL07",
    # "GL08",  # magic method __init__ has no docstring
    "GL09",
    "GL10",
    # "SS01",  # magic method __init__ has no docstring
    "SS02",
    # "SS03",  # Attributes cannot be documented for named tuples (inheritance).
    "SS04",
    # "SS05",  # We sometimes start with noun, e.g., Green's function
    "SS06",
    # "PR01",  # For short docstrings we don't document parameters
    "PR02",
    "PR03",
    "PR04",
    "PR05",
    "PR06",
    "PR07",
    "PR08",
    "PR09",
    "PR10",
    # "RT01",  # For short docstrings we don't document the return values
    # "RT02",  # We return sometimes multiple arguments in a single line
    "RT03",
    "RT04",
    "RT05",
    "YD01",
    "SA02",
    "SA03",
}

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
plot_pre_code = """
import warnings

import numpy as np
import matplotlib.pyplot as plt
import gftool as gt

np.random.seed(0)
warnings.filterwarnings(  # ignore warnings in doctest
    action='ignore', category=UserWarning,
    message='Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.'
)
"""
doctest_global_setup = plot_pre_code  # make doctests consistent
doctest_global_cleanup = """
try:
    plt.close()  # close any open figures
except:
    pass
"""
plot_include_source = True
plot_html_show_source_link = False
plot_formats = [('png', 100), 'pdf']

phi = (math.sqrt(5) + 1)/2

plot_rcparams = {
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3*phi, 3),
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,
}

# -----------------------------------------------------------------------------
# Intersphinx
# -----------------------------------------------------------------------------
# taken from https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209
intersphinx_mapping = {'python': ('https://docs.python.org/3.10', None),
                       'scipy': ('http://docs.scipy.org/doc/scipy/', None),
                       'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       'mpmath': ('https://mpmath.org/doc/current/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'numexpr': ('https://numexpr.readthedocs.io/en/stable/', None)
                       }

# -----------------------------------------------------------------------------
# extlinks
# -----------------------------------------------------------------------------
extlinks = {'commit': ('https://github.com/DerWeh/gftools/commit/%s', '%s')}
