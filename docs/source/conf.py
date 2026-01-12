# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys, os, toml
# Parse pyproject.toml to get the release version
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.abspath(os.path.join(project_root, 'luminet')))
pyproject_path = os.path.join(project_root, 'pyproject.toml')
with open(pyproject_path, 'r') as f:
    pyproject_data = toml.load(f)
    release = pyproject_data['project']['version']
    version = release
project = 'Luminet'
author = 'Bjorge Meulemeester, J. P. Luminet'
copyright = '2025, Bjorge Meulemeester'


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    # 'sphinx.ext.autosummary',  # Create neat summary tables 
    "autoapi.extension",       # Generate API documentation from code
    'sphinx.ext.napoleon',     # Preprocess docstrings to convert Google-style docstrings to reST
    'sphinx_paramlinks',       # Parameter links
    'sphinx.ext.todo',         # To-do notes
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',  # Link to other project's documentation
    'sphinxcontrib.bibtex',    # For citations
    'sphinx.ext.mathjax',      # For math equations
    'sphinx_copybutton',       # For copying code snippets
    'sphinx_inline_tabs',      # For inline tabs
    # 'sphinxext.opengraph',   # For OpenGraph metadata, only enable when the site is actually hosted. See https://github.com/wpilibsuite/sphinxext-opengraph for config options when that happens.
]


autoapi_own_page_level = "method"
autoapi_type = "python"
autoapi_keep_files = True
autoapi_add_toctree_entry = False  # we use a manual autosummary directive in api_reference.rst thats included in the toctree
autoapi_generate_api_docs = True
# generate the .rst stub files. The template directives don't do this. 
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
]
autoapi_dirs = ['../../luminet']
autoapi_template_dir = "../_templates"
templates_dir = ["../_templates"]
toc_object_entries_show_parents = 'hide'  # short toc entries

default_domain = "python"
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}


# -- bibtex files ----------------------------------------------------------
bibtex_bibfiles = ['bibliography.bib']

# -- Napoleon settings -----------------------------------------------------
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False  # use a single ":parameters:" section instead of ":param arg1: description" for each argument
napoleon_use_rtype = False  # if True, separate return type from description. otherwise, it's included in the description inline
napoleon_preprocess_types = False  # otherwise custom argument types will not work
napoleon_type_aliases = None
napoleon_attr_annotations = True

## Include Python objects as they appear in source files
## Default: alphabetically ('alphabetical')
# autodoc_member_order = 'bysource'

## Generate autodoc stubs with summaries from code
paramlinks_hyperlink_param = 'name'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False  # less verbose for nested packages

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
sys.path.append(os.path.abspath("../_pygments"))
pygments_style = 'style.LightStyle'
pygments_dark_style = 'material'  # furo specific

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo" 

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_static_path = ['../../assets', '../../docs/_static/_images/']
html_theme_options = {
    "dark_logo": "isofluxlines_white.png",
    "light_logo": "isofluxlines_black.png",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#000000",  # black instead of blue
        "color-foreground-secondary": "#797979",  # slightly more muted than default
        "color-sidebar-background": "#F2F1ED",
        "color-sidebar-item-background--hover": "#E5E3DC",
        "color-background-hover": "#E5E3DC",
        "color-highlight-on-target": "#FDF8EB",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fefaee",  # Off-white
        "color-foreground-primary": "#E6E1D4",
        "color-brand-content": "#FFB000",  # Gold instead of dark blue
        "color-sidebar-background": "#1A1C1E",
        "color-sidebar-item-background--hover": "#1e2124",
        "color-link": "#FFC23E",
        "color-highlight-on-target": "#2D222B",
        "color-link--visited": "#a58abf",
    },
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_css_files = [
    'default.css',  # relative to html_static_path defined above
]


# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.

## I don't like links to page reST sources
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
