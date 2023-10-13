# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'DRAGON'
copyright = '2023, Julie Keisler'
author = 'Julie Keisler'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'nbsphinx'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_logo = 'dragon_logo.png'
html_theme_options = {
    "repository_url": "https://github.com/ThomasFirmin/zellij",
    "use_repository_button": True,
    "collapse_navigation": False,
    "logo_only": True,
    "extra_navbar": f"<p>Version: {release}</p>",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

autodoc_member_order = "bysource"



# -- Options for EPUB output
epub_show_urls = 'footnote'
