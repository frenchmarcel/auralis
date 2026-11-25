import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

project = "Auralis"
author = "Auralis contributors"
copyright = "2026, Auralis contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
# the project journal is in French and stays outside the docs
exclude_patterns = ["_build", "*.fr.md", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "Auralis documentation"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}

autodoc_member_order = "bysource"
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "undoc-members": False}
# the native C++ and CUDA libraries are not built on Read the Docs
os.environ.setdefault("AURALIS_GPU", "0")
