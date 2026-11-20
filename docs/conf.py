"""Sphinx configuration for the QALPA documentation (built on Read the Docs)."""

import os

from docutils import nodes
from sphinx.util.nodes import split_explicit_title

# -- Project -----------------------------------------------------------------

project = "QALPA"
author = "Michael Hanna, Julian Cremer, Zekiye Erarslan, Leonardo Medrano Sandonas"
copyright = f"2026, {author}"

DEFAULT_REPO_URL = "https://github.com/lmedranos/qalpa"


def _repo_url() -> str:
    """GitHub page of the repository the docs are built from.

    On Read the Docs, this is the repository the project was imported from (a fork, for example);
    locally, it is the main QALPA repository.
    """
    url = os.environ.get("READTHEDOCS_GIT_CLONE_URL", "").strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url[len("git@github.com:"):]
    if not url.startswith("https://github.com/"):
        return DEFAULT_REPO_URL
    return url.removesuffix(".git").rstrip("/")


REPO_URL = _repo_url()


def _git_ref() -> str:
    """Branch, tag or commit that source-file links point to.

    On Read the Docs, links follow the version being built; locally they point to main.
    Pull-request builds use the commit, because the PR number is not a git ref.
    """
    if os.environ.get("READTHEDOCS_VERSION_TYPE") == "external":
        return os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "main")
    return os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main")


REPO_REF = _git_ref()

# -- General -----------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3

# Copy only the command, not the shell prompt or the output.
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "QALPA"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": f"{REPO_URL}/",
    "source_branch": REPO_REF,
    "source_directory": "docs/",
    "navigation_with_keys": True,
}


# -- {repo}`path` role: link to a file or folder in the GitHub repository ------


def _repo_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """``{repo}`global_config.yaml``` or ``{repo}`title <src/configs/al.yaml>```."""
    _, title, target = split_explicit_title(text)
    url = f"{REPO_URL}/blob/{REPO_REF}/{target.lstrip('/')}"
    ref = nodes.reference(rawtext, "", refuri=url, **(options or {}))
    ref += nodes.literal(title, title)
    return [ref], []


def setup(app):
    app.add_role("repo", _repo_role)
