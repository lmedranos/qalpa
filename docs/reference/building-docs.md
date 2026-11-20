# Building these docs

The documentation is written in Markdown ([MyST](https://myst-parser.readthedocs.io/)) and built with [Sphinx](https://www.sphinx-doc.org/) and the [Furo](https://pradyunsg.me/furo/) theme. [Read the Docs](https://about.readthedocs.com/) builds it from the repository on every push.

## Files

| Path | Purpose |
|---|---|
| {repo}`.readthedocs.yaml` | Read the Docs build: OS, Python version, Sphinx config, requirements |
| {repo}`docs/conf.py` | Sphinx configuration: extensions, theme, the `{repo}` link role |
| {repo}`docs/requirements.txt` | Packages for the build; QALPA itself is not installed |
| `docs/**/*.md` | The pages; the sidebar order is set by the `toctree` blocks in `docs/index.md` |

## Build locally

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -r docs/requirements.txt
.venv-docs/bin/sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. Add `-W --keep-going` to treat warnings, such as broken cross-references, as errors.

## Writing conventions

- **Links to repository files:** use the `repo` role, for example ``{repo}`global_config.yaml` `` or ``{repo}`al.yaml <src/configs/al.yaml>` ``. On Read the Docs, it links to the branch or tag being built.
- **Links between pages:** use relative Markdown links, with a heading anchor where needed: `[Hit rule](../user-guide/kde-planning.md#closing-the-loop-the-hit-rule)`.
- **Shipped files:** include them with `literalinclude` instead of copying them, so the docs stay in sync with the code (see the full-file dropdowns in [Configuration](../configuration/index.md)).
- **Diagrams:** use a `{mermaid}` block; it is rendered in the browser.
- **Images with a dark variant:** add two images with the classes `only-light` and `only-dark`.

## Set up the project on Read the Docs

1. Sign in at [app.readthedocs.org](https://app.readthedocs.org/) with the GitHub account that can access the repository.
2. **Add project** → choose the repository. Read the Docs finds `.readthedocs.yaml` in the repository root.
3. Under **Settings → Versions**, activate the branches to publish (by default only `latest`, the default branch).
4. Optional: under **Settings → Pull request builds**, turn on previews for pull requests.
