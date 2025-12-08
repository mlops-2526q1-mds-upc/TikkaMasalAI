Generating the docs
----------

Docs are built with [MkDocs](http://www.mkdocs.org/) using the config at `docs/mkdocs.yml`.

Preferred (Makefile targets):

    make docs-build   # builds site to docs/site
    make docs-serve   # serves at http://127.0.0.1:8001 with live reload

Direct MkDocs (alternative):

    uv run mkdocs build --strict -f docs/mkdocs.yml
    uv run mkdocs serve -f docs/mkdocs.yml -a 127.0.0.1:8001
