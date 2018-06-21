# DL4J Keras model-import documentation

Build and serve documentation for Keras model-import with MkDocs (install with `pip install mkdocs`)
The source for Keras documentation is in this directory under `doc_sources/`.

The structure of this project (template files, generating code, mkdocs YAML) is closely aligned
with the [Keras documentation](keras.io) and heavily inspired by the [Keras docs repository](https://github.com/keras-team/keras/tree/master/docs).

- `python generate_docs.py`  # Generates docs into `keras-import/doc_sources` folder.
- `cd keras-import`          # Change into project directory.
- `mkdocs serve`             # Starts a [local webserver on port 8000](localhost:8000).
- `mkdocs build`             # Builds a static site in `docs/site` directory.
