# DataVec auto-import documentation

Build and serve documentation for DataVec with MkDocs (install with `pip install mkdocs`)
The source for Keras documentation is in this directory under `doc_sources/`.

The structure of this project (template files, generating code, mkdocs YAML) is closely aligned
with the [Keras documentation](keras.io) and heavily inspired by the [Keras docs repository](https://github.com/keras-team/keras/tree/master/docs).

To generate docs into the`keras-import/doc_sources` folder, run

```
python generate_docs.py \
    --project keras-import \
    --code ../deeplearning4j/deeplearning4j-modelimport/src/main/java/org/deeplearning4j/nn/modelimport/keras/

```

After that, to locally test or build documentation you can do the following:

- `cd keras-import`          # Change into project directory.
- `mkdocs serve`             # Starts a [local webserver on port 8000](localhost:8000).
- `mkdocs build`             # Builds a static site in `docs/site` directory.
