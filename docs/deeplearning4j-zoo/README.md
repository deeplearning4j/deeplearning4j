# deeplearning4j-zoo documentation

Build and serve documentation for DataVec with MkDocs (install with `pip install mkdocs`)
The source for Keras documentation is in this directory under `doc_sources/`.

The structure of this project (template files, generating code, mkdocs YAML) is closely aligned
with the [Keras documentation](keras.io) and heavily inspired by the [Keras docs repository](https://github.com/keras-team/keras/tree/master/docs).

To generate docs into the `deeplearning4j-zoo/doc_sources` folder, first `cd docs` then run:

```shell
python generate_docs.py \
    --project deeplearning4j-zoo \
    --code ../deeplearning4j
	--out_language en
```