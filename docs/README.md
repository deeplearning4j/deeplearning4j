# DL4J auto-generated documentation

Run `./gen_all_docs.sh` to generate documentation from source for all supported projects,
then change into the subdirectory of a project and run `mkdocs serve` to see the respective documentation
at `http://localhost:8000`.

To deploy a new version of documentation, first make sure to set `$DL4J_DOCS_DIR` to your local copy of 
https://github.com/deeplearning4j/deeplearning4j-docs. Then run `./copy-to-dl4j-docs.sh`. This puts documentation
into the right folders and you can create a PR from there to change docs.

The structure of this project (template files, generating code, mkdocs YAML) is closely aligned
with the [Keras documentation](keras.io) and heavily inspired by the [Keras docs repository](https://github.com/keras-team/keras/tree/master/docs).

## Testing with Mkdocs

Although Mkdocs is not actually used for serving the Deeplearning4j website, you can locally test the documentation with Mkdocs (`pip install mkdocs`).

After installation, to locally test or build documentation modules by doing the following:

- `cd keras-import`          # Change into project directory.
- `mkdocs serve`             # Starts a [local webserver on port 8000](localhost:8000).
- `mkdocs build`             # Builds a static site in `docs/site` directory.
