# Documentation for DL4J SameDiff computation graph engine

Build and serve documentation for SameDiff with MkDocs (install with `pip install mkdocs`)
To generate SameDiff docs into the`keras-import/doc_sources` folder, run

```
python generate_docs.py \
    --project samediff \
    --language java \
    --code ../nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/ \
    --docs_root deeplarning4j.org/samediff/
```

After that, to locally test or build documentation you can do the following:

- `cd samediff`              # Change into project directory.
- `mkdocs serve`             # Starts a [local webserver on port 8000](localhost:8000).
- `mkdocs build`             # Builds a static site in `docs/site` directory.
