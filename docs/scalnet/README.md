# ScalNet documentation

Build and serve documentation for ScalNet with MkDocs (install with `pip install mkdocs`), built into `doc_sources/`.

To generate docs into the`scalnet/doc_sources` folder, run

```
python generate_docs.py \
    --project scalnet \
    --language scala \
    --code ../scalnet/src/main/scala/org/deeplearning4j/scalnet
```

After that, to locally test or build documentation you can do the following:

- `cd scalnet`          # Change into project directory.
- `mkdocs serve`        # Starts a [local webserver on port 8000](localhost:8000).
- `mkdocs build`        # Builds a static site in `docs/site` directory.
