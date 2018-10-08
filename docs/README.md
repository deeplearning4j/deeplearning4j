# DL4J auto-generated documentation

## Building

Run `./gen_all_docs.sh` to generate documentation from source for all supported projects. For each documentation module, files will be put into a `doc_sources` folder where they are staged for copying to the primary docs repository.

To deploy a new version of documentation, first make sure to set `$DL4J_DOCS_DIR` to your local copy of 
https://github.com/deeplearning4j/deeplearning4j-docs and set `$DL4J_VERSION` to a URI-friendly version string such as `v100-RC` (note the lack of decimals). Then run `./copy-to-dl4j-docs.sh`. This puts documentation
into the right folders and you can use `git` to create a PR and update the live docs.

The structure of this project (template files, generating code, mkdocs YAML) is closely aligned
with the [Keras documentation](keras.io) and heavily inspired by the [Keras docs repository](https://github.com/keras-team/keras/tree/master/docs).

## File structure

Each major module or library in Eclipse Deeplearning4j has its own folder. Inside that folder are three essential files:

- `templates/`
- `pages.json`
- `README.md`

Note that the folder names don't exactly match up with the modules in the `pom.xml` definitions across DL4J. This is because some of the documentation is consolidated (such as DataVec) or omitted due to its experimental status or because it is low-level in the code.

Templates must maintain a flat file structure. This is to accommodate Jekyll collections when the docs are published. Don't worry about having similarly named files in different doc modules - the module name is prepended when the docs are generated.

## Creating templates

Each template has a Jekyll header at the top:

```markdown
---
title: Deeplearning4j Autoencoders
short_title: Autoencoders
description: Supported autoencoder configurations.
category: Models
weight: 3
---
```

All of these definitions are necessary. 

- `title` is the HTML title that appears for a Google result or at the top of the browser window.
- `short_title` is a short name for simple navigation in the user guide.
- `description` is the text that appears below the title in a search engine result.
- `category` is the high-level category in the user guide.
- `weight` is the ordering that the doc will appear in navigation, the larger the lower the listing.

## Creating links

**All links to other docs need to be relative.** This prolongs the life of the documentation and reduces maintenance. The basic structure of a link to another doc looks like:

```
<module name>-<file name>
```

So if you created a DataVec doc with the name `iterators.md` in the `datavec` module, your relative link will look like:

```
./datavec-iterators
```

Note the omission of the file extension `.md`. Jekyll automatically generates a clean URL for us to use.