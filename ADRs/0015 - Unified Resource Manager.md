# Replace old model zoo

## Status
**Discussion**

Proposed by: Adam Gibson (12th Jan 2022)

TODO:
1. centralized MD5 sum directory
2. List of directories stored in a local .dl4jresources config file
3. Configuration file format for listing directories and their types
4. Additional checks for old directories at default values not covered by newer support
5. Pre cataloging based on default dataset directories found from prior releases
6. 


## Context

A number of current downloaders exist for various resources
deeplearning4j needs to function. These include the following:
1. Strumpf resource resolver (manages test resources)
and relies on azure. The original code for strumpf can be found [here](https://github.com/KonduitAI/strumpf)
2. Deeplearning4j model zoo (the legacy model zoo)
3. Omnihub: The new model zoo replacing #2.
4. Deeplearning4j datasets: dataset download for various datasetiterators


These have accumulated over the years and have made maintenance of download related logic
complex.

Relevant ADRs include:
[Omnihub zoo download](./0011%20-%20OmniHub-Zoo%20Download.md)
[Omnihub zoo download implementations](./0012%20-%20OmniHub-Zoo%20Download%20Implementations.md)
[Omnihub replace old model zoo](./0013%20-%20OmniHub-Zoo%20Consumption.md)


## Proposal

All resources are hosted on  github LFS.
A  resource abstraction for binding the  various resource types in
to 1 abstraction and downloader.

A Resource  is how we handle this. It is be aware of the following concepts:
1. A base url for downloading a file
2. A cache directory for managing the resource
3. Common download + retry logic for ensuring a download succeeds


A Resource manages a remote resource like a file. Similar to the current resource types
in deeplearning4j-common. These resources are mostly be stored on git LFS.

As part of this introduction of a unified resource abstraction
is cache aware exposing the cache so users can delete if they wish.

For existing datasets we  use the old sources but have a common abstraction
for knowing which dataset we want to download.

Another problem is file verification.

The legacy model zoo uses simpler adler checksums for verification.
Some download cache verification implementations use md5sum.

We  use md5sum and standardize this for all resources.


Note that in order to avoid maintenance burdens md5 checksum verification
is optional. By default, if a resource returns null or an empty
string verification is not  performed. This distinction is important
for resource types such as test resources vs end user assets like pretrained model
weights.

This is also important for compatibility. Due to the legacy checksum
verification in the zoo module, md5 checksum verification can come later.

This leads us to 5 resource types:
1. Omnihub: The omnihub pretrained models
2. Datasets: the legacy datasets for custom iterators like mnist and lfw
3. Dl4j zoo: The legacy zoo models
4. Strumpf: the legacy test resource manager
5. Custom: custom resources where a user can specify a URL and file destination

## Consequences

### Advantages

* Reduced costs by migrating to github
* One module for handling resources
* Replaces legacy abstractions while preserving backwards compatibility
* Allows easier management of local resources


### Disadvantages
* Potential bugs
* Migration will take time