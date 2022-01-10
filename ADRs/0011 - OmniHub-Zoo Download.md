# Model Hub Zoo - Download

## Status
**Discussion**

Proposed by: Adam Gibson (1st Jan 2022)


## Context

Model zoos or hubs are web services from different vendors to provide
a venue for researchers and engineering teams who want to open source their
work to publish models. The use case is typically to finetune models.

Finetuning models means adapting a model trained for one task to generalize for another
by replacing its objective.

Coupling this with binary file formats, distributing model files typically happens
as large binary files + some optional metadata. In the case of tensorflow and onnx
it's using protobuf. With pytorch it uses python pickle archives.

Model hubs provide SDKs for downloading and using models within python.




## Proposal

The goal is to interop with these model hubs using an integrated python library
and add the appropriate tooling for converting these models to something consumable
by the model import framework.


A user is able to download models with a standard python interface using
ModelHub. A ModelHub implementation might look like:
```python
class ModelHub(object):
    def __init__(self):
       self.hub_url = ''
       self.framework_name = ''
     
    def download_model(self,path: string):
       ...
     
    def stage_model(self,model):
        ...
```




The storage type will be specific to the model hub. The concrete functions
enums like this have is to specify to the underlying web service what kind of model we want.



In order to load a model, a model hub tends to provide different ways of
downloading a model. This can be via a compressed archive or uncompressed.

In order to use this we need to be able to specify the access type.
This will be an enum such as:

```python
enum  StorageType {
    COMPRESSED,UNCOMPRESSED
}
```
Loading a model will be done using either samediff or deeplearning4j.
This will leverage and extend the existing work in the model import
work built previously.


### Storage directory

Every model downloaded by this interface will be stored in an uncompressed
format (.onnx,.pb,..) files with their original names under a standard unified directory
separated by framework. This ensures ease of use and debugging
in case a user wants to directly import a model or view
it in a model viewer like netron.

This directory will default to a .modelhub directory under $USER.
A user can also override this directory with a MODELHUB_PATH
environment variable.

Note that the models will be stored as duplicates copied under
the $MODELHUB_PATH. The reason for this is to preserve
the original framework's underlying model in case a user
needs to work around bugs or needs to work with the underlying
libraries in a separate environment. 

This also has the added  benefit of allowing a previous model cache from the underlying
libraries to avoid download. In this case, models will just be 
stored under the $MODELHUB_PATH in the form they need to be in
to work with the model import framework.


### Staging models

Every model will be downloaded by its original SDK and then preprocessed.
We will call this staging. Staging is a secondary step that takes
each model and adjusts it to be its end form that can be worked with.

For example in tensorflow, we may need to freeze models first
before storing them for use with import.


## Consequences

### Advantages
* Allow interop of different ecosystems
* Provide a foundation for finetuning models (finetuning models is out of the scope of this ADR)
* Provide a way to download and manage models for testing model import functionality
* Provide a way to enhance the built in dl4j model zoo by importing models from other ecosystems
in a standardized way
* Unified way of downloading and accessing models for multiple frameworks
* Standardized directory allowing models to be easily worked with.
* Keeps underlying framework models around for debugging. Also prevents underlying libraries
from re-downloading libraries and benefiting from already downloaded models
if a user uses the underlying model libraries elsewhere.



### Disadvantages
* More complexity in maintaining an ongoing SDK for downloading and maintaining models in different ecosystems
* Potential storage complexity involved in running and maintaining/testing models
* No way to know when a model hub changes
* Loading and importing models from different ecosystems can be messy. Additional work may need to be done per model
in order to make them usable. That additional work is out of the scope of this ADR.
* More storage on a user's system due to the secondary cache
