# Model Hub Zoo Download Implementation

## Status
**Accepted**

Proposed by: Adam Gibson (3rd Jan 2022)


## Context

In order to properly consume models from OmniHub a proper interface
is needed to allow people to download models. Due to the sheer volume
of models out there on the market now a scalable way of interfacing with
various pretrained models from java is needed. 

## Proposal

An interface allowing people to create pretrained model instances
 using a new interface. Note that this will supplant the old dl4j
model zoo. A user will be able to download models via the following interface:

```java
//access the samediff interface to create a resnet18 model
Pretrained.samediff().resnet18(..).create();


```

A user will have 2 interfaces from the Pretrained namespace class: 
samediff() and dl4j().


Similar to the [codegen work](../contrib/codegen-tools/codegen)
this will use poet to generate interfaces that are dynamically generated
from a DSL. This DSL uses kotlin and poet.

We download model configurations and setup code that downloads a model
and instantiates it in the user's code. These models will be loaded locally
using a .omnihub directory downloaded from a pre specified github repo.


Both the directory and the URL of the model zoo will be configurable
using environment variables. These environment variables + default values will be:
```bash
export OMNIHUB_HOME="$USER/.omnihub"
export OMNIHUB_URL="https://media.githubusercontent.com/media/KonduitAI/omnihub-zoo/main"
```
The default directory will be under $HOME/.omnihub/samediff
$HOME/.omnihub/dl4j for downloaded models. 

Note this is the same directory as the python omnihub downloads
specified in [zoo download](./0011%20-%20OmniHub-Zoo%20Download.md)
Samediff and dl4j will be subdirectories for converted models to be downloaded to.

Underneath the covers each Pretrained namespace will point to different
URL sub folders to resolve models from. dl4j() will point to a /dl4j folder
and samediff() will point to a /samediff folder.






## Consequences

### Advantages

* Java will be used to consume models and allow for creation of the model zoo 
repository using git lfs as the base
* Increases the number of models have access to without needing to convert them manually




### Disadvantages
* Workflow for converting models is not completely automated and requires
manual curation
* The tool isn't a complete solution to adding new models as they come out
* Models found are not guaranteed to be converted and may need manual interference