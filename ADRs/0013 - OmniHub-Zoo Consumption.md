# Model Hub Zoo Download Implementation

## Status
**Discussion**

Proposed by: Adam Gibson (3st Jan 2022)


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

* Greatly strengthens our ability to test and execute models
needed for different use cases
* Allows us to be flexible in what we can enable users to do with
our framework as a starting point
* Further, builds out the work from downloading models and greatly increases
the testing allowed for our model import framework



### Disadvantages
* Different work is needed for each ecosystem
* APIs may change and need to be updated
* Just pre-processing models does not mean they are guaranteed to be imported. Additional
  work will need to be done on model import to allow models to execute. This comes with additional validation work.
* Not a comprehensive solution, users will still need to know things like the inputs and outputs
and may still need to refer to the underlying docs for a given model to use effectively.
* A user may still need to understand how to pre-process different kinds of models