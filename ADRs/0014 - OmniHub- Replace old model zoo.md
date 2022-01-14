# Replace old model zoo

## Status
**Accepted**

Proposed by: Adam Gibson (3rd Jan 2022)


## Context

The deeplearning4j-zoo module has been around for a long time
and provides a few models out of the box. It also relies on manually implementing models
and does not allow deeplearning4j to benefit from the innovation happening in the space.

There is also no samediff support in this current model zoo.


## Proposal
Replace the model zoo with omnihub. Migrate existing models
from the azure hosting to the omnihub github repo at:
https://github.com/KonduitAI/omnihub-zoo

This will allow users to selectively download pretrained models
from the deeplearning4j zoo.


Redirect all URL calls in the old zoo to the new one
to prevent disruption of user's workflows.

Extend deeplearning4j-zoo's ZooModel to support the new
omnihub zoo. 

Provide a bridge interface to ZooModel with OmniHubZooModel
allowing for seamless transition and expansion of new models.

Migrate off of azure storage for the models saving costs
and reducing complexity.



## Consequences

### Advantages

* Reduced maintenance cost
* Increased model availability for users
* Allows for support of samediff
* Reuse existing model zoo as is but increase support for new models 




### Disadvantages
* Potential bugs
* New infra means manually uploading new models
* Need to ensure smooth migration of ZooModel interface to seamlessly
work with the new model zoo