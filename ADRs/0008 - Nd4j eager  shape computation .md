# Nd4j eager shape computation

## Status
Accepted

Proposed by: Adam Gibson (11-19-2021)

Discussed with: Paul Dubs

## Context
Nd4j's model import framework often has the need to 
compute  shapes as variables are created.

This is  in order to resolve how to properly
create a graph based on a graph descriptor from another framework
such as tensorflow or pytorch.

This is often called eager mode. This proposal focuses on just eager shape computation 
intended for use in model import. The assumption is that we could 
build on this later for fully eager computation.


## Decision

In order to aid building model import easier,
this proposal is focused on implementing just dynamic shape computation
for use in the model import context.

This will be composed of a few parts:

1. Each outputVariables() call in SDVariable triggers
an Nd4j.getExecutioner().exec(..) call on the relevant operation
to extract out op shapes. It then sets the appropriate shapes
based on the result for each SDVariable field.


2. This will intentionally include dummy calls for control flow ops
such as if, enter, and while. Shapes from these don't matter
beyond knowing the number of outputs.


3. Each SameDiff instance will have an eager mode boolean
that will determine whether this functionality is invoked.
This eager mode variable will be required for some model import use cases.
Usually the model import framework will turn eager on as needed
without the user needing to be involved.


4. Each SameDiff instance will have a separate ArrayHolder
that will be used for looking up ndarrays relevant
to the eager computation. This will not use proper sessions
but instead store that will be used once for computing shapes.





## Discussion
Paul: Originally we would need full eager mode

Adam: We don't need a fully implemented eager mode
just for model import. Full eager mode would mean proper session support,
training support. This would just be incremental shape calculations
for model import.

## Consequences
### Advantages

* Allows more model import flexibility
* Adds a base for real eager mode later on

### Disadvantages

* Adds more complexity to model import with the addition
dynamic shape calculations during model import

* Could be hard to debug if you want to see the full would be imported graph
when a computation is blocking it

* The import workflow has more state attached to it
with the eager array holder attached to each samediff instance
and the need for another flag for turning the feature on/off