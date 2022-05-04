# Dynamic Indexing

## Status
**Discussion**

Proposed by: Adam Gibson (30th April 2022)


## Context

Dynamic indexing has a wide variety of applications for building deep learning graphs.

Expressing a dynamic index that gets resolved at runtime entails specifying a negative index as the desired index for an indexing operation.
At runtime, the indexing engine will then count backwards from the element the user specified. An example being: -1 starts at the end, -2 starts at the second to last.

Samediff and nd4j have numpy style indexing support.


Samediff does this 1 of 2 ways. Either through nd4j's indexing engine building on [previous work](./0021%20-%20Create%20View.md) or using a similar interface to nd4j's indexing engine to build a strided slice operation call.
Previously, the indexing would not allow initialization with a negative index without passing in an ndarray. The problem with this is ndarrays are not known in samediff till execution time.



## Proposal


We support the ability to dynamically resolve indices during execution. This happens as follows:
1. Each index  has a concept of initialized().
   This returns a boolean indicating whether the operation was initialized or not. Initialized  means there are no negative values present in the index and a specific boolean flag has been set representing the index being fully initialized.
2. If a negative index is specified, the index is set on the index itself but it will not be considered initialized.
3. At runtime, upon use a deep copy of the index happens and then initialization will occur upon use relative to the ndarray using it. This currently happens at the java level. Indexing does not exist fully at the c++ level.


## Consequences

### Advantages

* Allows for more flexible indexing only possible with just in time resolution.
* Less errors are thrown during indexing


### Disadvantages
* A bit more overhead in the indexing process
* Harder to debug