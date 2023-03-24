# UDFs

## Status

Implemented

Proposed by: Adam Gibson (20 Mar 2023)

Discussed with: Paul Dubs

Finalized by: Adam Gibson (24 Mar 2023)


## Context

It can be challenging to reproduce a specific graph execution between the different APIs, SameDiff and DL4J, as they both use the underlying
libnd4j operations to execute code. By enabling verbose or debug mode in the op executioner,
we can observe the executed operations.

We often need to compare the execution of a graph between the two APIs. This is currently
done by enabling verbose mode and observing the output. This is not ideal since it requires
the user to manually compare the output of the two APIs.




## Proposal
We can now save execution traces in a format that can be used to generate a SameDiff graph that emulates the executed steps.
Once enabled, operation executions are collected in a vector. These executions only store metadata, such as input and output shapes and arguments for each operation. The executions are stored in a vector in sequence.


For example, if we have a graph with a convolution operation, we can trigger the scope in C++ to indicate that we are currently in
a convolution operation.
This allows us to track the execution of the convolution operation and its nested operations, such as the im2col operation.

Using this vector of executions, we can reproduce a graph. To save the graph, use:

```java
SameDiff sd = SameDiff.collectTrace();
sd.save(new File("mygraph.fb"));
```


Once we are done, purge the trace to avoid memory leaks:

```java
Nd4j.purgeTrace();
```


## Consequences

### Advantages

* Allows easier reproduction of a given graph
* Allows decomposition of nested op execution like with attention
* Adds complexity when implementing ops since we need to remember to trigger
   notify the tracer we are currently in a parent op.

### Disadvantages
* Graph generated may be missing names or other metadata
* Execution tracing should only be done when executing one  op at a time.
