# Graph Execution Trace Collection and Reproduction

## Status
Implemented

Proposed by: Adam Gibson (20 Mar 2023)

Discussed with: Paul Dubs

Finalized by: Adam Gibson (24 Mar 2023)

## Context

Reproducing a specific graph execution between the SameDiff and DL4J APIs can be
challenging, as both use the underlying libnd4j operations to execute code.
Currently, users enable verbose or debug mode in the op executioner to observe
executed operations and manually compare the output of the two APIs. This method is
suboptimal and time-consuming.

In the context of this proposal, the term "vector" refers to an `std::vector` in C++
that stores the metadata of each operation execution. It does not refer to a
mathematical vector or a tensor typically used in deep learning libraries. The
`std::vector` is a dynamic array-like container provided by the C++ Standard Library,
which is used here to store the sequence of operation executions.

## Decision

To improve the process, we will save execution traces in a format that can generate a
SameDiff graph, emulating the executed steps. Once enabled, operation executions will
be collected in a vector, storing only metadata such as input/output shapes and
arguments for each operation. These executions will be stored in the vector
sequentially.

For instance, when executing a convolution operation, we can trigger the scope in C++
to indicate the current operation. This enables tracking the execution of the
convolution operation and its nested operations, like the im2col operation.

Graph tracing can be enabled using the following command:
```java
Nd4j.toggleTrace(true);
```

Using the vector of executions, we can reproduce a graph. To save the graph, use:
```java
SameDiff sd = SameDiff.collectTrace();
sd.save(new File("mygraph.fb"));
```

Afterward, purge the trace to prevent memory leaks:
```java
Nd4j.purgeTrace();
```

When purge is done you can disable trace with:
```java 
Nd4j.toggleTrace(false);
```

## Consequences
### Advantages
* Simplifies graph reproduction
* Enables decomposition of nested op execution, such as attention
* Increases complexity when implementing ops, requiring the developer to notify the tracer of the current parent op

### Disadvantages
* Generated graph may lack names or other metadata
* Execution tracing should be performed only when executing one op at a time