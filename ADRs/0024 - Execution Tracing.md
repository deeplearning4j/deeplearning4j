# UDFs

## Status

Implemented

Proposed by: Adam Gibson (20 Mar 2023)

Discussed with: Paul Dubs

Finalized by: Adam Gibson (2nd Feb 2023)


## Context

Due to having different apis,  samediff 
and dl4j it can be hard to reproduce what a given graph 
executes from one api to another. 

All of these use the underlying libnd4j ops to execute code.
We can see what ops are being executed by enabling verbose/debug mode
in the ope executioner.




## Proposal

Execution traces are now savable as a format and can be used to generate a samediff graph
that emulates the steps executed.

A user can enable this with:
```java
        NativeOpsHolder.getInstance().getDeviceNativeOps().toggleOpTrace(true);

```

Once enabled, op executions are collected in a vector. These executions
purely save metadata including input and output shapes as well as arguments for each op.
It's stored in a vector in order. 

We can then use this vector of executions to reproduce a graph.
A user can save a graph with 
```java
        NativeOpsHolder.getInstance().getDeviceNativeOps().saveOpTrace("path");

```

We use op scopes triggered in c++ to delineate when an op executed is part of a main parent op.


## Consequences

### Advantages

* Allows easier reproduction of a given graph
* Allows decomposition of nested op execution like with attention
* Adds complexity when implementing ops since we need to remember to trigger
   notify the tracer we are currently in a parent op.

### Disadvantages
* Graph generated may be missing names or other metadata
* Execution tracing should only be done when executing one  op at a time.
