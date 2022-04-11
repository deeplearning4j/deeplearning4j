# Invoke

## Status
**Discussion**

Proposed by: Adam Gibson (10th April 2022)


## Context

A common use case in model import is the ability to modularize graphs as sub functions.
Typically, these graphs are attributes of a node. Samediff already possesses the ability to store
samediff graphs in a dictionary called SameDiffFunctionInstances.

Graphs with control flow ops such as switch, if and while also tend to heavily use sub graphs.


## Proposal

Invoke builds on the work from [SDValue](./0018%20-%20SDValue.md) and leverages returning an ExecutionResult (a dictionary of name to SDValue)
passing the result of the invoke function as an op output in a larger parent graph.


### InvokeParams
Invoke takes node outputs from the parent graph and passes them through renamed to match the expected inputs and outputs from the sub graph.
This takes the form of InvokeParams:
```java
    public static class InvokeParams {
        private String functionName;
        private SDVariable[] inputs;
        private String[] inputVarNames;
        private String[] outputVarNames;
        private String[] subGraphInputVarNames;
        private String[] subGraphOutputVarNames;
    }


```

InvokeParams has the expected input and output names for the graph and matching subgraph input and output variable names.


### Outputs

Invoke has a special outputVariables() method for returning output values that match the outputs of the subgraph.
All outputs will have the same data types as the underlying result. 

These outputs are all converted to the ARRAY type
since the output variables are derived from the output of a function. For easy readability these op output variables will have
the same name of the output with a _functionName where functionName is the name of the function used to lookup the subgraph
to invoke.



## Consequences

### Advantages

* Allows invocation of subgraphs easily as an op
* Allows for more flexibility in how a user sets a graph up
* Can be combined with control flow to easily build loops



### Disadvantages
* More complexity in a graph
* Potential source of bugs when naming parameters and setting up the proper invoke calls