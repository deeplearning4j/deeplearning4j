# Import node pre processing

## Status
Discsusion

Proposed by: Adam Gibson (11-25-2021)

Discussed with: Paul Dubs

## Context
Nd4j's model import framework supports different protobuf based frameworks
for importing and executing models. This was introduced in [0003-Import_IR.md](0003-Import_IR.md)
One problem with importing models is compatibility between different versions of frameworks.
Often,migrations are needed to handle compatibility between versions. A node pre processor is proposed
that: when combined with the model import framework allows for 
annotation based automatic upgrades of graphs.

## Decision

In order to handle preprocessing a node to handle things like upgrades.
An end user can specify a pre processor via a combination of 2 interfaces:
1. An annotation for specifying a class that implements a relevant rule
for processing. This will automatically be discoverable via annotation scanning
similar to other frameworks. This annotation looks as follows:
```kotlin
annotation class NodePreProcessor(val nodeTypes: Array<String>, val frameworkName: String)

```
The information include the nodeTypes which are the operation types to scan for when doing upgrades on a graph.
The framework name: relevant if multiple import modules are on the classpath. Filters rules
by their intended framework for import.

2. The necessary pre processing hook that will handle processing the node 
and may modify the graph. Graph modification maybe necessary if we need to add new nodes to compensate
for modification of a node such as an attribute moving to being an input.

```kotlin

interface NodePreProcessorHook<NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {


            fun modifyNode(
                node: IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>,
                graph: IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
            ): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

}
```


## Discussion

## Consequences
### Advantages

* An automatic way of extending support for model import by providing users a
hook mechanism for handling graph modification

* Extends the model import process to handle nodes in an op specific way
allowing a way of handling op specific interactions simplifying maintenance
of other aspects of the import framework

### Disadvantages

* Adds a 3rd kind of hook for model import, thus more for users to learn
* Can be difficult to implement if user doesn't know how to work with graphs
* May have unforeseen consequences of testing due to graph modification
after creation
