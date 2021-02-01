# SameDiff file format proposal

## Status
Accepted

Proposed by: Alex Black (15-05-2020)

Discussed with: raver119

## Context

SameDiff models need to be serializable - i.e., something we can save to disk or send over the network.
Additionally, we need to be able to save and load model files in C++, and have those be readable in other languages (mainly Java).

Currently, we have a FlatBuffers-based format for SameDiff graph serialization, but it has a number of problems, as discussed in this issue: https://github.com/eclipse/deeplearning4j/issues/8312


## Decision

We will transition from a pure FlatBuffers to a Zip + FlatBuffers model format.

FlatBuffers will be used for the graph structure only. Parameters will be stored separately to the graph structure, also within the zip.

We will introduce the ability to support multiple versions of a graph in the model files.
This will enable the model file to support storing 
* Multiple data types (for example, a FP32 version and a quantized INT8 version)
* Multiple different checkpoints (parameters after 1000 iterations, after 5000, and so on)
* Multiple versions of a given model (English vs. Chinese, or cased/uncased, etc)

By default when loading a graph (unless it is otherwise specified) we will load the most recent model tag.
Tags must be valid file/folder identifiers, and are not case sensitive.


The structure of the zip file will be as follows:
```
tags.txt                         //List of graph tags, one per line, in UTF8 format, no duplicates. Oldest first, newest last
<tag_name>/graph.fb              //The graph structure, in FlatBuffers format
<tag_name>/params.txt            //The mapping between variable names and parameter file names
<tag_name>/params/*.fb           //The set of NDArrays that are the parameters, in FlatBuffers format
<tag_name>/trainingConfig.fb     //The training configuration - updater, learning rate, etc
<tag_name>/updater.txt           //The mapping between variable names and the updater state file names
<tag_name>/updater/*.fb          //The set of NDArrays that are the updater state
```

Note that params.txt will allow for parameter sharing via references to other parameters:
```
my_normal_param 0
shared_param <other_tag_name>/7
```
This means the parameters values for parameter "my_normal_param" are present at `<tag_name>/params/0.fb` within the zip file, and the parameter values for "shared_param" are available at `<other_tag_name>/params/7.fb`

Note also that the motivation for using the params.txt file (instead of the raw parameter name as the file name) is that some parameters will have invalid or ambiguous file names - "my/param/name", "&MyParam*" etc

In terms of updater state, they will be stored in a similar format. For example, for the Adam updater with the M and V state arrays (each of same shape as the parameter):
```
my_param 0 1
other_param 2 3
```
That means my_param(M) is `<tag_name>/updater/0.fb` and my_param(V) is at `<tag_name>/updater/1.fb`
This format also allows for updater state sharing, if we need it.


**Graph Structure**

The graph structure will be encoded in FlatBuffers format using a schema with 2 parts:
1. A list of variables - each with name, datatype, and (for placeholders, constants and parameters) a shape
2. A list of operations - each with a name, op name/type, input variable names, output variable names, and arguments

Note that both legacy and custom ops will be encoded in the same way. For legacy ops, we simply need the operation type, and the operation number.

Operation argument encoding will be done using named arguments: essentially, a `Map<String,T>` structure, where T is one of `{long, double, boolean, datatype}`.
This allows for improved backward compatibility (no ambiguity as ops are modified after a graph file was written) and improved interpretability compared to using simple arrays of iargs, bargs, targs and dargs.
One consequence/downside of this is that we need to define mapping between our named arguments and iargs/bargs/targs/dargs. In Java we have essentially done this manually, though clearly don't want to replicate this work in C++ (or any future languages).

To avoid the need to do a significant amount of work (such as moving the name/arg mapping to code generation) the following is proposed:
The `Map<String,T>` is split up in the FlatBuffers schema into 4 pairs of fields.
* `String[] iArgNames`, `long[] iArgs`
* `String[] tArgNames`, `double[] dArgs`
* `String[] bArgNames`, `boolean[] bArgs`
* `String[] dArgNames`, `DataType[] dArgs`

Clearly the name and value arrays (for each pair) would each be the same length, and name/value correspondence is by array index.

This is essentially equivalent to the `Map<String,T>` representation, but has the benefit of not needing us to define the mapping for named args to array-style args any time soon in C++, but also allowing us to add it in the future (mainly before we can write graphs from C++, or have better/proper backward compatibility after op changes)


**Extensibility to Other Types**

Suppose in the future we want to store other data for a variable, not just an array?
Examples include lists and maps (for example, for NLP applications).

While we will not implement this right now, there are a number of options for adding this without breaking backward compatibility.

First: we can enhance the params.txt file format, perhaps using something like the following:
```
map_param 0 MAP
```

Second: We can add a similar text file for other types. For example, a params_maps.txt, same format as params.txt, with content at `<tag_name>/params_maps/*.fb`

