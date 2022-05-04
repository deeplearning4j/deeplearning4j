# Create View

## Status
**Discussion**

Proposed by: Adam Gibson (27th April 2022)
Discussed with: Paul Dubs
Finalized by: Adam Gibson (5th May 2022)


## Context

Samediff's op graph mainly focuses on immutability to prevent bugs making many variable inputs and outputs for various ops copy on write.

This prevents performance gains with in place ops such as +=.

Currently, the samediff strided_slice operation works to provide a view of an array
but the view is a copy. This limitation prevents performance gains you typically see in views.


## Decision
 
CreateView is an op that takes in a set of SDVariables that represents index information similar to nd4j's point, interval,all, and new axis. This op allows for  dynamic generation of views of variables.

CreateView is a building block for other ops to execute in place operations. Usage of CreateView should be deliberate and only used in certain circumstances.
CreateView simply (using the aforementioned index inputs) creates a view using the existing data buffer and returns  an output that wraps the same exact buffer as is rendered  as an alternative view in a similar way as nd4j's indexing mechanisms.



These inputs are represented as follows:

1. Interval: INTERVAL_TYPE,2,1,start,end,stride,inclusive
2. Point: POINT_TYPE,1,1,offset, DEFAULT_INCLUSIVE
3. All: ALL_TYPE,0,1, DEFAULT_INCLUSIVE
4. New Axis: NEW_AXIS,1,10, DEFAULT_INCLUSIVE


This describes the general pattern the above described buffers follow:
1. type of index
2. Number of indices (representing offsets)
3. Stride
4. Inclusive/exclusive


Of note here are a few constants representing types to be passed to the ops:
1. *_TYPE: a pre-defined constant representing the kind of index this is
2. DEFAULT_INCLUSIVE: whether the index's end is inclusive or not (only needed for intervals)
this is by default 0 most of the time since the value is only relevant for intervals.

These are created as INT64 ndarrays passed in to the
operation itself.

An omission of indexing here is SpecifiedIndex.  Since SpecifiedIndex requires a copy most of the time, this op will mainly be focused on indexing that is guaranteed to use the same buffer.


## In place exception in gradient checks


Usually, arrays during training should not modify their outputs. Instead, new output arrays are allocated with calculated results being inserted into these pre-defined outputs. 

However, CreateView is, by definition, special since it is a building block for enabling manipulation of a view of the same data buffer as the input. The gradient checks in the [NonInplaceValidationListener](https://github.com/eclipse/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/validation/listeners/NonInplaceValidationListener.java#L43) make an exception to this rule to account for this particular behaviour.

## Consequences

### Advantages

* Self contained op for creating a view leveraging nd4j's existing indexing engine but
better integrated as a libnd4j op allowing for better dynamic indexing of arrays
* Similar in usage to indexes
* Contains the potential bugs from views to a pre-specified op


### Disadvantages
* Could introduce new bugs upon use
* Not the easiest interface requiring some constant methods
* Not fully integrated in to the main engine/not as transparent as a tool as it should be