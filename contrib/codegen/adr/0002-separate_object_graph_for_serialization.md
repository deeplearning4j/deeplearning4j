# Splitting Object Graph for Code Gen and Serialization

## Status

ACCEPTED

Discussed by: Paul Dubs & Alex Black on 07. November 2019


## Context

Serialization and Code Generation have different needs when it comes to how the object graph should be laid out. When
generating code, it is a lot easier to be able to directly access referenced objects and traverse through the graph.
However, if the same object graph is used for serialization, the same object will appear in multiple places.

This becomes very apparent when defining constraints. A single constraint might be referring to an input multiple times,
and then there can also be multiple constraints that refer to multiple inputs.

The main reason why we want to keep serialization in mind, is that we want to keep code generators in other languages as
a viable option. Just serializing the graph that is meant to be used during runtime in code generation however, would
can easily become a problem when object identity is required for equality comparisons. 

An implementer in a different language would therefore need work through that graph and find identical objects AND would
have to know where that identity is a coincidence and where it is meant to be that way. By creating an object graph that
makes this explicit, we make that work easier.

## Decision

We use two distinct object graphs. One is used for code generation, and the other is used for serialization.  


## Consequences

### Advantages
* Easier to work with object graphs that are aligned with their use-case
* Less error prone when access is direct

### Disadvantages
* We have to explicitly transform one object graph into another
* If we want to support reading JSON back, we will have to also define the backwards transformation