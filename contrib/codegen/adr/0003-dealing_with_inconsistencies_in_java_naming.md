# Dealing with Inconsistencies in Java Naming

## Status

ACCEPTED

Discussed by: Paul Dubs, Alex Black, raver119 on 22. November 2019


## Context

There are slight inconsistencies in naming between existing op class definitions and factory methods. For example a 
factory method called `bernoulli` in the `random` namespace with a corresponding op class called 
`BernoulliDistribution`.

Two possible solutions where suggested:
1. Add an additional property that provides us with the correct class name
2. Rename classes in ND4J to ensure consistency and provide backwards compatibility via deprecated subclasses   

## Decision

For now we will introduce a `javaOpClass` property which in cases of inconsistency provides us with the correct class
name.

## Consequences

### Advantages
* We can start using this property immediately
* No need to change anything of the existing ND4J / SameDiff API

### Disadvantages
* Inconsistency continues to exist within the Java codebase
* We have to take extra care to add the new property where needed