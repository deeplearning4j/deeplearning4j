# Use Kotlin-based DSL as the source of truth 

## Status

ACCEPTED

Discussed by: Paul Dubs, Alex Black, raver119 on 19. October 2019


## Context

This code generation experiment is meant to be our starting point for both the API unification for ND4J and SameDiff, 
and the multi-language support. For this reason we have to define ops, or their interface, in a language neutral way.

The initial idea was to use a Language Workbench like MPS. This had to be discarded because of bugs and limitations
encountered while trying to define a language that would work for a few simple examples.

The next idea was to use Ops defined in JSON files. This would have allowed us to define Ops as human readable data and
read and write those files from any programming language. However, the drawback with this approach is that writing json
manually invites many problems if written manually (e.g. typos, bad structuring, having to look up the proper keys,...).
In order to rectify that drawback, we would have to create custom tooling, that we would have to maintain and that
contributors would have to use. 

Using a Java builder pattern based approach is very verbose. 

## Decision

We use a Kotlin-based DSL to define Ops.

## Consequences

Using a full programming language as the platform to build our DSL has both advantages and drawbacks.

### Drawbacks
* Contributors will have to install a JVM in order to be able to run code generation or get a serialized op graph
* Serialization is a one way road, we only output to JSON, but don't read from it
* Contributors to Op definitions have to learn about our DSL and maybe some Kotlin
* Contributors to the DSL will have to learn about Kotlin 


### Advantages
* We can utilize the Java knowledge of the existing team for writing code generators as Kotlin is two-way interoperable
  with Java and other JVM languages. 
* We can utilize IntelliJ (or other IDEs supporting Kotlin) as an existing editor for Ops definitions. This provides us 
  with benefits like code completion, error highlighting
* We get compile time checks, freeing us from trivial errors like typos
* We can utilize some base features of Kotlin, like variable assignment to simplify the implementation
* Kotlin has first class DSL definition support, allowing us to make Op definitions almost as easy to read as a full
  language workbench would have allowed