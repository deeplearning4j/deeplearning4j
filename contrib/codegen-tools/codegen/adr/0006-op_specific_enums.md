# Op specific enums

## Status

ACCEPTED

Discussed by: Alex Black, Robert Altena and Paul Dubs on 26. November 2019

## Context
Some ops have an ordinal parameter which switches between a few possible modes. Giving those modes a proper name
makes usage and documentation easier. 


## Decision
We allow `Arg` sections to have an `ENUM` data type and add a `possibleValues` property to define the possible values
for this arg. The ordinal number of the enum is the same as its position within the `possibleValues` list starting from 
`0`. 

A runtime check on op construction, will ensure that each enum arg has one or more possible values, and that default
values match one of the possible values (if applicable).

On code generation, an appropriate representation of this enum will be generated in the target language. The name of 
the generated enum will be derived from the name of the arg.

### Example
```kotlin
Arg(ENUM, "padMode"){ 
  possibleValues = listOf("CONSTANT", "REFLECT", "SYMMETRIC")
  description = "padding mode"  
}
```

## Consequences

### Advantages
* We get easily understandable names for otherwise in-transparent ordinal mode modifiers

### Disadvantages
* The defined enum can only be used for a single op
* The defined enum is only usable with a single arg
