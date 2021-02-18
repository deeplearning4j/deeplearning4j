# Configuration Objects

## Status

ACCEPTED

Discussed by Alex Black, raver119 and Paul Dubs on 27. November 2019.

## Context
Some Ops (esp. convolution) have many parameters. Many of them can have reasonable defaults, but even then creating
signatures for evey reasonable configuration may be impossible, as those signatures would require different naming in
order to be actually distinguishable from each other.

In other cases, an op may have a lot of same typed parameters that are required (e.g. GRU, LSTM, SRU) but it is very
easy to mix them up. 

For both of those cases (many optional parameters, easily mixed up required parameters) it is reasonable to use a
config holder with builder pattern in languages that do not support named or default parameters. 

In our current codebase those configurations are often used across several related ops. 


## Decision
We add a `Config("name"){ ... }` section to the namespace context. It supports `Input` and `Arg` definitions in the same
way that `Op` does.

Ops that want to use that config can use `useConfig(conf)`. As configs are often reused across related objects, this
will have the effect of a mixin: All inputs and args defined in that config will also be automatically defined on that
Op. If there is a naming conflict, an exception will be thrown at construction time.

For default signatures, configs will be passed at the end, in the order that they were added to the Op.

If other signatures are desired, configs, like regular inputs and args, can be passed to `Signature`.

In languages that do not support default or named parameters, a config holder will be created, that will take the
parameters of the config using a builder pattern. For languages with default and named parameters, no additional config
holder will be created, and the parameters of the config will be treated as if they were directly configured on the Op.

### Example
This example shows a very simple case in order to highlight how this feature would be used. 
```kotlin
fun RNN() = Namespace("RNN"){
    val sruWeights = Config("SRUWeights"){
        Input(FLOATING_POINT, "weights"){ description = "Weights, with shape [inSize, 3*inSize]" }
        Input(FLOATING_POINT, "bias"){ description = "Biases, with shape [2*inSize]" }
    }

    Op("SRU"){
        Input(FLOATING_POINT, "x"){ description = "..." }
        Input(FLOATING_POINT, "initialC"){ description = "..." }
        Input(FLOATING_POINT, "mask"){ description = "..." }
        
        useConfig(sruWeights)
    
        Output(FLOATING_POINT, "out"){ description = "..." }
    }
    
    Op("SRUCell"){
        val x = Input(FLOATING_POINT, "x"){ description = "..." }
        val cLast = Input(FLOATING_POINT, "cLast"){ description = "..." }
        
        val conf = useConfig(sruWeights)
    
        Output(FLOATING_POINT, "out"){ description = "..." }
    
        // Just for demonstration purposes
        Signature(x, cLast, conf)
    }
}
```
 
## Consequences

### Advantages
* Ops that share parameters can make that sharing explicit
* Easier definition of related ops with common parameters
* Simplifies usage of complex ops in languages without named parameters

### Disadvantages
* Not all parameters are defined directly within an op anymore
* Configs are not shareable across namespaces