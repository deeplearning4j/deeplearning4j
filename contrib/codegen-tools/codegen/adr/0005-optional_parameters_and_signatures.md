# Optional parameters and signatures

## Status

ACCEPTED

Discussed by: Alex Black, raver 119 and Paul Dubs on 25. November 2019

## Context

Not all inputs or args (= parameters) are always required. 

Often there are sensible defaults available. We want to be able to make those defaults explicit where possible.

Even though some parameters may be optional, they might become required in the presence of other optional parameters.

We need a way to explicitly define what combinations are possible.


## Decision
We drop the `optional` property on parameters. Instead, parameters get an additional property `defaultValue`. It can be 
set to either a fixed literal value (e.g. `7`, `"something"`, `null`), an Arg, or it may reference the specific methods 
`shape()` and `dataType()` on inputs and outputs. Parameters with `defaultValue` specified are treated as optional.

To be able to deal with languages that do not support default values for arguments, Signatures will be specified.
Signatures are specified using a `Signature(a,b,c){ "signature specific documentation" }` section for each signature.
With the signature specific documentation being optional. 

Signatures making use of outputs will only be generated for NDArray programming mode, not in SameDiff mode. This also
means that parameters with a `defaultValue` based on an output will be treated as required in SameDiff mode.

If signatures are specified, only the specified signatures will be generated.

If no signatures are explicitly specified, only the "all-arg" and "no-optional-arg" signatures will be generated. In
NDArray programming mode, the default signatures also include a variant that includes the output.


## Examples
### BatchNorm with all otherwise auto generated signatures stated explicitly
 
```kotlin
Op("batchNorm") {
    val input    = Input(NUMERIC, "input") { description = "Input variable" }
    val mean     = Input(NUMERIC, "mean") { description = "Mean value. For 1d axis, this should match input.size(axis)" }
    val variance = Input(NUMERIC, "variance") { description = "Variance value. For 1d axis, this should match input.size(axis)" }
    val gamma    = Input(NUMERIC, "gamma") { description = "Gamma value. For 1d axis, this should match input.size(axis)" }
    val beta     = Input(NUMERIC, "beta") { description = "Beta value. For 1d axis, this should match input.size(axis)" }

    val applyGamma = Arg(BOOL, "applyGamma") { description = ""; defaultValue = true}
    val applyBeta  = Arg(BOOL, "applyBeta") { description = ""; defaultValue = true}
    val axis       = Arg(INT, "axis"){
        count = AtLeast(1) 
        description = """
        For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.
        For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC
        For 1d/RNN activations: 1 for NCW format, 2 for NWC
        """.trimIndent()
    }

    val out = Output(INT, "output"){ description = "Output variable for batch normalization" }

    Doc(Language.ANY, DocScope.ALL){
        """
        Neural network batch normalization operation.
        For details, see <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a>
        """.trimIndent()
    }
  
    Signature(input, mean, variance, gamma, beta, axis)
    Signature(input, mean, variance, gamma, beta, applyGamma, applyBeta, axis)
    Signature(out, input, mean, variance, gamma, beta, axis)
    Signature(out, input, mean, variance, gamma, beta, applyGamma, applyBeta, axis)
}
```

### Random Uniform initialization with support for (dataType, shape) and (out) invocation
```kotlin
Op("uniform") {
    val out = Output(NUMERIC, "output") { description = "new random %INPUT_TYPE%, where values are randomly sampled according to a uniform distribution" }    
    
    val min = Arg(FLOATING_POINT, "min") { description = "Minimum value" }
    val max = Arg(FLOATING_POINT, "max") { description = "Maximum value." }
    val dataType = Arg(DATA_TYPE, "dataType") { description = "Data Type of the output array"; defaultValue = out.dataType() }
    val shape = Arg(INT, "shape") { count = AtLeast(1); description = "Shape of the new random %INPUT_TYPE%, as a 1D array"; defaultValue = out.dataType() }

    Doc(Language.ANY, DocScope.ALL) {
        """
        Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a uniform distribution,
        U(min,max)
        """.trimIndent()
    }

    Signature(min, max, dataType, shape)
    Signature(out, min, max)
}
```


## Consequences

### Advantages
* We get to explicitly define edge cases
* We can make Signatures compatible with existing code
* Even in languages with default value support, the added signatures may become useful parts of the documentation

### Disadvantages
* The order of definitions within the op changes to Output first, if inputs need to reference it

