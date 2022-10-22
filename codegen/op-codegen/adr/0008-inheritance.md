# Inheritance 

## Status
ACCEPTED

Discussed by Alex Black and Paul Dubs on 29. November 2019 and 2. December 2019. 

## Context
In many cases ops have a similar interface. For example all transform ops take a single input, but some of them take
additional arguments; all pairwise ops take two inputs, and so on. The documentation of those ops is often the result
of copy & paste with just a few little modifications, and changing anything later on suddenly becomes a huge undertaking
because what should effectively be a change in a single place, has to be changed in many places.

Another issue that copy & paste based definitions bring to the table is that this practice effectively makes any
relationship between those ops implicit.

When defining ops with the DSL we prefer to make things as explicit as possible, while also reducing repetition and
boilerplate.

The existing inheritance mechanism, added without a formal proposal at the beginning of the project, allows the
definition of an abstract base op. The new op based on it, can copy any of the given parts before it gets to define its
own properties. This approach has the problem that we can't inherit from multiple base ops, and that we do not get any
direct access to its fields for ease of use.

# Decision
We introduce an explicit mixin mechanism `Mixin("name") {...}` which can define any parts of any op, but isn't an Op
definition on its own. Mixins can be defined at top-level, thereby being usable across namespaces.

A mixin is mixed into an op with `useMixin(mixinReference, ...options...)` within an op context. It will add all (if 
not otherwise configured) definitions of the mixin to the current op as if they were copied into its place. 
If `useMixin(ref)` is used as the first thing within an op definition, then it will behave exactly like the old 
inheritance mechanism.

`useMixin(ref)` returns a holder object, that can be used to reference its parameters.

The available options on `useMixin` are `keepInputs`, `keepArgs`, `keepOutputs`, `keepSignatures`, `keepDoc`,
`keepConstraints`. They default to `true`.

If there is a naming conflict between mixins or between mixin and op definition, the last definition wins.

### Example
```kotlin

val indexAccum = Mixin("indexAccum"){
    legacy = true
    javaPackage = "org.nd4j.linalg.api.ops.impl.indexaccum"
    val input = Input(NUMERIC, "in") { description = "Input variable" }
    val keepDims = Arg(BOOL, "keepDims") { description = "If true: keep the dimensions that are reduced on (as length 1). False: remove the reduction dimensions"; defaultValue = false }
    val dims = Arg(INT, "dimensions"){ count = AtLeast(1); description = "Dimensions to reduce over. If dimensions are not specified, full array reduction is performed" }
    Output(NUMERIC, "output"){ description = "Reduced array of rank (input rank - num dimensions)" }

    Signature(input, dims)
    AllParamSignature(withOutput = false)
}


Namespace("math"){
    Op("firstIndex") {
        val idxAccum = useMixin(indexAccum, keepSignatures=false)
        var c = Arg(CONDITION, "condition") { description = "Condition to check on input variable" }
        Signature(idxAccum.input("in"), c, idxAccum.arg("dimensions"))
        Signature(idxAccum.input("in"), c, idxAccum.arg("keepDims"), idxAccum.arg("dimensions"))

        Doc(Language.ANY, DocScope.ALL){
            """
                First index reduction operation.
                Returns a variable that contains the index of the first element that matches the specified condition (for each
                slice along the specified dimensions)
                Note that if keepDims = true, the output variable has the same rank as the input variable,
                with the reduced dimensions having size 1. This can be useful for later broadcast operations (such as subtracting
                the mean along a dimension).
                Example: if input has shape [a,b,c] and dimensions=[1] then output has shape:
                keepDims = true: [a,1,c]
                keepDims = false: [a,c]
            """.trimIndent()
        }
    }
}
```
 
  
## Consequences
### Advantages
* We can have multiple inheritance 
* We can share op similarities across namespaces
* We get explicit access to parameters defined in mixins

### Disadvantages
* We have to adapt our current usage