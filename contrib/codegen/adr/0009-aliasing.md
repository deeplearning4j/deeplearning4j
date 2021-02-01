# Aliasing

## Status
ACCEPTED

Discussed by Alex Black and Paul Dubs on 05. March 2020. 
 

## Context
The API surface area is very large over all those namespaces. For consistency with our previous manually created API we want to be able to alias ops into different namespaces. Those aliases are meant as a convenience for users, so they can find ops more easily. 

# Decision
We introduce an aliasing mechanism `Alias(NamespaceName().op("opName"))` at the Namespace level, which can reference an op directly.

When an op is aliased like that, all of that ops signatures will be available in the referencing namespace. However, they will get an additional note in their documentation saying that it is an alias to the original op. In addition, the implementation of an alias signature, is a direct call of the same signature in the original namespace.  

It is not allowed to alias an op from a namespace that only has it because it has aliased it itself. 

When the requested op is not part of the given namespace, trying to alias it will throw an OpNotFoundException. 

### Example
#### Definitions
```kotlin
fun BaseOps() = Namespace("BaseOps"){
    Op("mmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.reduce"
        Input(NUMERIC, "x") { description = "First input variable" }
        Input(NUMERIC, "y") { description = "Second input variable" }
        Arg(BOOL, "transposeX") { description = "Transpose x (first argument)"; defaultValue=false }
        Arg(BOOL, "transposeY") { description = "Transpose y (second argument)"; defaultValue=false }
        Arg(BOOL, "transposeZ") { description = "Transpose result array"; defaultValue=false }
        Output(NUMERIC, "output"){ description = "" }
        Doc(Language.ANY, DocScope.ALL){
            """
                Matrix multiplication: out = mmul(x,y)
                Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc.
            """.trimIndent()
        }
    }
}

fun Linalg() = Namespace("Linalg"){
    Alias(BaseOps(), "mmul")
}
```

#### Output
This output is meant as a possible example. It is not what it will look like exactly once this feature is implemented.

```java
public class Linalg{
    /**
     * Matrix multiplication: out = mmul(x,y)<br>
     * Supports specifying transpose argument to perform operation such as mmul(a^T, b), etc. <br>
     * 
     * Alias of basepackage.baseOps.mmul(x, y)<br>
     * 
     * @param x First input variable
     * @param y Second input variable
     */
    public static INDArray mmul(INDArray x, INDArray y){
        return basepakage.baseOps.mmul(x,y);
    }
}
```
  
## Consequences
### Advantages
* We can make the API more flexible by allowing ops to be available from other namespaces

### Disadvantages
* The API becomes more crowded