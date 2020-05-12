# Gradle Kotlin Extension

Kotlin Extension to simplify nd4j operation (numpy style)

usage exemple:
```
val a:INDArray = ...
val b:INDArray = ...

// operations on array
val multiplication = a * b   // => a.mul(b)
val dotProduct = a.dot(b)    // => a.mmul(b) 
val transpose = a.T()        // => a.transpose()
...

// operations on Number
1 + a   // => a.add(1)
a * 2   // => a.mul(2)
...

fun sigmoid(x: INDArray): INDArray { 
    return 1.0 / (1.0 + exp(-x)) 
}

fun sigmoidDerivative(x: INDArray): INDArray { 
    return ( 1 - sigmoid(x) ) * sigmoid(x) 
}
```

