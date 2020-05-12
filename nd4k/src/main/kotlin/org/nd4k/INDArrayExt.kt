package org.nd4k

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.exp

// a + b    &&    a += b
operator fun INDArray.plus(other: INDArray): INDArray { return this.add(other) }
operator fun INDArray.plus(other: Number): INDArray { return this.add(other) }
operator fun Number.plus(other: INDArray): INDArray { return other.add(this) }

// -a
operator fun INDArray.unaryMinus(): INDArray { return this.mul(-1.0) }

// a - b    &&    a -= b
operator fun INDArray.minus(other: INDArray): INDArray { return this.sub(other) }
operator fun INDArray.minus(other: Number): INDArray { return this.sub(other) }
operator fun Number.minus(other: INDArray): INDArray { return this + -other }

// a * b
operator fun INDArray.times(other: INDArray): INDArray { return this.mul(other) }
operator fun INDArray.times(other: Number): INDArray { return this.mul(other) }
operator fun Number.times(other: INDArray): INDArray { return other * this }

// a / b
operator fun INDArray.div(other: INDArray): INDArray { return this.div(other) }
// operator fun INDArray.div(other: Number): INDArray // already define in INDArray
operator fun Number.div(other: INDArray): INDArray { return this * other.rdiv(1) }

// a.T()
fun INDArray.T(): INDArray { return this.transpose() }

// a.dot(b) ~numpy style
fun INDArray.dot(other: INDArray): INDArray { return this.mmul(other) }

// exp(a)
fun exp(a: INDArray): INDArray { return exp(a, false) }