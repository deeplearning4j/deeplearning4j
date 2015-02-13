package org.nd4j.api.linalg

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex


/**
 * Scala DSL for arrays
 */
class INDArrayExt(a: INDArray) {


  def +(that: INDArray): INDArray = a.add(that)
  def -(that: INDArray): INDArray = a.sub(that)
  /** element-by-element multiplication */
  def *(that: INDArray) : INDArray = a.mul(that)
  /** matrix multiplication */
  def **(that: INDArray) : INDArray = a.mmul(that)
  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray) : INDArray = a.mmul(that)
  def /(that: INDArray): INDArray = a.div(that)
  def +=(that: INDArray): INDArray = a.addi(that)
  def -=(that: INDArray): INDArray = a.subi(that)
  /** element-by-element multiplication */
  def *=(that: INDArray) : INDArray = a.muli(that)
  /** matrix multiplication */
  def **=(that: INDArray) : INDArray = a.mmuli(that)
  def /=(that: INDArray): INDArray = a.divi(that)
  
  def +(that: Number): INDArray = a.add(that)
  def -(that: Number): INDArray = a.sub(that)
  def x(that: Number) : INDArray = a.mul(that)
  def *(that: Number) : INDArray = a.mul(that)
  def /(that: Number): INDArray = a.div(that)
  def +=(that: Number): INDArray = a.addi(that)
  def -=(that: Number): INDArray = a.subi(that)
  def *=(that: Number) : INDArray = a.muli(that)
  def /=(that: Number): INDArray = a.divi(that)

  def apply(i: Int): Double = a.getDouble(i)
  def apply(i: Int, j: Int): Double = a.getDouble(i, j)
  def apply(indices: Int*): Double = a.getDouble(indices:_*)
  def apply(indices: Array[Int]): Double = apply(indices:_*)

  def update(i: Int, element: INDArray): INDArray = a.put(i, element)
  def update(indices: Array[NDArrayIndex], element: INDArray) = a.put(indices, element)
  def update(indices: Array[Int], element: INDArray) = a.put(indices, element)
  def update(i: Int, j: Int, element: INDArray) = a.put(i, j, element)
  def update(i: Int, value: Double) = a.putScalar(i, value)
  def update(i: Int, value: Float) = a.putScalar(i, value)
  def update(i: Int, value: Int) = a.putScalar(i, value)
  def update(i: Array[Int], value: Double) = a.putScalar(i, value)
  def update(i: Array[Int], value: Float) = a.putScalar(i, value)
  def update(i: Array[Int], value: Int) = a.putScalar(i, value)

  def t: INDArray = a.transpose()

}