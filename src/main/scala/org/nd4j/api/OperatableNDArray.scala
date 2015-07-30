/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.api

import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.NDArrayIndex


/**
 * Scala DSL for arrays
 */
trait OperatableNDArray[A <: INDArray] {
  val underlying: A

  // --- INDArray operators
  def +(that: INDArray): INDArray = underlying.add(that)

  def -(that: INDArray): INDArray = underlying.sub(that)

  /** element-by-element multiplication */
  def *(that: INDArray): INDArray = underlying.mul(that)

  /** matrix multiplication */
  def **(that: INDArray): INDArray = underlying.mmul(that)

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray): INDArray = underlying.mmul(that)

  def /(that: INDArray): INDArray = underlying.div(that)

  /** right division ... is this the correct symbol? */
  def \(that: INDArray): INDArray = underlying.rdiv(that)

  // --- In-place INDArray opertors
  /** In-place addition */
  def +=(that: INDArray): INDArray = underlying.addi(that)

  /** In-place subtraction */
  def -=(that: INDArray): INDArray = underlying.subi(that)

  /** In-placeelement-by-element multiplication */
  def *=(that: INDArray): INDArray = underlying.muli(that)

  /** In-place matrix multiplication */
  def **=(that: INDArray): INDArray = underlying.mmuli(that)

  /** In-place division */
  def /=(that: INDArray): INDArray = underlying.divi(that)

  /** In-place right division */
  def \=(that: INDArray): INDArray = underlying.rdivi(that)

  // --- Number operators
  def +(that: Number): INDArray = underlying.add(that)

  def -(that: Number): INDArray = underlying.sub(that)

  def *(that: Number): INDArray = underlying.mul(that)

  def /(that: Number): INDArray = underlying.div(that)

  def \(that: Number): INDArray = underlying.rdiv(that)

  // --- In-place Number operators
  def +=(that: Number): INDArray = underlying.addi(that)

  def -=(that: Number): INDArray = underlying.subi(that)

  def *=(that: Number): INDArray = underlying.muli(that)

  def /=(that: Number): INDArray = underlying.divi(that)

  /** right division ... is this the correct symbol? */
  def \=(that: Number): INDArray = underlying.rdivi(that)

  // --- Complex operators
  def +(that: IComplexNumber): IComplexNDArray = underlying.add(that)

  def -(that: IComplexNumber): IComplexNDArray = underlying.sub(that)

  def *(that: IComplexNumber): IComplexNDArray = underlying.mul(that)

  def /(that: IComplexNumber): IComplexNDArray = underlying.div(that)

  def get(i: Int): Double = underlying.getDouble(i)

  def get(i: Int, j: Int): Double = underlying.getDouble(i, j)

  def get(indices: Int*): Double = underlying.getDouble(indices: _*)

  def get(indices: Array[Int]): Double = underlying.getDouble(indices: _*)

  def update(i: Int, element: INDArray): INDArray = underlying.put(i, element)

  def update(indices: Array[NDArrayIndex], element: INDArray): INDArray = underlying.put(indices, element)

  def update(indices: Array[Int], element: INDArray): INDArray = underlying.put(indices, element)

  def update(i: Int, j: Int, element: INDArray): INDArray = underlying.put(i, j, element)

  def update(i: Int, value: Double): INDArray = underlying.putScalar(i, value)

  def update(i: Int, value: Float): INDArray = underlying.putScalar(i, value)

  def update(i: Int, value: Int): INDArray = underlying.putScalar(i, value)

  def update(i: Array[Int], value: Double): INDArray = underlying.putScalar(i, value)

  def update(i: Array[Int], value: Float): INDArray = underlying.putScalar(i, value)

  def update(i: Array[Int], value: Int): INDArray = underlying.putScalar(i, value)

  def unary_-(): INDArray = underlying.neg()

  def T: INDArray = underlying.transpose()

  def ===(other: Number): INDArray = underlying.eq(other)

  def ===(other: INDArray): INDArray = underlying.eq(other)

  def sumT(implicit ev: NDArrayEvidence[A]): ev.T = ev.sum(underlying)

  def meanT(implicit ev: NDArrayEvidence[A]): ev.T = ev.sum(underlying)

  def normMaxT(implicit ev: NDArrayEvidence[A]): ev.T = ev.normMax(underlying)

  def norm1T(implicit ev: NDArrayEvidence[A]): ev.T = ev.norm1(underlying)

  def norm2T(implicit ev: NDArrayEvidence[A]): ev.T = ev.norm2(underlying)

  def maxT(implicit ev: NDArrayEvidence[A]): ev.T = ev.max(underlying)

  def minT(implicit ev: NDArrayEvidence[A]): ev.T = ev.min(underlying)

  def stdT(implicit ev: NDArrayEvidence[A]): ev.T = ev.standardDeviation(underlying)

  def prodT(implicit ev: NDArrayEvidence[A]): ev.T = ev.product(underlying)

  def varT()(implicit ev: NDArrayEvidence[A]): ev.T = ev.variance(underlying)
}

// https://gist.github.com/teroxik/5349331
class RichComplexNDArray(a: IComplexNDArray) {

  def +(that: INDArray): IComplexNDArray = a.add(that)

  def -(that: INDArray): IComplexNDArray = a.sub(that)

  /** element-by-element multiplication */
  def *(that: INDArray): IComplexNDArray = a.mul(that)

  /** matrix multiplication */
  def **(that: INDArray): IComplexNDArray = a.mmul(that)

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray): IComplexNDArray = a.mmul(that)

  def /(that: INDArray): IComplexNDArray = a.div(that)

  /** right division ... is this the correct symbol? */
  def \(that: INDArray): IComplexNDArray = a.rdiv(that)
}
