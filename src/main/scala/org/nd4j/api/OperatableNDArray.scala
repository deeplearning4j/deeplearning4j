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
  def +(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.add(underlying,that)

  def -(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.sub(underlying,that)

  /** element-by-element multiplication */
  def *(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.mul(underlying,that)

  /** matrix multiplication */
  def **(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.mmul(underlying,that)

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.mmul(underlying,that)

  def /(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray  = ev.div(underlying,that)

  /** right division ... is this the correct symbol? */
  def \(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray  = ev.rdiv(underlying,that)

  // --- In-place INDArray opertors
  /** In-place addition */
  def +=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.addi(underlying,that)

  /** In-place subtraction */
  def -=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.subi(underlying,that)

  /** In-placeelement-by-element multiplication */
  def *=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.muli(underlying,that)

  /** In-place matrix multiplication */
  def **=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.mmuli(underlying,that)

  /** In-place division */
  def /=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.divi(underlying,that)

  /** In-place right division */
  def \=(that: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.rdivi(underlying,that)

  // --- Number operators
  def +(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.add(underlying,that)

  def -(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.sub(underlying,that)

  def *(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.mul(underlying,that)

  def /(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.div(underlying,that)

  def \(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.rdiv(underlying,that)

  // --- In-place Number operators
  def +=(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.addi(underlying,that)

  def -=(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.subi(underlying,that)

  def *=(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.muli(underlying,that)

  def /=(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.divi(underlying,that)

  /** right division ... is this the correct symbol? */
  def \=(that: Number)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.rdivi(underlying,that)

  // --- Complex operators
  def +(that: IComplexNumber): IComplexNDArray = underlying.add(that)

  def -(that: IComplexNumber): IComplexNDArray = underlying.sub(that)

  def *(that: IComplexNumber): IComplexNDArray = underlying.mul(that)

  def /(that: IComplexNumber): IComplexNDArray = underlying.div(that)

  def get(i: Int)(implicit ev:NDArrayEvidence[A]): ev.Value = ev.get(underlying,i)

  def get(i: Int, j: Int)(implicit ev:NDArrayEvidence[A]): ev.Value = ev.get(underlying,i, j)

  def get(indices: Int*)(implicit ev:NDArrayEvidence[A]): ev.Value = ev.get(underlying,indices: _*)

  def get(indices: Array[Int])(implicit ev:NDArrayEvidence[A]): ev.Value = ev.get(underlying,indices: _*)

  def update(i: Int, element: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.put(underlying,i, element)

  def update(indices: Array[Int], element: INDArray)(implicit ev:NDArrayEvidence[A]): ev.NDArray = ev.put(underlying, indices, element)

  def update(indices: Array[NDArrayIndex], element: INDArray): INDArray = underlying.put(indices, element)

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

  def sumT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.sum(underlying)

  def meanT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.sum(underlying)

  def normMaxT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.normMax(underlying)

  def norm1T(implicit ev: NDArrayEvidence[A]): ev.Value = ev.norm1(underlying)

  def norm2T(implicit ev: NDArrayEvidence[A]): ev.Value = ev.norm2(underlying)

  def maxT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.max(underlying)

  def minT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.min(underlying)

  def stdT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.standardDeviation(underlying)

  def prodT(implicit ev: NDArrayEvidence[A]): ev.Value = ev.product(underlying)

  def varT()(implicit ev: NDArrayEvidence[A]): ev.Value = ev.variance(underlying)
}
