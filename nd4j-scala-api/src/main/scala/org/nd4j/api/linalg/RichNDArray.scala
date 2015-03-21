/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.nd4j.api.linalg

import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.scalar.comparison.{ScalarGreaterThanOrEqual, ScalarLessThanOrEqual}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms


/**
 * Scala DSL for arrays
 */
class RichNDArray(a: INDArray) {


  // --- INDArray operators
  def +(that: INDArray): INDArray = a.add(that)

  def -(that: INDArray): INDArray = a.sub(that)

  /** element-by-element multiplication */
  def *(that: INDArray): INDArray = a.mul(that)

  /** matrix multiplication */
  def **(that: INDArray): INDArray = a.mmul(that)

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray): INDArray = a.mmul(that)

  def /(that: INDArray): INDArray = a.div(that)

  /** right division ... is this the correct symbol? */
  def \(that: INDArray): INDArray = a.rdiv(that)

  // --- In-place INDArray opertors
  /** In-place addition */
  def +=(that: INDArray): INDArray = a.addi(that)

  /** In-place subtraction */
  def -=(that: INDArray): INDArray = a.subi(that)

  /** In-placeelement-by-element multiplication */
  def *=(that: INDArray): INDArray = a.muli(that)

  /** In-place matrix multiplication */
  def **=(that: INDArray): INDArray = a.mmuli(that)

  /** In-place division */
  def /=(that: INDArray): INDArray = a.divi(that)

  /** In-place right division */
  def \=(that: INDArray): INDArray = a.rdivi(that)

  // --- Number operators
  def +(that: Number): INDArray = a.add(that)

  def -(that: Number): INDArray = a.sub(that)

  def *(that: Number): INDArray = a.mul(that)

  def /(that: Number): INDArray = a.div(that)

  def \(that: Number): INDArray = a.rdiv(that)

  // --- In-place Number operators
  def +=(that: Number): INDArray = a.addi(that)

  def -=(that: Number): INDArray = a.subi(that)

  def *=(that: Number): INDArray = a.muli(that)

  def /=(that: Number): INDArray = a.divi(that)

  /** right division ... is this the correct symbol? */
  def \=(that: Number): INDArray = a.rdivi(that)

  // --- Complex operators
  def +(that: IComplexNumber): IComplexNDArray = a.add(that)

  def -(that: IComplexNumber): IComplexNDArray = a.sub(that)

  def *(that: IComplexNumber): IComplexNDArray = a.mul(that)

  def /(that: IComplexNumber): IComplexNDArray = a.div(that)

  def apply(i: Int): Double = a.getDouble(i)

  def apply(i: Int, j: Int): Double = a.getDouble(i, j)

  def apply(indices: Int*): Double = a.getDouble(indices: _*)

  def apply(indices: Array[Int]): Double = a.getDouble(indices: _*)

  def apply(indexes: NDArrayIndex*): INDArray = a.get(indexes: _*)

  def update(i: Int, element: INDArray): INDArray = a.put(i, element)

  def update(indices: Array[NDArrayIndex], element: INDArray): INDArray = a.put(indices, element)

  def update(indices: Array[Int], element: INDArray): INDArray = a.put(indices, element)

  def update(i: Int, j: Int, element: INDArray): INDArray = a.put(i, j, element)

  def update(i: Int, value: Double): INDArray = a.putScalar(i, value)

  def update(i: Int, value: Float): INDArray = a.putScalar(i, value)

  def update(i: Int, value: Int): INDArray = a.putScalar(i, value)

  def update(i: Array[Int], value: Double): INDArray = a.putScalar(i, value)

  def update(i: Array[Int], value: Float): INDArray = a.putScalar(i, value)

  def update(i: Array[Int], value: Int): INDArray = a.putScalar(i, value)


  def unary_-(): INDArray = a.neg()

  def T: INDArray = a.transpose()

  def ===(other: Number): INDArray = a.eq(other)

  def >(other: Number): INDArray = a.gt(other)

  def <(other: Number): INDArray = a.lt(other)

  def <=(other: Number): INDArray = Nd4j.getExecutioner.exec(new ScalarLessThanOrEqual(a, other)).z()

  def >=(other: Number): INDArray = Nd4j.getExecutioner.exec(new ScalarGreaterThanOrEqual(a, other)).z()

  def ===(other: INDArray): INDArray = a.eq(other)

  def >(other: INDArray): INDArray = a.gt(other)

  def <(other: INDArray): INDArray = a.lt(other)

  def <=(other: INDArray): INDArray = Transforms.lessThanOrEqual(a, other)

  def >=(other: INDArray): INDArray = Transforms.greaterThanOrEqual(a, other)


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