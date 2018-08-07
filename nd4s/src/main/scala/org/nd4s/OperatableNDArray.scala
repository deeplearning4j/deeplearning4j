/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4s

import org.nd4j.linalg.api.complex.{IComplexNDArray, IComplexNumber}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.INDArrayIndex


/**
 * Scala DSL for arrays
 */
trait OperatableNDArray[A <: INDArray] {
  val underlying: A

  // to keep compatibility with Predef.any2stringadd syntax.
  def +(that: String): String = underlying.toString + that

  // --- INDArray operators
  def +(that: INDArray)(implicit ev:NDArrayEvidence[A, _]): A = ev.add(underlying,that)

  def -(that: INDArray)(implicit ev:NDArrayEvidence[A, _]): A = ev.sub(underlying,that)

  /** element-by-element multiplication */
  def *(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.mul(underlying,that)

  /** matrix multiplication */
  def **(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.mmul(underlying,that)

  /** matrix multiplication using Numpy syntax for arrays */
  def dot(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.mmul(underlying,that)

  def /(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A  = ev.div(underlying,that)

  /** right division ... is this the correct symbol? */
  def \(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A  = ev.rdiv(underlying,that)

  // --- In-place INDArray opertors
  /** In-place addition */
  def +=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.addi(underlying,that)

  /** In-place subtraction */
  def -=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.subi(underlying,that)

  /** In-placeelement-by-element multiplication */
  def *=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.muli(underlying,that)

  /** In-place matrix multiplication */
  def **=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.mmuli(underlying,that)

  /** In-place division */
  def /=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.divi(underlying,that)

  /** In-place right division */
  def \=(that: INDArray)(implicit ev:NDArrayEvidence[A,_]): A = ev.rdivi(underlying,that)

  // --- Number operators
  def +(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.add(underlying,that)

  def -(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.sub(underlying,that)

  def *(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.mul(underlying,that)

  def /(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.div(underlying,that)

  def \(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.rdiv(underlying,that)

  // --- In-place Number operators
  def +=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.addi(underlying,that)

  def -=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.subi(underlying,that)

  def *=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.muli(underlying,that)

  def /=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.divi(underlying,that)

  def %=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.remainderi(underlying,that)


  /** right division ... is this the correct symbol? */
  def \=(that: Number)(implicit ev:NDArrayEvidence[A,_]): A = ev.rdivi(underlying,that)

  // --- Complex operators
  def +(that: IComplexNumber): IComplexNDArray = underlying.add(that)

  def -(that: IComplexNumber): IComplexNDArray = underlying.sub(that)

  def *(that: IComplexNumber): IComplexNDArray = underlying.mul(that)

  def /(that: IComplexNumber): IComplexNDArray = underlying.div(that)

  def get[B](i: Int)(implicit ev:NDArrayEvidence[A,B]): B = ev.get(underlying,i)

  def get[B](i: Int, j: Int)(implicit ev:NDArrayEvidence[A,B]): B = ev.get(underlying,i, j)

  def get[B](indices: Int*)(implicit ev:NDArrayEvidence[A,B]): B = ev.get(underlying,indices: _*)

  def apply[B](i: Int)(implicit ev:NDArrayEvidence[A,B]): B = get(i)

  def apply[B](i: Int, j: Int)(implicit ev:NDArrayEvidence[A,B]): B = get(i,j)

  def apply[B](indices: Int*)(implicit ev:NDArrayEvidence[A,B]): B = get(indices: _*)

  def get[B](indices: Array[Int])(implicit ev:NDArrayEvidence[A,B]): B = ev.get(underlying,indices: _*)

  def unary_-(): INDArray = underlying.neg()

  def T: INDArray = underlying.transpose()

  def ===(other: Number): INDArray = underlying.eq(other)

  def ===(other: INDArray): INDArray = underlying.eq(other)

  def sumT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.sum(underlying)

  def meanT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.mean(underlying)

  def normMaxT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.normMax(underlying)

  def norm1T[B](implicit ev: NDArrayEvidence[A,B]): B = ev.norm1(underlying)

  def norm2T[B](implicit ev: NDArrayEvidence[A,B]): B = ev.norm2(underlying)

  def maxT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.max(underlying)

  def minT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.min(underlying)

  def stdT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.standardDeviation(underlying)

  def prodT[B](implicit ev: NDArrayEvidence[A,B]): B = ev.product(underlying)

  def varT[B]()(implicit ev: NDArrayEvidence[A,B]): B = ev.variance(underlying)
}
