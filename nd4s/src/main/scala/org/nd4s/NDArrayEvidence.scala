/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.indexing.INDArrayIndex
import org.nd4s.Implicits._

object Evidences {
  implicit val double = DoubleNDArrayEvidence
  implicit val float = FloatNDArrayEvidence
  implicit val int = IntNDArrayEvidence
  implicit val long = LongNDArrayEvidence
  implicit val byte = ByteNDArrayEvidence
}

object NDArrayEvidence {
  implicit val doubleNDArrayEvidence = DoubleNDArrayEvidence

}

trait NDArrayEvidence[NDArray <: INDArray, Value] {

  def sum(ndarray: NDArray): Value

  def mean(ndarray: NDArray): Value

  def normMax(ndarray: NDArray): Value

  def norm1(ndarray: NDArray): Value

  def norm2(ndarray: NDArray): Value

  def max(ndarray: NDArray): Value

  def min(ndarray: NDArray): Value

  def standardDeviation(ndarray: NDArray): Value

  def product(ndarray: NDArray): Value

  def variance(ndarray: NDArray): Value

  def remainder(a: NDArray, that: INDArray): NDArray

  def add(a: NDArray, that: INDArray): NDArray

  def sub(a: NDArray, that: INDArray): NDArray

  def mul(a: NDArray, that: INDArray): NDArray

  def mmul(a: NDArray, that: INDArray): NDArray

  def div(a: NDArray, that: INDArray): NDArray

  def rdiv(a: NDArray, that: INDArray): NDArray

  def addi(a: NDArray, that: INDArray): NDArray

  def subi(a: NDArray, that: INDArray): NDArray

  def muli(a: NDArray, that: INDArray): NDArray

  def mmuli(a: NDArray, that: INDArray): NDArray

  def remainderi(a: NDArray, that: INDArray): NDArray

  def remainderi(a: NDArray, that: Number): NDArray

  def divi(a: NDArray, that: INDArray): NDArray

  def rdivi(a: NDArray, that: INDArray): NDArray

  def remainder(a: NDArray, that: Number): NDArray

  def add(a: NDArray, that: Number): NDArray

  def sub(a: NDArray, that: Number): NDArray

  def mul(a: NDArray, that: Number): NDArray

  def div(a: NDArray, that: Number): NDArray

  def rdiv(a: NDArray, that: Number): NDArray

  def addi(a: NDArray, that: Number): NDArray

  def subi(a: NDArray, that: Number): NDArray

  def muli(a: NDArray, that: Number): NDArray

  def divi(a: NDArray, that: Number): NDArray

  def rdivi(a: NDArray, that: Number): NDArray

  def put(a: NDArray, i: Int, element: INDArray): NDArray

  def put(a: NDArray, i: Array[Int], element: INDArray): NDArray

  def get(a: NDArray, i: Long): Value

  //def get(a: NDArray, i: Long, j: Long): Value

  def get(a: NDArray, i: Long*): Value

  def get(a: NDArray, i: INDArrayIndex*): NDArray

  def reshape(a: NDArray, i: Int*): NDArray

  def linearView(a: NDArray): NDArray

  def dup(a: NDArray): NDArray

  def create(arr: Array[Value]): NDArray

  def create(arr: Array[Value], shape: Int*): NDArray

  def create(arr: Array[Value], shape: Array[Int], ordering: NDOrdering, offset: Int): NDArray

  def update(a: NDArray, indices: Array[IndexRange], i: Value): NDArray

  def update(a: NDArray, indices: Array[IndexRange], i: NDArray): NDArray

  def greaterThan(left: Value, right: Value): Boolean

  def lessThan(left: Value, right: Value): Boolean
}

trait RealNDArrayEvidence[Value] extends NDArrayEvidence[INDArray, Value] {

  override def remainder(a: INDArray, that: INDArray): INDArray =
    a.remainder(that)

  override def add(a: INDArray, that: INDArray): INDArray = a.add(that)

  override def div(a: INDArray, that: INDArray): INDArray = a.div(that)

  override def mul(a: INDArray, that: INDArray): INDArray = a.mul(that)

  override def rdiv(a: INDArray, that: INDArray): INDArray = a.rdiv(that)

  override def sub(a: INDArray, that: INDArray): INDArray = a.sub(that)

  override def mmul(a: INDArray, that: INDArray): INDArray = a.mmul(that)

  override def addi(a: INDArray, that: INDArray): INDArray = a.addi(that)

  override def subi(a: INDArray, that: INDArray): INDArray = a.subi(that)

  override def muli(a: INDArray, that: INDArray): INDArray = a.muli(that)

  override def mmuli(a: INDArray, that: INDArray): INDArray = a.mmuli(that)

  override def remainderi(a: INDArray, that: INDArray): INDArray =
    a.remainderi(that)

  override def remainderi(a: INDArray, that: Number): INDArray =
    a.remainderi(that)

  override def divi(a: INDArray, that: INDArray): INDArray = a.divi(that)

  override def rdivi(a: INDArray, that: INDArray): INDArray = a.rdivi(that)

  override def remainder(a: INDArray, that: Number): INDArray =
    a.remainder(that)

  override def add(a: INDArray, that: Number): INDArray = a.add(that)

  override def sub(a: INDArray, that: Number): INDArray = a.sub(that)

  override def mul(a: INDArray, that: Number): INDArray = a.mul(that)

  override def div(a: INDArray, that: Number): INDArray = a.div(that)

  override def rdiv(a: INDArray, that: Number): INDArray = a.rdiv(that)

  override def addi(a: INDArray, that: Number): INDArray = a.addi(that)

  override def subi(a: INDArray, that: Number): INDArray = a.subi(that)

  override def muli(a: INDArray, that: Number): INDArray = a.muli(that)

  override def divi(a: INDArray, that: Number): INDArray = a.divi(that)

  override def rdivi(a: INDArray, that: Number): INDArray = a.rdivi(that)

  override def put(a: INDArray, i: Array[Int], element: INDArray): INDArray =
    a.put(i, element)

  override def put(a: INDArray, i: Int, element: INDArray): INDArray =
    a.put(i, element)

  override def get(a: INDArray, i: INDArrayIndex*): INDArray = a.get(i: _*)

  override def reshape(a: INDArray, i: Int*): INDArray =
    a.reshape(i.map(_.toLong): _*)

  override def linearView(a: INDArray): INDArray = a.reshape(-1)

  override def dup(a: INDArray): INDArray = a.dup()

  override def update(underlying: INDArray, ir: Array[IndexRange], num: INDArray): INDArray = {
    if (ir.exists(_.hasNegative))
      underlying.indicesFrom(ir: _*).indices.foreach { i =>
        underlying.put(i, num)
      } else
      underlying.put(underlying.getINDArrayIndexfrom(ir: _*).toArray, num)
    underlying
  }
}

case object DoubleNDArrayEvidence extends RealNDArrayEvidence[Double] {

  override def sum(ndarray: INDArray): Double =
    ndarray.sumNumber().doubleValue()

  override def mean(ndarray: INDArray): Double =
    ndarray.meanNumber().doubleValue()

  override def variance(ndarray: INDArray): Double =
    ndarray.varNumber().doubleValue()

  override def norm2(ndarray: INDArray): Double =
    ndarray.norm2Number().doubleValue()

  override def max(ndarray: INDArray): Double =
    ndarray.maxNumber().doubleValue()

  override def product(ndarray: INDArray): Double =
    ndarray.prodNumber().doubleValue()

  override def standardDeviation(ndarray: INDArray): Double =
    ndarray.stdNumber().doubleValue()

  override def normMax(ndarray: INDArray): Double =
    ndarray.normmaxNumber().doubleValue()

  override def min(ndarray: INDArray): Double =
    ndarray.minNumber().doubleValue()

  override def norm1(ndarray: INDArray): Double =
    ndarray.norm1Number().doubleValue()

  override def get(a: INDArray, i: Long): Double = a.getDouble(i)

  //override def get(a: INDArray, i: Int): Double = a.getDouble(i.toLong)

  //override def get(a: INDArray, i: Int, j: Int): Double = a.getDouble(i.toLong, j.toLong)

  //override def get(a: INDArray, i: Int*): Double = a.getDouble(i: _*)

  override def get(a: INDArray, i: Long*): Double = a.getDouble(i: _*)

  override def create(arr: Array[Double]): INDArray = arr.toNDArray

  override def create(arr: Array[Double], shape: Int*): INDArray =
    arr.asNDArray(shape: _*)

  override def create(arr: Array[Double], shape: Array[Int], ordering: NDOrdering, offset: Int): INDArray =
    arr.mkNDArray(shape, ordering, offset)

  override def update(underlying: INDArray, ir: Array[IndexRange], num: Double): INDArray = {
    if (ir.length == 1 && !ir.head.hasNegative && ir.head
          .isInstanceOf[IntRange])
      underlying.putScalar(ir.head.asInstanceOf[IntRange].underlying, num)
    else if (ir.exists(_.hasNegative))
      underlying.indicesFrom(ir: _*).indices.foreach { i =>
        underlying.putScalar(i, num)
      } else
      underlying.put(underlying.getINDArrayIndexfrom(ir: _*).toArray, num)
    underlying
  }

  override def greaterThan(left: Double, right: Double): Boolean = left > right

  override def lessThan(left: Double, right: Double): Boolean = left < right

}

case object FloatNDArrayEvidence extends RealNDArrayEvidence[Float] {

  override def sum(ndarray: INDArray): Float = ndarray.sumNumber().floatValue()

  override def mean(ndarray: INDArray): Float =
    ndarray.meanNumber().floatValue()

  override def variance(ndarray: INDArray): Float =
    ndarray.varNumber().floatValue()

  override def norm2(ndarray: INDArray): Float =
    ndarray.norm2Number().floatValue()

  override def max(ndarray: INDArray): Float = ndarray.maxNumber().floatValue()

  override def product(ndarray: INDArray): Float =
    ndarray.prodNumber().floatValue()

  override def standardDeviation(ndarray: INDArray): Float =
    ndarray.stdNumber().floatValue()

  override def normMax(ndarray: INDArray): Float =
    ndarray.normmaxNumber().floatValue()

  override def min(ndarray: INDArray): Float = ndarray.minNumber().floatValue()

  override def norm1(ndarray: INDArray): Float =
    ndarray.norm1Number().floatValue()

  override def get(a: INDArray, i: Long): Float = a.getFloat(i)

  //override def get(a: INDArray, i: Long, j: Long): Float = a.getFloat(i, j)

  //override def get(a: INDArray, i: Int*): Float = a.getFloat(i: _*)

  override def get(a: INDArray, i: Long*): Float = a.getFloat(i: _*)

  override def create(arr: Array[Float]): INDArray = arr.toNDArray

  override def create(arr: Array[Float], shape: Int*): INDArray =
    arr.asNDArray(shape: _*)

  override def create(arr: Array[Float], shape: Array[Int], ordering: NDOrdering, offset: Int): INDArray =
    arr.mkNDArray(shape, ordering)

  override def update(underlying: INDArray, ir: Array[IndexRange], num: Float): INDArray = {
    if (ir.length == 1 && !ir.head.hasNegative && ir.head
          .isInstanceOf[IntRange])
      underlying.putScalar(ir.head.asInstanceOf[IntRange].underlying, num)
    else if (ir.exists(_.hasNegative))
      underlying.indicesFrom(ir: _*).indices.foreach { i =>
        underlying.putScalar(i, num)
      } else
      underlying.put(underlying.getINDArrayIndexfrom(ir: _*).toArray, num)
    underlying
  }

  override def greaterThan(left: Float, right: Float): Boolean = left > right

  override def lessThan(left: Float, right: Float): Boolean = left < right
}

trait IntegerNDArrayEvidence[Value] extends NDArrayEvidence[INDArray, Value] {
  def remainder(a: INDArray, that: INDArray): INDArray = a.remainder(that)

  def add(a: INDArray, that: INDArray): INDArray = a.add(that)

  def sub(a: INDArray, that: INDArray): INDArray = a.sub(that)

  def mul(a: INDArray, that: INDArray): INDArray = a.mul(that)

  def mmul(a: INDArray, that: INDArray): INDArray = a.mmul(that)

  def div(a: INDArray, that: INDArray): INDArray = a.div(that)

  def rdiv(a: INDArray, that: INDArray): INDArray = a.rdiv(that)

  def addi(a: INDArray, that: INDArray): INDArray = a.addi(that)

  def subi(a: INDArray, that: INDArray): INDArray = a.subi(that)

  def muli(a: INDArray, that: INDArray): INDArray = a.muli(that)

  def mmuli(a: INDArray, that: INDArray): INDArray = a.mmuli(that)

  def remainderi(a: INDArray, that: INDArray): INDArray = a.remainder(that)

  def remainderi(a: INDArray, that: Number): INDArray = a.remainderi(that)

  def divi(a: INDArray, that: INDArray): INDArray = a.divi(that)

  def rdivi(a: INDArray, that: INDArray): INDArray = a.rdivi(that)

  def remainder(a: INDArray, that: Number): INDArray = a.remainder(that)

  def add(a: INDArray, that: Number): INDArray = a.add(that)

  def sub(a: INDArray, that: Number): INDArray = a.sub(that)

  def mul(a: INDArray, that: Number): INDArray = a.mul(that)

  def div(a: INDArray, that: Number): INDArray = a.div(that)

  def rdiv(a: INDArray, that: Number): INDArray = a.rdiv(that)

  def addi(a: INDArray, that: Number): INDArray = a.addi(that)

  def subi(a: INDArray, that: Number): INDArray = a.subi(that)

  def muli(a: INDArray, that: Number): INDArray = a.muli(that)

  def divi(a: INDArray, that: Number): INDArray = a.divi(that)

  def rdivi(a: INDArray, that: Number): INDArray = a.rdivi(that)

  def put(a: INDArray, i: Int, element: INDArray): INDArray = a.put(i, element)

  def put(a: INDArray, i: Array[Int], element: INDArray): INDArray = a.put(i, element)

  def get(a: INDArray, i: INDArrayIndex*): INDArray = a.get(i: _*)

  def reshape(a: INDArray, i: Int*): INDArray = a.reshape(i.map(_.toLong): _*)

  def linearView(a: INDArray): INDArray = a.reshape(-1)

  def dup(a: INDArray): INDArray = a.dup()

  def update(a: INDArray, indices: Array[IndexRange], i: Int): INDArray =
    a.update(indices, i)

  def update(a: INDArray, indices: Array[IndexRange], i: INDArray): INDArray =
    a.update(indices, i)
}

case object IntNDArrayEvidence extends IntegerNDArrayEvidence[Int] {

  def sum(ndarray: INDArray): Int = ndarray.sumNumber().intValue()

  def mean(ndarray: INDArray): Int = ndarray.meanNumber().intValue()

  def normMax(ndarray: INDArray): Int = ndarray.normmaxNumber().intValue()

  def norm1(ndarray: INDArray): Int = ndarray.norm1Number().intValue()

  def norm2(ndarray: INDArray): Int = ndarray.norm2Number().intValue()

  def max(ndarray: INDArray): Int = ndarray.maxNumber().intValue()

  def min(ndarray: INDArray): Int = ndarray.minNumber().intValue()

  def standardDeviation(ndarray: INDArray): Int = ndarray.stdNumber().intValue()

  def product(ndarray: INDArray): Int = ndarray.prodNumber().intValue()

  def variance(ndarray: INDArray): Int = ndarray.varNumber().intValue()

  def get(a: INDArray, i: Long): Int = a.getInt(i.toInt)

  def get(a: INDArray, i: Int): Int = a.getInt(i)

  def get(a: INDArray, i: Int, j: Int): Int = a.getInt(i, j)

  def get(a: INDArray, i: Long*): Int =
    a.getInt(i.map(_.toInt): _*)

  def create(arr: Array[Int]): INDArray = arr.toNDArray

  def create(arr: Array[Int], shape: Int*): INDArray = arr.toNDArray

  def create(arr: Array[Int], shape: Array[Int], ordering: NDOrdering, offset: Int): INDArray =
    arr.mkNDArray(shape, ordering, offset)

  def greaterThan(left: Int, right: Int): Boolean = left > right

  def lessThan(left: Int, right: Int): Boolean = left < right
}

case object LongNDArrayEvidence extends IntegerNDArrayEvidence[Long] {

  def sum(ndarray: INDArray): Long = ndarray.sumNumber().longValue()

  def mean(ndarray: INDArray): Long = ndarray.meanNumber().longValue()

  def normMax(ndarray: INDArray): Long = ndarray.normmaxNumber().longValue()

  def norm1(ndarray: INDArray): Long = ndarray.norm1Number().longValue()

  def norm2(ndarray: INDArray): Long = ndarray.norm2Number().longValue()

  def max(ndarray: INDArray): Long = ndarray.maxNumber().longValue()

  def min(ndarray: INDArray): Long = ndarray.minNumber().longValue()

  def standardDeviation(ndarray: INDArray): Long = ndarray.stdNumber().longValue()

  def product(ndarray: INDArray): Long = ndarray.prodNumber().longValue()

  def variance(ndarray: INDArray): Long = ndarray.varNumber().longValue()

  def get(a: INDArray, i: Long): Long = a.getLong(i)

  def get(a: INDArray, i: Int): Long = a.getLong(i)

  def get(a: INDArray, i: Int, j: Int): Long = a.getLong(i, j)

  def get(a: INDArray, i: Long*): Long = a.getLong(i: _*)

  def create(arr: Array[Long]): INDArray = arr.toNDArray

  def create(arr: Array[Long], shape: Int*): INDArray = arr.toNDArray

  def create(arr: Array[Long], shape: Array[Int], ordering: NDOrdering, offset: Int): INDArray =
    arr.mkNDArray(shape, ordering, offset)

  def greaterThan(left: Long, right: Long): Boolean = left > right

  def lessThan(left: Long, right: Long): Boolean = left < right

  def update(a: INDArray, indices: Array[IndexRange], i: Long): INDArray =
    a.update(indices, i)
}

case object ByteNDArrayEvidence extends IntegerNDArrayEvidence[Byte] {

  def sum(ndarray: INDArray): Byte = ndarray.sumNumber().byteValue()

  def mean(ndarray: INDArray): Byte = ndarray.meanNumber().byteValue()

  def normMax(ndarray: INDArray): Byte = ndarray.normmaxNumber().byteValue()

  def norm1(ndarray: INDArray): Byte = ndarray.norm1Number().byteValue()

  def norm2(ndarray: INDArray): Byte = ndarray.norm2Number().byteValue()

  def max(ndarray: INDArray): Byte = ndarray.maxNumber().byteValue()

  def min(ndarray: INDArray): Byte = ndarray.minNumber().byteValue()

  def standardDeviation(ndarray: INDArray): Byte = ndarray.stdNumber().byteValue()

  def product(ndarray: INDArray): Byte = ndarray.prodNumber().byteValue()

  def variance(ndarray: INDArray): Byte = ndarray.varNumber().byteValue()

  def get(a: INDArray, i: Long): Byte = a.getInt(i.toInt).toByte

  def get(a: INDArray, i: Int): Byte = a.getInt(i).toByte

  def get(a: INDArray, i: Int, j: Int): Byte = a.getInt(i, j).toByte

  def get(a: INDArray, i: Long*): Byte = a.getInt(i.map(_.toInt): _*).toByte

  def create(arr: Array[Byte]): INDArray = arr.toNDArray

  def create(arr: Array[Byte], shape: Int*): INDArray = arr.toNDArray

  def create(arr: Array[Byte], shape: Array[Int], ordering: NDOrdering, offset: Int): INDArray =
    arr.mkNDArray(shape, ordering, offset)

  def greaterThan(left: Byte, right: Byte): Boolean = left > right

  def lessThan(left: Byte, right: Byte): Boolean = left < right

  def update(a: INDArray, indices: Array[IndexRange], i: Byte): INDArray =
    a.update(indices, i)
}
