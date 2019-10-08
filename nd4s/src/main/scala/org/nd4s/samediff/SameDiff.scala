/*******************************************************************************
  * Copyright (c) 2015-2019 Skymind, Inc.
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
package org.nd4s.samediff

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.autodiff.samediff.{ SDIndex, SDVariable, SameDiff }
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
  * Provides wrappers for nd4j SameDiff and related classes.
  *
  * Wrappers are designed to be used implicitly, client code
  * should be similar to nd4j with additional syntactic sugar
  * and Scala specific stuff.
  *
  * @author Alexander Stoyakin
  */
class SameDiffWrapper {

  var sd: SameDiff = SameDiff.create()

  def this(sd: SameDiff) {
    this
    this.sd = sd
  }

  def bind(name: String, data: INDArray): SDVariable =
    sd.`var`(name, data)

  def bind(name: String, dataType: DataType, shape: Array[Long]): SDVariable =
    sd.`var`(name, dataType, shape: _*)

  def bind(data: INDArray): SDVariable =
    sd.`var`("", data)

  def bind(name: String, dataType: DataType, shape: Array[Int]): SDVariable =
    sd.`var`(name, dataType, shape: _*)

  def placeHolder(name: String, dataType: DataType, shape: Long*): SDVariable =
    sd.placeHolder(name, dataType, shape: _*)
}

case class SDIndexWrapper(start: Long) {

  def ::(end: Long): SDIndex =
    SDIndex.interval(start, end)
}

object --- extends SDIndex {
  val thisIndex: SDIndex = SDIndex.all()
}

class SDVariableWrapper {

  var thisVariable: SDVariable = null
  var isScalar: Boolean = false
  val --- : SDIndex = SDIndex.all()

  def this(variable: SDVariable) {
    this
    thisVariable = variable
  }

  def apply(index: Long): SDVariable = thisVariable.get(SDIndex.point(index))

  def apply(index: SDIndex*): SDVariable = thisVariable.get(index: _*)

  /*def apply(x: SDIndex, y: SDIndex): SDVariable =
    (x, y) match {
      case (_, y) => thisVariable.get(SDIndex.all(), y)
      case (x, _) => thisVariable.get(x, SDIndex.all())
      case (_, _) => thisVariable.get(SDIndex.all(), SDIndex.all())
      case (x, y) => thisVariable.get(x, y)
    }*/

  def add(other: Double): Unit = thisVariable.add(other)

  def *(other: SDVariable): SDVariable =
    thisVariable.mul(other)

  def +(other: SDVariable): SDVariable =
    thisVariable.add(other)

  def /(other: SDVariable): SDVariable =
    if (isScalar)
      thisVariable.rdiv(other)
    else
      thisVariable.rdiv(other)

  def -(other: SDVariable): SDVariable =
    if (isScalar)
      thisVariable.rsub(other)
    else
      thisVariable.sub(other)

  def %(other: SDVariable): SDVariable = thisVariable.mod(null, other)

  def `//`(other: SDVariable): SDVariable = thisVariable.fdiv(null, other)

  def unary_-(): SDVariable = thisVariable.neg

  def ^(other: SDVariable)(implicit sameDiff: SameDiff): SDVariable = sameDiff.math.xor(thisVariable, other)
  def |(other: SDVariable)(implicit sameDiff: SameDiff): SDVariable = sameDiff.math.or(thisVariable, other)
  def &(other: SDVariable)(implicit sameDiff: SameDiff): SDVariable = sameDiff.math.and(thisVariable, other)

  def <<(other: SDVariable)(implicit sameDiff: SameDiff): SDVariable = sameDiff.math.bitShift(null, thisVariable, other)
  def >>(other: SDVariable)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.bitShiftRight(null, thisVariable, other)
  def <<(x: Int)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.bitShift(null, thisVariable, sameDiff.constant(x))
  def >>(x: Int)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.bitShiftRight(null, thisVariable, sameDiff.constant(x))

  // Overloads for numeric arguments
  // Float
  def *(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.mul(sameDiff.constant(other))

  def +(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.add(sameDiff.constant(other))

  def -(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    if (isScalar)
      thisVariable.rsub(sameDiff.constant(other))
    else
      thisVariable.sub(sameDiff.constant(other))

  def /(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    if (isScalar)
      thisVariable.rdiv(sameDiff.constant(other))
    else
      thisVariable.div(sameDiff.constant(other))

  def %(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.mod(null, sameDiff.constant(other))

  def `//`(other: Float)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.fdiv(null, sameDiff.constant(other))

  //Double
  def *(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.mul(sameDiff.constant(other))

  def +(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.add(sameDiff.constant(other))

  def -(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    if (isScalar)
      thisVariable.rsub(sameDiff.constant(other))
    else
      thisVariable.sub(sameDiff.constant(other))

  def /(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    if (isScalar)
      thisVariable.rdiv(sameDiff.constant(other))
    else
      thisVariable.div(sameDiff.constant(other))

  def %(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.mod(null, sameDiff.constant(other))

  def `//`(other: Double)(implicit sameDiff: SameDiff): SDVariable =
    thisVariable.fdiv(null, sameDiff.constant(other))

  // Int
  def **(x: Int): SDVariable =
    thisVariable.pow(x)

  def ^(other: Boolean)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.xor(thisVariable, sameDiff.constant(Nd4j.scalar(other)))
  def |(other: Boolean)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.or(thisVariable, sameDiff.constant(Nd4j.scalar(other)))
  def &(other: Boolean)(implicit sameDiff: SameDiff): SDVariable =
    sameDiff.math.and(thisVariable, sameDiff.constant(Nd4j.scalar(other)))
}
