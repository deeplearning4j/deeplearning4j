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

import org.nd4j.autodiff.samediff.{ SDVariable, SameDiff }
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4s.samediff.implicits.Implicits._
import org.scalatest.{ FlatSpec, Matchers }

class MathTest extends FlatSpec with Matchers {

  "SameDiff" should "allow composition of arithmetic operations" in {

    val sd = SameDiff.create()
    val ph1 = sd.placeHolder("ph1", DataType.FLOAT, 3, 4)
    val w1 = sd.bind("w1", Nd4j.rand(DataType.FLOAT, 4, 5))
    val b1 = sd.bind("b1", Nd4j.rand(DataType.FLOAT, 5))

    val mmul1 = ph1 * w1
    val badd1 = mmul1 + b1

    val loss1 = badd1.std("loss1", true)

    sd.setLossVariables("loss1")
    sd.createGradFunction
    for (v <- Array[SDVariable](ph1, w1, b1, mmul1, badd1, loss1)) {
      assert(v.getVarName != null && v.gradient != null)
    }
  }

  "SameDiff" should "provide arithmetic operations for float arguments in arbitrary order" in {

    implicit val sd = SameDiff.create()
    val w1 = sd.bind("w1", 4.0f.toScalar)
    var evaluated = w1.eval.castTo(DataType.FLOAT)
    evaluated.toFloatVector.head shouldBe 4.0f

    val w2 = w1 * 2.0f
    w2.eval.toFloatVector.head shouldBe 8.0f
    val w3 = w2 + 2.0f
    w3.eval.toFloatVector.head shouldBe 10.0f

    val w4 = 2.0f * w1
    w4.eval.toFloatVector.head shouldBe 8.0f
    val w5 = 2.0f + w2
    w5.eval.toFloatVector.head shouldBe 10.0f

    val w6 = w1 / 2.0f
    w6.eval.toFloatVector.head shouldBe 2.0f
    val w7 = w2 - 2.0f
    w7.eval.toFloatVector.head shouldBe 6.0f

    val w8 = 2.0f / w1
    w8.eval.toFloatVector.head shouldBe 2.0f

    val w9 = 2.0f - w2
    w9.eval.toFloatVector.head shouldBe 6.0f
  }

  "SameDiff" should "provide arithmetic operations for double arguments in arbitrary order" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.bind("w1", 4.0.toScalar)
    var evaluated = w1.eval.castTo(DataType.DOUBLE)
    evaluated.toFloatVector.head shouldBe 4.0

    val w2 = w1 * 2.0
    w2.eval.toFloatVector.head shouldBe 8.0
    val w3 = w2 + 2.0
    w3.eval.toFloatVector.head shouldBe 10.0

    val w4 = 2.0 * w1
    w4.eval.toFloatVector.head shouldBe 8.0
    val w5 = 2.0 + w2
    w5.eval.toFloatVector.head shouldBe 10.0

    val w6 = w1 / 2.0
    w6.eval.toFloatVector.head shouldBe 2.0
    val w7 = w2 - 2.0
    w7.eval.toFloatVector.head shouldBe 6.0

    val w8 = 2.0 / w1
    w8.eval.toFloatVector.head shouldBe 2.0
    val w9 = 2.0 - w2
    w9.eval.toFloatVector.head shouldBe 6.0f
  }

  "SameDiff" should "provide floor division" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.bind("w1", 4.0.toScalar)
    val w2 = sd.bind("w2", 1.2.toScalar)
    val w3 = w1 `//` w2
    w3.eval.toFloatVector.head shouldBe 3.0

    val w4 = w1 `//` 1.5
    w4.eval.toFloatVector.head shouldBe 2.0

    val w5 = 9.5 `//` w1
    w5.eval.toFloatVector.head shouldBe 2.0
  }

  "SameDiff" should "provide remainder division" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.bind("w1", 40.0.toScalar)
    val w2 = sd.bind("w2", 12.0.toScalar)
    val w3 = w2 % w1
    w3.eval.toFloatVector.head shouldBe 12.0
    val w4 = w1 % w2
    w4.eval.toFloatVector.head shouldBe 4.0

    val w5 = w1 % 15.0
    w5.eval.toFloatVector.head shouldBe 10.0

    val w6 = 10.0 % w1
    w6.eval.toFloatVector.head shouldBe 10.0
  }

  "SameDiff" should "provide unary math operators" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.bind("w1", 4.0.toScalar)
    var evaluated = w1.eval.castTo(DataType.DOUBLE)
    evaluated.toFloatVector.head shouldBe 4.0

    val w2 = -w1
    var evaluated2 = w2.eval.castTo(DataType.DOUBLE)
    evaluated2.toFloatVector.head shouldBe -4.0

    val w3 = w1 ** 2
    var evaluated3 = w3.eval.castTo(DataType.DOUBLE)
    evaluated3.toFloatVector.head shouldBe 16.0
  }

  "SameDiff" should "provide boolean logic operators" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.constant(Nd4j.scalar(true))
    val w2 = sd.constant(Nd4j.scalar(true))

    val w3 = w1 | w2
    w3.eval.toIntVector.head shouldBe 1

    val w4 = w1 & w2
    w4.eval.toIntVector.head shouldBe 1

    val w5 = w1 ^ w2
    w5.eval.toIntVector.head shouldBe 0

    val w6 = w1 | false
    w6.eval.toIntVector.head shouldBe 1

    val w7 = w1 & false
    w7.eval.toIntVector.head shouldBe 0

    val w8 = w1 ^ false
    w8.eval.toIntVector.head shouldBe 1

    val w9 = false | w1
    w9.eval.toIntVector.head shouldBe 1

    val w10 = false & w1
    w10.eval.toIntVector.head shouldBe 0

    val w11 = false ^ w1
    w11.eval.toIntVector.head shouldBe 1
  }

  "SameDiff" should "provide shifting operations" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.constant(16)

    val w2 = w1 << 2
    w2.eval.toIntVector.head shouldBe 64

    val w3 = w1 >> 2
    w3.eval.toIntVector.head shouldBe 4
  }

  "SameDiff" should "provide shifting operations with SDVariable argument" in {
    implicit val sd = SameDiff.create()
    val w1 = sd.constant(16)
    val two = sd.constant(2)

    val w2 = w1 << two
    w2.eval.toIntVector.head shouldBe 64

    val w3 = w1 >> two
    w3.eval.toIntVector.head shouldBe 4
  }
}
