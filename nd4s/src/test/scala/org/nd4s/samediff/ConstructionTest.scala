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

import org.nd4j.autodiff.samediff.{ SDIndex, SDVariable, SameDiff, TrainingConfig }
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4s.Implicits._
import org.nd4s.samediff.implicits.Implicits._
import org.scalatest.{ FlatSpec, Matchers }

class ConstructionTest extends FlatSpec with Matchers {

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

  "classification example" should "work" in {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE)
    val seed = 1000
    val learning_rate = 0.1
    val x1_label1 = Nd4j.rand(3, 1)
    val x2_label1 = Nd4j.rand(2, 1)
    val x1_label2 = Nd4j.rand(7, 1)
    val x2_label2 = Nd4j.rand(6, 1)

    // np.append, was not able to guess proper method
    val x1s = Nd4j.concat(0, x1_label1, x1_label2)
    val x2s = Nd4j.concat(0, x2_label1, x2_label2)

    // Must have implicit sd here for some ops, inobvious
    implicit val sd = SameDiff.create
    val ys = (Nd4j.scalar(0.0) * x1_label1.length()) + (Nd4j.scalar(1.0) * x1_label2.length())

    // empty sequence
    val X1 = sd.placeHolder("x1", DataType.FLOAT, 10, 1)
    val X2 = sd.placeHolder("x2", DataType.FLOAT, 8, 1)
    val y = sd.placeHolder("y", DataType.FLOAT)
    // There's no option to pass Trainable=True
    val w = sd.`var`("w", DataType.FLOAT, 0, 0, 0)

    // tf.math.sigmoid -> where can I get sigmoid ? sd.nn : Transform
    val tmp = w.get(SDIndex.point(1)) * (X1 + w.get(SDIndex.point(0)))
    val tmp2 = sd.math.neg(w.get(SDIndex.point(2)) * (X2 + tmp))
    val y_model = sd.nn.sigmoid(tmp2)

    // what is target for reduce_mean? What is proper replacement for np.reduce_mean?
    // 1 - SDVariable - doesn't work as is
    // lost in long formula!!!
    // java.lang.IllegalStateException: Only floating point types are supported for strict tranform ops - got INT - log
    val cost_fun = -sd.math.log(y_model) * y - sd.math.log(sd.constant(1.0) - y_model) * (sd.constant(1.0) - y)
    val loss = cost_fun.mean("loss")

    val updater = new Sgd(learning_rate)
    // mapping between values and ph

    sd.setLossVariables("loss")
    val conf = new TrainingConfig.Builder()
      .updater(updater)
      //.minimize("loss")
      .dataSetFeatureMapping("x1", "x2", "y")
      .markLabelsUnused()
      .build()

    val mds = new MultiDataSet(Array[INDArray](x1s, x2s, ys), new Array[INDArray](0))

    sd.setTrainingConfig(conf)
    sd.fit(new SingletonMultiDataSetIterator(mds), 1)

    import org.nd4j.linalg.api.ndarray.INDArray
    Console.print(sd.outputs())
  }
}
