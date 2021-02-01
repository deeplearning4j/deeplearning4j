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

package org.nd4s.samediff

import java.lang.reflect.Field
import java.util
import java.util.{ Arrays, Collections, HashMap, List, Map }

import org.nd4j.shade.guava.collect.{ Lists, Maps }
import org.junit.Assert._
import org.junit.Assume.assumeNotNull
import org.nd4j.autodiff.samediff._
import org.nd4j.autodiff.samediff.impl.DefaultSameDiffConditional
import org.nd4j.autodiff.validation.{ OpValidation, TestCase }
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.blas.params.MMulTranspose
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.{ Conv2DConfig, LocalResponseNormalizationConfig }
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax
import org.nd4j.linalg.api.ops.impl.transforms.custom.{ Max, Min }
import org.nd4j.linalg.api.ops.impl.transforms.custom._
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution
import org.nd4j.linalg.api.shape.LongShapeDescriptor
import org.nd4j.linalg.checkutil.NDArrayCreationUtil
import org.nd4j.linalg.dataset.{ DataSet, MultiDataSet }
import org.nd4j.linalg.dataset.adapter.SingletonMultiDataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.indexing.NDArrayIndex.all
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.weightinit.impl.{ OneInitScheme, UniformInitScheme, ZeroInitScheme }
import org.nd4s.samediff.implicits.Implicits._
import org.scalatest.{ FlatSpec, Matchers }
import scala.collection.JavaConversions._

class SameDiffTest extends FlatSpec with Matchers {

  "SameDiff" should "allow Mse backwards execution" in {

    implicit val sd: SameDiff = SameDiff.create

    val nOut: Int = 4
    val minibatch: Int = 3
    val input: SDVariable = sd.bind("in", DataType.FLOAT, Array[Long](minibatch, nOut))
    val label: SDVariable = sd.bind("label", DataType.FLOAT, Array[Long](minibatch, nOut))

    val diff: SDVariable = input - label
    val sqDiff: SDVariable = diff * diff
    //val sqDiff: SDVariable = diff ** 2
    val msePerEx: SDVariable = sd.mean("msePerEx", sqDiff, 1)
    val avgMSE: SDVariable = sd.mean("loss", msePerEx, 0)

    val inputArr: INDArray = Nd4j.rand(DataType.FLOAT, minibatch, nOut)
    val labelArr: INDArray = Nd4j.rand(DataType.FLOAT, minibatch, nOut)

    sd.associateArrayWithVariable(inputArr, input)
    sd.associateArrayWithVariable(labelArr, label)

    val result = sd.output(null: java.util.Map[String, org.nd4j.linalg.api.ndarray.INDArray], "loss")
    assertEquals(1, result.values().size())

    val emptyMap = new HashMap[String, INDArray]()
    sd.output(emptyMap, "loss")
  }

  "SameDiff" should "run test dense layer forward pass" in {
    Nd4j.getRandom.setSeed(12345)
    implicit val sd = SameDiff.create
    val iInput = Nd4j.rand(3, 4)
    val iWeights = Nd4j.rand(4, 5)
    val iBias = Nd4j.rand(1, 5)
    val input = sd.bind("input", iInput)
    val weights = sd.bind("weights", iWeights)
    val bias = sd.bind("bias", iBias)
    val mmul = sd.mmul("mmul", input, weights)

    val z = mmul + bias

    val out = sd.nn.sigmoid("out", z)
    val expMmul = iInput.mmul(iWeights)
    val expZ = expMmul.addRowVector(iBias)
    val expOut = Transforms.sigmoid(expZ, true)
    sd.output(new HashMap[String, INDArray](), "mmul", "out", "bias", "add")
    assertEquals(expMmul, mmul.getArr)
    assertEquals(expZ, z.getArr)
    assertEquals(expOut, out.getArr)
  }

  "SameDiff" should "convert placeholder to constant" in {
    Nd4j.getRandom.setSeed(12345)
    val sd = SameDiff.create
    val in = sd.placeHolder("in", DataType.FLOAT, 1, 3)
    val in2 = sd.placeHolder("in2", DataType.FLOAT, 3, 4)
    val b = sd.bind("b", Nd4j.rand(DataType.FLOAT, 1, 4))
    val mmul = in.mmul(in2)
    val add = mmul + b
    val tanh = sd.math.tanh(add)
    val loss = sd.variance(tanh, true)
    val inArr = Nd4j.rand(DataType.FLOAT, 1, 3)
    in.setArray(inArr)
    val inArr2 = Nd4j.rand(DataType.FLOAT, 3, 4)
    val c = TrainingConfig.builder
      .updater(new Adam(0.1))
      .weightDecay(0.01, true)
      .dataSetFeatureMapping("in", "in2")
      .skipBuilderValidation(true)
      .build

    val data = new HashMap[String, INDArray]()
    data.put("in", Nd4j.randn(1, 3))
    data.put("in2", Nd4j.randn(3, 4))
    in.convertToConstant
    val out = sd.output(data, "tanh")
    val out2 = sd.output(data, "tanh")
    assertEquals(out, out2)
    assertEquals(VariableType.CONSTANT, in.getVariableType)
    assertEquals(inArr, in.getArr)
    //Sanity check on fitting:
    sd.setTrainingConfig(c)
    sd.fit(new SingletonMultiDataSetIterator(new MultiDataSet(Array[INDArray](inArr, inArr2), null)), 1)
  }
}
