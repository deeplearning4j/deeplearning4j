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
package org.deeplearning4j.scalnet.models

import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.core.{ Dense, OutputLayer }
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.layers.reshaping.{ Flatten3D, Unflatten3D }
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest._

/**
  * Created by maxpumperla on 29/06/17.
  */
class SequentialTest extends FunSpec with BeforeAndAfter {

  var model: Sequential = Sequential()
  val shape = 100
  val wrongInputShape = 10

  val height: Int = 28
  val width: Int = 28
  val channels: Int = 1
  val nClasses: Int = 10

  val weightDecay: Double = 0.005

  before {
    model = Sequential()
  }

  describe("A Sequential network") {

    it("without layers should produce an IllegalArgumentException when compiled") {
      assertThrows[java.lang.IllegalArgumentException] {
        model.compile(null)
      }
    }

    it("without buildOutput called should not have an output layer") {
      model.add(Dense(shape, shape))
      assert(!model.getLayers.last.asInstanceOf[OutputLayer].output.isOutput)
    }

    it("with buildOutput called should have an output layer") {
      model.add(Dense(shape, shape))
      model.buildOutput(LossFunction.NEGATIVELOGLIKELIHOOD)
      assert(model.getLayers.last.asInstanceOf[OutputLayer].output.isOutput)
    }

    it("should infer the correct shape of an incorrectly initialized layer") {
      model.add(Dense(shape, shape))
      model.add(Dense(shape, wrongInputShape))
      assert(model.getLayers.last.inputShape == List(shape))
    }

    it("should propagate the correct shape of all layers and preprocessors") {
      model.add(Unflatten3D(List(height, width, channels), nIn = height * width))
      model.add(Convolution2D(20, List(5, 5), channels, regularizer = L2(weightDecay), activation = Activation.RELU))
      model.add(MaxPooling2D(List(2, 2), List(2, 2)))

      model.add(Convolution2D(50, List(5, 5), regularizer = L2(weightDecay), activation = Activation.RELU))
      model.add(MaxPooling2D(List(2, 2), List(2, 2)))
      model.add(Flatten3D())

      val preprocessorOutShapes = model.getPreprocessors.values.map(_.outputShape)
      assert(preprocessorOutShapes == List(List(height, width, channels), List(4 * 4 * 50)))

      val layerOutShapes = model.getLayers.map(_.outputShape)
      assert(layerOutShapes == List(List(24, 24, 20), List(12, 12, 20), List(8, 8, 50), List(4, 4, 50)))

    }
  }
}
