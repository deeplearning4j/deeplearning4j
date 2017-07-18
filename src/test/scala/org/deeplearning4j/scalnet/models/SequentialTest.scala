/*
 *
 *  * Copyright 2017 Skymind,Inc.
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
 */

package org.deeplearning4j.scalnet.models

import org.deeplearning4j.nn.weights.WeightInit
import org.scalatest.FunSpec
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.layers.reshaping.{Flatten3D, Unflatten3D}
import org.deeplearning4j.scalnet.regularizers.L2

/**
  * Created by maxpumperla on 29/06/17.
  */
class SequentialTest extends FunSpec {

  describe("A Sequential network") {
    describe("without layers") {
      val model: Sequential = new Sequential
      it("should produce NoSuchElementException when compiled") {
        assertThrows[scala.MatchError] {
          model.compile(null)
        }
      }
    }

    describe("when adding layers it") {
      val model: Sequential = new Sequential
      val shape = 100
      val wrongInputShape = 10
      model.add(Dense(shape, shape))
      model.add(Dense(shape, wrongInputShape))

      it("should infer the correct shape") {
        assert(model.getLayers.last.inputShape == List(shape))
      }
    }

    describe("when building a more complex model it") {
      val nbRows: Int = 28
      val nbColumns: Int = 28
      val nbChannels: Int = 1
      val nbOutput: Int = 10
      val weightDecay: Double = 0.0005
      val momentum: Double = 0.9
      val learningRate: Double = 0.01

      val model: Sequential = new Sequential
      model.add(Unflatten3D(List(nbRows, nbColumns, nbChannels), nIn = nbRows * nbColumns))
      model.add(Convolution2D(nFilter = 20, kernelSize = List(5, 5), stride = List(1, 1),
        weightInit = WeightInit.XAVIER, regularizer = L2(weightDecay)))
      model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
      model.add(Convolution2D(nFilter = 50, kernelSize = List(5, 5), stride = List(1, 1),
        weightInit = WeightInit.XAVIER, regularizer = L2(weightDecay)))
      model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
      model.add(Flatten3D())

      it("should propagate the correct shape") {
        assert(model.getLayers.last.inputShape == List(8, 8, 50))
      }
    }

  }
}