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

package org.deeplearning4j.scalnet.layers

import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.nn.conf.layers.{ DenseLayer, OutputLayer => JOutputLayer }
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 29/06/17.
  */
class DenseTest extends FunSpec {

  describe("A dense layer of size 100") {
    val nOut = 100
    val dense = Dense(nOut)
    it("should have inputShape List(0)") {
      assert(dense.inputShape == List(0))
    }
    it("should have an outputShape of List(100)") {
      assert(dense.outputShape == List(nOut))
    }
    it("when calling toOutputLayer without proper loss, it does not become an output layer") {
      val denseOut = dense.toOutputLayer(null)
      assert(!denseOut.output.isOutput)
      val compiledLayer = denseOut.compile
      assert(compiledLayer.isInstanceOf[DenseLayer])
    }
    it("when calling toOutputLayer, it becomes an output layer") {
      val denseOut = dense.toOutputLayer(LossFunction.NEGATIVELOGLIKELIHOOD)
      assert(denseOut.output.isOutput)
      val compiledLayer = denseOut.compile
      assert(compiledLayer.isInstanceOf[JOutputLayer])
    }
  }
}
