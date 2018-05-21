/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.layers.reshaping

import org.deeplearning4j.nn.conf.InputPreProcessor
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class ReshapeTest extends FunSpec {

  describe("A reshape layer with in and out shapes") {
    val inShape = List(20, 10)
    val outShape = List(10, 20)
    val reshape = Reshape(outShape, inShape)
    it("should have inputShape as provided") {
      assert(reshape.inputShape == inShape)
    }
    it("should have outputShape as provided") {
      assert(reshape.outputShape == outShape)
    }
    it("should accept a new input shape when provided") {
      val reshaped = reshape.reshapeInput(List(10, 2, 10))
      assert(reshaped.inputShape == List(10, 2, 10))
    }
    it("should become a DL4J InputPreProcessor when compiled") {
      val compiledReshape = reshape.compile
      assert(compiledReshape.isInstanceOf[InputPreProcessor])
    }
  }
}
