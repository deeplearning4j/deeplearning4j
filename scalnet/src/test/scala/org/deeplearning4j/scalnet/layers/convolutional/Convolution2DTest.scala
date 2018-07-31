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
package org.deeplearning4j.scalnet.layers.convolutional

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class Convolution2DTest extends FunSpec {

  describe("A 2D convolutional layer with 20 filters and kernel size (5,5)") {
    val nFilter = 20
    val kernelSize = List(5, 5)
    val conv = Convolution2D(nFilter, kernelSize)
    it("should have inputShape List(0)") {
      assert(conv.inputShape == List(0))
    }
    it("should have an outputShape of List(20)") {
      assert(conv.outputShape == List(nFilter))
    }
    it("should accept a new input shape when provided") {
      val reshapedConv = conv.reshapeInput(List(1, 2, 3))
      assert(reshapedConv.inputShape == List(1, 2, 3))
    }
    it("should become a DL4J convolution layer when compiled") {
      val compiledConv = conv.compile
      assert(compiledConv.isInstanceOf[ConvolutionLayer])
    }
  }
}
