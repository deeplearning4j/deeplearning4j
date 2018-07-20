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

package org.deeplearning4j.scalnet.layers.reshaping

import org.deeplearning4j.nn.conf.InputPreProcessor
import org.scalatest.FunSpec

/**
  * Created by maxpumperla on 19/07/17.
  */
class Flatten3DTest extends FunSpec {

  describe("A 3D flatten layer with output dim 20") {
    val outShape = List(20)
    val flatten = Flatten3D(outShape)
    it("should have output shape as input shape") {
      assert(flatten.inputShape == outShape)
    }
    it("should have outputShape as provided") {
      assert(flatten.outputShape == outShape)
    }
    it("should accept a new input shape when provided") {
      val reshapedFlatten = flatten.reshapeInput(List(10, 2, 10))
      assert(reshapedFlatten.inputShape == List(10, 2, 10))
    }
    it("should not compile when input shape is not 3D") {
      assertThrows[java.lang.IllegalArgumentException] {
        flatten.compile
      }
    }
    it("should become a DL4J InputPreProcessor when compiled") {
      val reshapedFlatten = flatten.reshapeInput(List(10, 2, 10))
      val compiledFlatten = reshapedFlatten.compile
      assert(compiledFlatten.isInstanceOf[InputPreProcessor])
    }
  }
}
