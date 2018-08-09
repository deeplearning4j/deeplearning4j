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
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor
import org.deeplearning4j.scalnet.layers.core.Preprocessor

/**
  * Flattens structured image-like inputs into vector. Input should have
  * three dimensions: height (number of rows), width (number of columns),
  * and number of channels.
  *
  * @author David Kale
  */
class Flatten3D(nIn: List[Int] = List(0, 0, 0)) extends Preprocessor {

  override val inputShape: List[Int] = nIn
  override val outputShape = List(inputShape.product)
  override val name = "Flatten3D"

  override def reshapeInput(newIn: List[Int]): Flatten3D =
    new Flatten3D(newIn)

  override def compile: InputPreProcessor = {
    if (inputShape.length != 3) {
      throw new IllegalArgumentException("Input shape must be length 3.")
    }
    new CnnToFeedForwardPreProcessor(inputShape.head, inputShape.tail.head, inputShape.last)
  }
}

object Flatten3D {
  def apply(nIn: List[Int] = List(0, 0, 0)): Flatten3D = new Flatten3D(nIn)
}
