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
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor
import org.deeplearning4j.scalnet.layers.core.Preprocessor

/**
  * Unflattens vector into structured image-like output. Input must be a
  * vector while output should have three dimensions: height (number of rows),
  * width (number of columns), and number of channels.
  *
  * @author David Kale
  */
class Unflatten3D(newOutputShape: List[Int], nIn: Int = 0) extends Preprocessor {
  if (newOutputShape.length != 3) {
    throw new IllegalArgumentException("New output shape must be length 3.")
  }
  override val outputShape: List[Int] = newOutputShape
  override val inputShape: List[Int] = List(nIn)
  override val name = "Unflatten3D"

  override def reshapeInput(newIn: List[Int]): Unflatten3D =
    new Unflatten3D(newOutputShape, newIn.head)

  override def compile: InputPreProcessor = {
    if (PartialFunction.cond(inputShape) { case Nil => true; case 0 :: Nil => true }) {
      throw new IllegalArgumentException("Input shape must be nonempty and nonzero.")
    }
    if (inputShape.last != outputShape.product) {
      throw new IllegalStateException("Overall output shape must be equal to original input shape.")
    }
    new FeedForwardToCnnPreProcessor(outputShape.head, outputShape.tail.head, outputShape.last)
  }
}

object Unflatten3D {
  def apply(newOutputShape: List[Int], nIn: Int = 0): Unflatten3D =
    new Unflatten3D(newOutputShape, nIn)
}
