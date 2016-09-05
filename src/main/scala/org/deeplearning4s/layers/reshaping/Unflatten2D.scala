/*
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4s.layers.reshaping

import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor
import org.deeplearning4s.layers.{Node, Preprocessor}


class Unflatten2D(
  nOut: List[Int],
  nIn: Int = 0)
  extends Node with Preprocessor {
  _outputShape = nOut
  if (nIn > 0)
    inputShape = List(nIn)

  override def outputShape: List[Int] = _outputShape

  override def compile: InputPreProcessor = {
    if (inputShape.isEmpty)
      throw new IllegalArgumentException("Input shape must be nonempty.")

    if (inputShape.last != outputShape.product)
      throw new IllegalStateException("Overall output shape must be equal to original input shape.")

    new FeedForwardToCnnPreProcessor(outputShape.head, outputShape(1), outputShape(2))
  }
}
