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
import org.deeplearning4j.nn.conf.preprocessor.ReshapePreProcessor
import org.deeplearning4s.layers.{Node, Preprocessor}


class Reshape(
  nOut: List[Int],
  nIn: List[Int] = List())
  extends Node with Preprocessor {
  inputShape = nIn
  _outputShape = nOut

  override def outputShape: List[Int] = _outputShape

  override def compile: InputPreProcessor = {
    if (inputShape.product != outputShape.product)
      throw new IllegalArgumentException("Overall input shape must equal overall output shape.")

    new ReshapePreProcessor(inputShape.toArray, outputShape.toArray)
  }
}
