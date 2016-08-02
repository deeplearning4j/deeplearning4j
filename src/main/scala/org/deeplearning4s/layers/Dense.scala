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

package org.deeplearning4s.layers

import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit

/**
  * Abstract base class for layers in neural net architectures.
  *
  * @author David Kale
  */
class Dense(
    nOut: Int,
    nIn: Int = 0,
    val weightInit: WeightInit = WeightInit.VI,
    val activation: String = "identity")
  extends Node with Layer with Output {

  _outputShape = List(nOut)
  if (nIn > 0)
    _inputShape = List(nIn)

  override def outputShape = _outputShape

  override def compile: org.deeplearning4j.nn.conf.layers.Layer = {
    if (inputShape.isEmpty)
      throw new IllegalArgumentException("Input shape must be nonempty.")

    if (isOutput) {
      new OutputLayer.Builder(_lossFunction)
        .nIn(inputShape.last)
        .nOut(outputShape.last)
        .activation(activation)
        .weightInit(weightInit)
        .build()
    } else {
      new DenseLayer.Builder()
        .nIn(inputShape.last)
        .nOut(outputShape.last)
        .activation(activation)
        .weightInit(weightInit)
        .build()
    }
  }
}
