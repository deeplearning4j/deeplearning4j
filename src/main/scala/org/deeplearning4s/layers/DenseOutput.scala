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

import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4s.regularizers.{NoRegularizer, WeightRegularizer}
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Fully connected output neural net layer.
  *
  * @author David Kale
  */
class DenseOutput(
    nOut: Int,
    activation: String,
    lossFunction: LossFunction,
    nIn: Int = 0,
    weightInit: WeightInit = WeightInit.VI,
    regularizer: WeightRegularizer = NoRegularizer(),
    dropOut: Double = 0.0,
    override val name: String = null)
  extends Dense(nOut, nIn, weightInit, activation, regularizer, dropOut) {

  makeOutput(lossFunction)
}
