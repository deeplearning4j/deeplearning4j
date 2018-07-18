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

package org.deeplearning4j.scalnet.layers.core

import org.deeplearning4j.nn.conf.layers.{ DenseLayer, OutputLayer => JOutputLayer }
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.regularizers.{ NoRegularizer, WeightRegularizer }
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Fully connected neural net layer.
  *
  * @author David Kale
  */
class Dense(nOut: List[Int],
            nIn: List[Int],
            weightInit: WeightInit,
            activation: Activation,
            regularizer: WeightRegularizer,
            dropOut: Double,
            override val name: String,
            lossFunction: Option[LossFunction])
    extends OutputLayer {

  // Make this an output layer if lossFunction is defined.
  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    if (output.isOutput) {
      new JOutputLayer.Builder(output.lossFunction)
        .nIn(inputShape.last)
        .nOut(outputShape.last)
        .weightInit(weightInit)
        .activation(activation)
        .l1(regularizer.l1)
        .l2(regularizer.l2)
        .dropOut(dropOut)
        .name(name)
        .build()
    } else {
      new DenseLayer.Builder()
        .nIn(inputShape.last)
        .nOut(outputShape.last)
        .weightInit(weightInit)
        .activation(activation)
        .l1(regularizer.l1)
        .l2(regularizer.l2)
        .dropOut(dropOut)
        .name(name)
        .build()
    }

  override val outputShape: List[Int] = nOut

  override val inputShape: List[Int] = nIn

  override val output: Output =
    Output(isOutput = lossFunction.isDefined, lossFunction = lossFunction.orNull)

  override def reshapeInput(newIn: List[Int]): Dense =
    new Dense(nOut, newIn, weightInit, activation, regularizer, dropOut, name, lossFunction)

  override def toOutputLayer(lossFunction: LossFunctions.LossFunction): OutputLayer =
    new Dense(nOut, nIn, weightInit, activation, regularizer, dropOut, name, Some(lossFunction))
}

object Dense {
  def apply(nOut: Int,
            nIn: Int = 0,
            weightInit: WeightInit = WeightInit.XAVIER_UNIFORM,
            activation: Activation = Activation.IDENTITY,
            regularizer: WeightRegularizer = NoRegularizer(),
            dropOut: Double = 0.0,
            name: String = "",
            lossFunction: Option[LossFunction] = None): Dense =
    new Dense(List(nOut), List(nIn), weightInit, activation, regularizer, dropOut, name, lossFunction)
}
