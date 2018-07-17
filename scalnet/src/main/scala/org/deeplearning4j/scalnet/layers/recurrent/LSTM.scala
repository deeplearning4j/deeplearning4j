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

package org.deeplearning4j.scalnet.layers.recurrent

import org.deeplearning4j.nn.conf.layers
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.core.Layer
import org.deeplearning4j.scalnet.regularizers.{ NoRegularizer, WeightRegularizer }
import org.nd4j.linalg.activations.Activation

class LSTM(nIn: Int,
           nOut: Int,
           activation: Activation,
           forgetGateBiasInit: Double,
           gateActivation: Activation,
           weightInit: WeightInit,
           regularizer: WeightRegularizer,
           dropOut: Double,
           override val name: String = "")
    extends Layer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    new layers.LSTM.Builder()
      .nIn(nIn)
      .nOut(nOut)
      .activation(activation)
      .forgetGateBiasInit(forgetGateBiasInit)
      .gateActivationFunction(gateActivation)
      .weightInit(weightInit)
      .l1(regularizer.l1)
      .l2(regularizer.l2)
      .dropOut(dropOut)
      .name(name)
      .build()

  override def inputShape: List[Int] = List(nIn, nOut)

  override def outputShape: List[Int] = List(nOut, nIn)

}

object LSTM {
  def apply(nIn: Int,
            nOut: Int,
            activation: Activation = Activation.IDENTITY,
            forgetGateBiasInit: Double = 1.0,
            gateActivationFn: Activation = Activation.SIGMOID,
            weightInit: WeightInit = WeightInit.XAVIER,
            regularizer: WeightRegularizer = NoRegularizer(),
            dropOut: Double = 0.0): LSTM =
    new LSTM(
      nIn,
      nOut,
      activation,
      forgetGateBiasInit,
      gateActivationFn,
      weightInit,
      regularizer,
      dropOut
    )
}
