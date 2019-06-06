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
import org.deeplearning4j.nn.conf.layers.{ OutputLayer => JOutputLayer }
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.scalnet.layers.core.{ Layer, Output, OutputLayer }
import org.deeplearning4j.scalnet.regularizers.{ NoRegularizer, WeightRegularizer }
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

class RnnOutputLayer(nIn: Int,
                     nOut: Int,
                     activation: Activation,
                     loss: Option[LossFunction],
                     weightInit: WeightInit,
                     regularizer: WeightRegularizer,
                     dropOut: Double,
                     override val name: String = "")
    extends Layer
    with OutputLayer {

  override def compile: org.deeplearning4j.nn.conf.layers.Layer =
    Option(output.lossFunction) match {
      case None =>
        new layers.RnnOutputLayer.Builder()
          .nIn(nIn)
          .nOut(nOut)
          .activation(activation)
          .weightInit(weightInit)
          .l1(regularizer.l1)
          .l2(regularizer.l2)
          .dropOut(dropOut)
          .name(name)
          .build()
      case _ =>
        new layers.RnnOutputLayer.Builder(output.lossFunction)
          .nIn(nIn)
          .nOut(nOut)
          .activation(activation)
          .weightInit(weightInit)
          .l1(regularizer.l1)
          .l2(regularizer.l2)
          .lossFunction(output.lossFunction)
          .dropOut(dropOut)
          .name(name)
          .build()
    }

  override val inputShape: List[Int] = List(nIn, nOut)

  override val outputShape: List[Int] = List(nOut, nIn)

  override val output: Output = Output(isOutput = loss.isDefined, lossFunction = loss.orNull)

  override def toOutputLayer(lossFunction: LossFunctions.LossFunction): OutputLayer =
    new RnnOutputLayer(
      nIn,
      nOut,
      activation,
      Some(lossFunction),
      weightInit,
      regularizer,
      dropOut
    )
}

object RnnOutputLayer {
  def apply(nIn: Int,
            nOut: Int,
            activation: Activation,
            loss: Option[LossFunction] = None,
            weightInit: WeightInit = WeightInit.XAVIER,
            regularizer: WeightRegularizer = NoRegularizer(),
            dropOut: Double = 0.0): RnnOutputLayer =
    new RnnOutputLayer(
      nIn,
      nOut,
      activation,
      loss,
      weightInit,
      regularizer,
      dropOut
    )

}
