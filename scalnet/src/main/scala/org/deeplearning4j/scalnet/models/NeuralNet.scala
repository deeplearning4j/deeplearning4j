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
package org.deeplearning4j.scalnet.models

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.scalnet.layers.core.{ Layer, Node }
import org.deeplearning4j.scalnet.logging.Logging
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Simple DL4J-style sequential neural net architecture with one input
  * node and one output node for each node in computational graph.
  *
  * Wraps DL4J MultiLayerNetwork. Enforces DL4J model construction
  * pattern: adds pre-processing layers automatically but requires
  * user to specify output layer explicitly.
  *
  * @author David Kale
  */
class NeuralNet(inputType: Option[InputType], miniBatch: Boolean, biasInit: Double, rngSeed: Long)
    extends Model
    with Logging {

  def add(layer: Node): Unit = layers = layers :+ layer

  override def compile(lossFunction: LossFunction,
                       optimizer: OptimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
                       updater: Updater = Updater.SGD): Unit = {
    val builder = buildModelConfig(optimizer, updater, miniBatch, biasInit, rngSeed)
    buildOutput(lossFunction)

    var listBuilder: NeuralNetConfiguration.ListBuilder = builder.list()
    inputType foreach (i => listBuilder.setInputType(i))

    for ((layer, layerIndex) <- layers.zipWithIndex) {
      logger.info("Layer " + layerIndex + ": " + layer.getClass.getSimpleName)
      listBuilder.layer(layerIndex, layer.asInstanceOf[Layer].compile)
    }

    listBuilder = listBuilder.pretrain(false).backprop(true)
    val conf: MultiLayerConfiguration = listBuilder.build()
    model = new MultiLayerNetwork(conf)
    model.init()
  }

}

object NeuralNet {
  def apply(inputType: InputType = null,
            miniBatch: Boolean = true,
            biasInit: Double = 0.0,
            rngSeed: Long = 0): NeuralNet =
    new NeuralNet(Option(inputType), miniBatch, biasInit, rngSeed)
}
