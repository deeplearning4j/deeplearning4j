/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.models

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.scalnet.layers.core.{ Layer, Node, Preprocessor }
import org.deeplearning4j.scalnet.logging.Logging
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Class for keras-style simple sequential neural net architectures
  * with one input node and one output node for each node
  * in computational graph.
  *
  * Wraps DL4J MultiLayerNetwork. Enforces keras model construction
  * pattern: preprocessing (reshaping) layers should be explicitly
  * provided by the user, while last layer is treated implicitly as
  * an output layer.
  *
  * @author David Kale
  */
class Sequential(miniBatch: Boolean, biasInit: Double, rngSeed: Long) extends Model with Logging {

  private var _preprocessors: Map[Int, Node] = Map()
  private var _inputShape: List[Int] = List()

  def inputShape: List[Int] = _inputShape
  def getPreprocessors: Map[Int, Node] = _preprocessors

  private val noLayers = inputShape.isEmpty && layers.isEmpty && _preprocessors.isEmpty
  private def emptyShape(layer: Node): Boolean =
    !(_preprocessors.contains(layers.length) || layers.nonEmpty) &&
    layer.inputShape.lengthCompare(1) == 0 && layer.inputShape.head == 0

  def inferInputShape(layer: Node): List[Int] =
    if (_preprocessors.contains(layers.length)) {
      _preprocessors(layers.length).outputShape
    } else layers.lastOption.map(_.outputShape).getOrElse(layer.inputShape)

  def checkShape(layer: Node): Unit =
    if (emptyShape(layer)) {
      throw new IllegalArgumentException("Input layer must have non-empty inputShape")
    } else if (noLayers) {
      _inputShape = layer.inputShape
    }

  def add(layer: Node): Unit = {
    val inferredInput: List[Int] = inferInputShape(layer)
    checkShape(layer)
    val inferredLayer = layer.reshapeInput(inferredInput)
    inferredLayer match {
      case _: Preprocessor => _preprocessors = _preprocessors + (layers.length -> inferredLayer)
      case _               => layers = layers :+ inferredLayer
    }
  }

  override def compile(lossFunction: LossFunction,
                       optimizer: OptimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
                       updater: Updater = Updater.SGD): Unit = {
    val builder = buildModelConfig(optimizer, updater, miniBatch, biasInit, rngSeed)
    buildOutput(lossFunction)

    var listBuilder: NeuralNetConfiguration.ListBuilder = builder.list()
    for ((layer, layerIndex) <- layers.zipWithIndex) {
      logger.info("Layer " + layerIndex + ": " + layer.getClass.getSimpleName)
      logger.info(" size: " + layer.describe())
      listBuilder.layer(layerIndex, layer.asInstanceOf[Layer].compile)
    }
    for ((layerIndex, preprocessor) <- _preprocessors) {
      logger.info("Preprocessor " + layerIndex + ": " + preprocessor.getClass.getSimpleName)
      logger.info(" size: " + preprocessor.describe())
      listBuilder.inputPreProcessor(layerIndex, preprocessor.asInstanceOf[Preprocessor].compile)
    }
    listBuilder = listBuilder.pretrain(false).backprop(true)

    model = new MultiLayerNetwork(listBuilder.build())
    model.init()
  }

}

object Sequential {
  def apply(miniBatch: Boolean = true, biasInit: Double = 0.0, rngSeed: Long = 0): Sequential =
    new Sequential(miniBatch, biasInit, rngSeed)
}
