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

package org.deeplearning4j.scalnet.models

import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.scalnet.layers._
import org.deeplearning4j.scalnet.optimizers.{Optimizer, SGD}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

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
class Sequential(val rngSeed: Long = 0) extends Model {

  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  private var preprocessors: Map[Int, Node] = Map()
  private var _inputShape: List[Int] = List()
  private val seed: Long = rngSeed

  def inputShape: List[Int] = _inputShape

  def inferInputShape(layer: Node): List[Int] = {
    if (preprocessors.contains(layers.length)) {
      preprocessors(layers.length).outputShape
    } else if (layers.nonEmpty) {
      layers.last.outputShape
    } else {
      layer.inputShape
    }
  }

  def checkShape(layer: Node): Unit = {
    if (!(preprocessors.contains(layers.length) || layers.nonEmpty)
      && layer.inputShape.length == 1 && layer.inputShape.head == 0) {
      throw new IllegalArgumentException("Input layer must have non-empty inputShape")
    } else if (inputShape.isEmpty && layers.isEmpty && preprocessors.isEmpty) {
      _inputShape = layer.inputShape
    }
  }


  def add(layer: Node): Unit = {
    val inferredInput: List[Int] = inferInputShape(layer)
    checkShape(layer)
    val inferredLayer = layer.reshapeInput(inferredInput)
    inferredLayer match {
      case _: Preprocessor =>
        preprocessors = preprocessors + (layers.length -> inferredLayer)
      case _ =>
        layers = layers :+ inferredLayer
    }
  }

  override def compile(lossFunction: LossFunction, optimizer: Optimizer = defaultOptimizer): Unit = {
    val builder = buildModelConfig(optimizer, seed)
    buildOutput(lossFunction)

    var listBuilder: NeuralNetConfiguration.ListBuilder = builder.iterations(1).list()
    for ((layer, layerIndex) <- layers.zipWithIndex) {
      log.info("Layer " + layerIndex + ": " + layer.getClass.getSimpleName)
      log.info(" size: " + layer.descibe())
      listBuilder.layer(layerIndex, layer.asInstanceOf[Layer].compile)
    }
    for ((layerIndex, preprocessor) <- preprocessors) {
      log.info("Preprocessor " + layerIndex + ": " + preprocessor.getClass.getSimpleName)
      log.info(" size: " + preprocessor.descibe())
      listBuilder.inputPreProcessor(layerIndex, preprocessor.asInstanceOf[Preprocessor].compile)
    }
    listBuilder = listBuilder.pretrain(false).backprop(true)

    model = new MultiLayerNetwork(listBuilder.build())
    model.init()
  }

  override def fit(iter: DataSetIterator, nbEpoch: Int = defaultEpochs, listeners: List[IterationListener]): Unit = {
    model.setListeners(listeners.asJavaCollection)
    for (_ <- 0 until nbEpoch)
      model.fit(iter)
  }

  override def predict(x: INDArray): INDArray = model.output(x, false)

  override def toString: String = model.getLayerWiseConfigurations.toString

  override def toJson: String = model.getLayerWiseConfigurations.toJson

  override def toYaml: String = model.getLayerWiseConfigurations.toYaml

  def getNetwork: MultiLayerNetwork = model
}
