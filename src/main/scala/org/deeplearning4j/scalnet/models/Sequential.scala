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
import org.deeplearning4j.scalnet.layers.{Layer, Node, Output, Preprocessor}
import org.deeplearning4j.scalnet.layers.convolutional.Convolution
import org.deeplearning4j.scalnet.optimizers.{Optimizer, SGD}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

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
  private var layers: List[Node] = List()
  private var model: MultiLayerNetwork = _
  private var preprocessors: Map[Int, Node] = Map()
  private var _inputShape: List[Int] = List()

  def inputShape: List[Int] = _inputShape

  def add(layer: Node): Unit = {
    if (preprocessors.contains(layers.length))
      layer.inputShape = preprocessors(layers.length).outputShape
    else if (layers.nonEmpty)
      layer.inputShape = layers.last.outputShape
    else if (layer.inputShape.length == 1 && layer.inputShape.head == 0)
      throw new IllegalArgumentException("Input layer must have non-empty inputShape")
    else if (inputShape.isEmpty && layers.isEmpty && preprocessors.isEmpty)
      _inputShape = layer.inputShape

    println("in=" + layer.inputShape + " out=" + layer.outputShape)

    layer match {
      case l: Preprocessor =>
        preprocessors = preprocessors + (layers.length -> layer)
      case _: Convolution =>
        layers = layers :+ layer
      case _ =>
        layers = layers :+ layer
    }
  }

  override def compile(lossFunction: LossFunction = null, optimizer: Optimizer = SGD(lr = 0.01)): Unit = {
    var builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    if (rngSeed != 0)
      builder = builder.seed(rngSeed)
    optimizer match {
      case o: SGD =>
        builder = builder.optimizationAlgo(o.optimizationAlgorithm)
          .learningRate(o.lr)
        if (o.nesterov)
          builder = builder.updater(Updater.NESTEROVS).momentum(o.momentum)
      case _ =>
        builder = builder.optimizationAlgo(optimizer.optimizationAlgorithm)
          .learningRate(optimizer.asInstanceOf[SGD].lr)
    }

    var listBuilder: NeuralNetConfiguration.ListBuilder = builder.iterations(1).list()
    if (!layers.last.isInstanceOf[Output])
      throw new IllegalArgumentException("Last layer must have Output trait")
    else if (!layers.last.asInstanceOf[Output].isOutput)
      layers.last.asInstanceOf[Output].makeOutput(lossFunction)
    for ((layer, layerIndex) <- layers.zipWithIndex) {
      println("Layer " + layerIndex + ": " + layer.getClass.getSimpleName)
      listBuilder.layer(layerIndex, layer.asInstanceOf[Layer].compile)
    }
    for ((layerIndex, preprocessor) <- preprocessors) {
      println("Preprocessor " + layerIndex + ": " + preprocessor.getClass.getSimpleName)
      listBuilder.inputPreProcessor(layerIndex, preprocessor.asInstanceOf[Preprocessor].compile)
    }
    listBuilder = listBuilder.pretrain(false).backprop(true)

    println("Length: " + layers.length)
    for ((layer, layerIndex) <- layers.zipWithIndex) {
      println(layerIndex + " size: " + layer.inputShape)
    }

    model = new MultiLayerNetwork(listBuilder.build())
    model.init()
  }

  override def fit(iter: DataSetIterator, nbEpoch: Int = 10, listeners: List[IterationListener]): Unit = {
    model.setListeners(listeners.asJavaCollection)
    for (epoch <- 0 until nbEpoch)
      model.fit(iter)
  }

  override def predict(x: INDArray): INDArray = model.output(x, false)

  override def toString: String = model.getLayerWiseConfigurations.toString
  override def toJson: String = model.getLayerWiseConfigurations.toJson
  override def toYaml: String = model.getLayerWiseConfigurations.toYaml

  def getNetwork: MultiLayerNetwork = model
}
