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

package org.deeplearning4s.models

import org.deeplearning4s.layers.{Layer, Node, Output, Preprocessor}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4s.layers.convolutional.Convolution
import org.deeplearning4s.optimizers.{Optimizer, SGD}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.collection.JavaConverters._

/**
  * Class for simple sequential neural net architectures
  * with one input node and one output node for each node
  * in computational graph.
  *
  * Wraps DL4J MultiLayerNetwork.
  *
  * @author David Kale
  */
class Sequential(var inputShape: List[Int] = List(),
                 val useNeuralNetConfiguration: Boolean = true,
                 val addReshapersAutomatically: Boolean = true) extends Model {
  private var layers: List[Node] = List()
  private var model: MultiLayerNetwork = _
  private var preprocessors: Map[Int, Node] = Map()
  private var hasConvolutionLayer: Boolean = false

  def add(layer: Node): Unit = {
    if (preprocessors.contains(layers.length))
      layer.inputShape = preprocessors(layers.length).outputShape
    else if (layers.nonEmpty)
      layer.inputShape = layers.last.outputShape
    else if (inputShape.nonEmpty && layers.isEmpty && preprocessors.isEmpty && layer.inputShape != inputShape)
      layer.inputShape = inputShape
    else if (layer.inputShape.isEmpty)
      throw new IllegalArgumentException("Input layer must have non-empty inputShape")
    else if (inputShape.isEmpty && layers.isEmpty && preprocessors.isEmpty)
      inputShape = layer.inputShape

    println("in=" + layer.inputShape + " out=" + layer.outputShape)

    layer match {
      case l: Preprocessor =>
        preprocessors = preprocessors + (layers.length -> layer)
      case l: Convolution =>
        hasConvolutionLayer = true
        layers = layers :+ layer
      case _ =>
        layers = layers :+ layer
    }
  }

  override def compile(lossFunction: LossFunction = null,
                       optimizer: Optimizer = SGD(lr = 0.01)): Unit = {
    var builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    optimizer match {
      case o: SGD =>
        builder = builder.optimizationAlgo(o.optimizationAlgorithm)
          .learningRate(o.lr)
      case _ =>
        print("ERROR!")
        builder = builder.optimizationAlgo(optimizer.optimizationAlgorithm)
          .learningRate(optimizer.asInstanceOf[SGD].lr)
    }
    var listBuilder: NeuralNetConfiguration.ListBuilder = builder.iterations(1).list()

    if (!layers.last.asInstanceOf[Output].isOutput)
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
/*
 * Setting cnnInputSize was necessary to trigger a call to
 * ConvolutionLayerSetup inside of the MultiLayerNetwork constructor.
 * ConvolutionLayerSetup is a convenience function for rectifying
 * input and output sizes between layers by setting nIn and nOut
 * properties (when they have not been set ahead of time), as well as
 * adding input preprocessors between them as needed (e.g., flattening
 * the output of a convolutional layer before feeding it into a dense
 * layer). However, our model construction code handles setting the input
 * and output sizes and forces the user to add explicit reshaping layers,
 * so we no longer require the call to ConvolutionLayerSetup and prefer
 * not to make it so that it the model architecture is transparent to the
 * user.
 */
//    if (layers.head.isInstanceOf[ConvolutionBaseLayer])
//      builder.cnnInputSize(layers.head.inputShape.toArray)
    println("Length: " + layers.length)
    for ((layer, layerIndex) <- layers.zipWithIndex) {
      println(layerIndex + " size: " + layer.inputShape)
    }

    if (addReshapersAutomatically)
      new ConvolutionLayerSetup(listBuilder, inputShape.head, inputShape.tail.head, inputShape.last)

    model = new MultiLayerNetwork(listBuilder.build())
    model.init()
  }

  override def fit(iter: DataSetIterator, nbEpoch: Int = 10, listeners: List[IterationListener]): Unit = {
    model.setListeners(listeners.asJavaCollection)
    for (epoch <- 0 until nbEpoch)
      model.fit(iter)
  }

  override def predict(x: INDArray): INDArray = model.output(x, false)

  def getNetwork(): MultiLayerNetwork = model

}
