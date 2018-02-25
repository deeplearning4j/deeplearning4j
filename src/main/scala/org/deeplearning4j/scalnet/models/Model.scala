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

import org.deeplearning4j.nn.conf.{ NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.scalnet.layers.{ Node, OutputLayer }
import org.deeplearning4j.scalnet.optimizers.{ Optimizer, SGD }
import org.deeplearning4j.scalnet.utils.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.{ Nesterovs, Sgd }
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Abstract base class for neural net architectures.
  *
  * @author David Kale
  */
abstract class Model {

  protected var layers: List[Node] = List()
  protected var model: MultiLayerNetwork = _

  protected val defaultEpochs = 10
  protected val defaultOptimizer = SGD(lr = 0.01)

  def getLayers: List[Node] = layers

  /**
    * Build model configuration from optimizer and seed.
    *
    * @param optimizer optimization algorithm to use in model
    * @param seed seed to use
    * @return NeuralNetConfiguration.Builder
    */
  def buildModelConfig(optimizer: Optimizer, seed: Long): NeuralNetConfiguration.Builder = {
    var builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    if (seed != 0) {
      builder = builder.seed(seed)
    }
    optimizer match {
      case sgd: SGD if sgd.nesterov =>
        builder.updater(new Nesterovs(sgd.lr, sgd.momentum))
      case sgd: SGD =>
        builder.updater(new Sgd(sgd.lr))
      case _ =>
        builder
          .optimizationAlgo(optimizer.optimizationAlgorithm)
          .updater(new Sgd(optimizer.asInstanceOf[SGD].lr))
    }
  }

  /**
    * Make last layer of architecture an output layer using
    * the provided loss function.
    *
    * @param lossFunction loss function to use
    */
  def buildOutput(lossFunction: LossFunction): Unit =
    layers.lastOption match {
      case Some(l) if !l.isInstanceOf[OutputLayer] =>
        throw new IllegalArgumentException("Last layer must have Output trait")
      case Some(l) if !l.asInstanceOf[OutputLayer].output.isOutput => {
        val last: OutputLayer = layers.last.asInstanceOf[OutputLayer].toOutputLayer(lossFunction)
        layers = layers.updated(layers.length - 1, last)
      }
      case _ =>
        throw new IllegalArgumentException(
          "Last layer must be an output layer with a valid loss function"
        )
    }

  /**
    * Compile neural net architecture. Call immediately
    * before training.
    *
    * @param lossFunction loss function to use
    * @param optimizer    optimization algorithm to use
    */
  def compile(lossFunction: LossFunction, optimizer: Optimizer): Unit

  /**
    * Fit neural net to data.
    *
    * @param iter      iterator over data set
    * @param nbEpoch   number of epochs to train
    * @param listeners callbacks for monitoring training
    */
  def fit(iter: DataSetIterator, nbEpoch: Int = defaultEpochs, listeners: List[IterationListener]): Unit

  /**
    * Use neural net to make prediction on input x
    *
    * @param x input represented as INDArray
    */
  def predict(x: INDArray): INDArray = model.output(x, false)

  /**
    * Use neural net to make prediction on input x.
    *
    * @param x input represented as DataSet
    */
  def predict(x: DataSet): INDArray = predict(x.getFeatures)

  override def toString: String = model.getLayerWiseConfigurations.toString

  def toJson: String = model.getLayerWiseConfigurations.toJson

  def toYaml: String = model.getLayerWiseConfigurations.toYaml

  def getNetwork: MultiLayerNetwork = model
}
