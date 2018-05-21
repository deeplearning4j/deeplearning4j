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

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ OutputLayer => JOutputLayer }
import org.deeplearning4j.nn.conf.{ NeuralNetConfiguration, Updater }
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.scalnet.layers.core.{ Node, OutputLayer }
import org.deeplearning4j.scalnet.logging.Logging
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.collection.JavaConverters._

/**
  * Abstract base class for neural net architectures.
  *
  * @author David Kale
  */
trait Model extends Logging {

  protected var layers: List[Node] = List()
  protected var model: MultiLayerNetwork = _

  def getLayers: List[Node] = layers

  /**
    * Build model configuration from optimizer and seed.
    *
    * @param optimizer optimization algorithm to use in model
    * @param seed      seed to use
    * @return NeuralNetConfiguration.Builder
    */
  def buildModelConfig(optimizer: OptimizationAlgorithm,
                       updater: Updater,
                       miniBatch: Boolean,
                       biasInit: Double,
                       seed: Long): NeuralNetConfiguration.Builder = {
    var builder: NeuralNetConfiguration.Builder = new NeuralNetConfiguration.Builder()
    if (seed != 0) {
      builder = builder.seed(seed)
    }
    builder
      .optimizationAlgo(optimizer)
      .updater(updater.getIUpdaterWithDefaultConfig)
      .miniBatch(miniBatch)
      .biasInit(biasInit)
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
      case Some(l) if !l.asInstanceOf[OutputLayer].output.isOutput =>
        val last: OutputLayer = layers.last.asInstanceOf[OutputLayer].toOutputLayer(lossFunction)
        layers = layers.updated(layers.length - 1, last)
      case _ =>
        throw new IllegalArgumentException("Last layer must be an output layer with a valid loss function")
    }

  /**
    * Compile neural net architecture. Call immediately
    * before training.
    *
    * @param lossFunction loss function to use
    * @param optimizer    optimization algorithm to use
    */
  def compile(lossFunction: LossFunction, optimizer: OptimizationAlgorithm, updater: Updater): Unit

  /**
    * Fit neural net to data.
    *
    * @param iter      iterator over data set
    * @param nbEpoch   number of epochs to train
    * @param listeners callbacks for monitoring training
    */
  def fit(iter: DataSetIterator, nbEpoch: Int, listeners: List[TrainingListener]): Unit = {
    model.setListeners(listeners.asJavaCollection)
    for (epoch <- 0 until nbEpoch) {
      logger.info("Epoch " + epoch)
      model.fit(iter)
    }
  }

  /**
    * Fit neural net to data.
    * @param dataset    data set
    * @param nbEpoch    number of epochs to train
    * @param listeners  callbacks for monitoring training
    */
  def fit(dataset: DataSet, nbEpoch: Int, listeners: List[TrainingListener]): Unit = {
    model.setListeners(listeners.asJavaCollection)
    for (epoch <- 0 until nbEpoch) {
      logger.info("Epoch " + epoch)
      model.fit(dataset)
    }
  }

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

  /**
    * Evaluate model against an iterator over data set
    *
    * @param iter iterator over data set
    * @return Evaluation instance
    */
  def evaluate(iter: DataSetIterator): Evaluation = {
    val evaluator = new Evaluation(layers.last.outputShape.last)
    iter.reset()
    for (dataset <- iter.asScala) {
      val output = predict(dataset)
      evaluator.eval(dataset.getLabels, output)
    }
    evaluator
  }

  /**
    * Evaluate model against an iterator over data set
    *
    * @param iter iterator over data set
    * @param numClasses output size
    * @return Evaluation instance
    */
  def evaluate(iter: DataSetIterator, numClasses: Int): Evaluation = {
    val evaluator = new Evaluation(numClasses)
    iter.reset()
    for (dataset <- iter.asScala) {
      val output = predict(dataset)
      evaluator.eval(dataset.getLabels, output)
    }
    evaluator
  }

  /**
    * Evaluate model against an iterator over data set
    *
    * @param dataset    data set
    * @return Evaluation instance
    */
  def evaluate(dataset: DataSet): Evaluation = {
    val evaluator = new Evaluation(layers.last.outputShape.last)
    val output = predict(dataset)
    evaluator.eval(dataset.getLabels, output)
    evaluator
  }

  /**
    * Evaluate model against an iterator over data set
    *
    * @param dataset    data set
    * @param numClasses output size
    * @return Evaluation instance
    */
  def evaluate(dataset: DataSet, numClasses: Int): Evaluation = {
    val evaluator = new Evaluation(numClasses)
    val output = predict(dataset)
    evaluator.eval(dataset.getLabels, output)
    evaluator
  }

  override def toString: String = model.getLayerWiseConfigurations.toString

  def toJson: String = model.getLayerWiseConfigurations.toJson

  def toYaml: String = model.getLayerWiseConfigurations.toYaml

  def getNetwork: MultiLayerNetwork = model
}
