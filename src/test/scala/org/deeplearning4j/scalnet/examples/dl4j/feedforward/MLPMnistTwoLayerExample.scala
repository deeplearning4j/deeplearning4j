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

package org.deeplearning4j.scalnet.examples.dl4j.feedforward

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.regularizers.L2
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.optimizers.SGD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

/**
  * Two-layer MLP for MNIST using DL4J-style NeuralNet
  * model construction pattern.
  *
  * @author David Kale
  */
object MLPMnistTwoLayerExample extends App {
  private val log: Logger = LoggerFactory.getLogger(MLPMnistTwoLayerExample.getClass)

  private val numRows: Int = 28
  private val numColumns: Int = 28
  private val outputNum: Int = 10
  private val batchSize: Int = 64
  private val rngSeed: Int = 123
  private val numEpochs: Int = 15
  private val learningRate: Double = 0.0015
  private val decay: Double = 0.005
  private val momentum: Double = 0.98
  private val scoreFrequency = 5

  private val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, rngSeed)
  private val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, rngSeed)

  log.info("Build model....")
  private val model: NeuralNet = NeuralNet(rngSeed = rngSeed)
  model.add(Dense(nOut = 500, nIn = numRows * numColumns, weightInit = WeightInit.XAVIER,
    activation = "relu", regularizer = L2(learningRate * decay)))
  model.add(Dense(nOut = 100, weightInit = WeightInit.XAVIER, activation = "relu",
    regularizer = L2(learningRate * decay)))
  model.add(Dense(outputNum, weightInit = WeightInit.XAVIER, activation = "softmax",
    regularizer = L2(learningRate * decay)))

  model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD,
    optimizer = SGD(learningRate, momentum = momentum, nesterov = true))

  log.info("Train model....")
  model.fit(mnistTrain, nbEpoch = numEpochs, List(new ScoreIterationListener(scoreFrequency)))

  log.info("Evaluate model....")
  val evaluator: Evaluation = new Evaluation(outputNum)
  while (mnistTest.hasNext) {
    val next: DataSet = mnistTest.next()
    val output: INDArray = model.predict(next)
    evaluator.eval(next.getLabels, output)
  }
  log.info(evaluator.stats())
  log.info("****************Example finished********************")
}
