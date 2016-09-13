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

package org.deeplearning4s.examples.keras.feedforward

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4s.layers.{Dense, DenseOutput}
import org.deeplearning4s.models.Sequential
import org.deeplearning4s.optimizers.SGD
import org.deeplearning4s.regularizers.l2
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

/**
  * Two-layer MLP for MNIST using keras-style Sequential
  * model construction pattern.
  *
  * @author David Kale
  */
object MLPMnistTwoLayerExample extends App {
  private val log: Logger = LoggerFactory.getLogger("yay")

  private val numRows: Int = 28
  private val numColumns: Int = 28
  private val outputNum: Int = 10
  private val batchSize: Int = 64
  private val rngSeed: Int = 123
  private val numEpochs: Int = 1
  private val learningRate: Double = 0.0015
  private val momentum: Double = 0.98

  private val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, rngSeed)
  private val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, rngSeed)

  log.info("Build model....")
  private val model: Sequential = new Sequential(rngSeed = rngSeed)
  model.add(new Dense(500, nIn = numRows*numColumns, weightInit = WeightInit.XAVIER, activation = "relu",
                      regularizer = l2(learningRate * 0.005)))
  model.add(new Dense(100, weightInit = WeightInit.XAVIER, activation = "relu", regularizer = l2(learningRate * 0.005)))
  model.add(new DenseOutput(outputNum, weightInit = WeightInit.XAVIER, activation = "softmax",
                            lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD, regularizer = l2(learningRate * 0.005)))
  model.compile(optimizer = SGD(learningRate, momentum = momentum, nesterov = true))

  log.info("Train model....")
  model.fit(mnistTrain, nbEpoch = numEpochs, List(new ScoreIterationListener(5)))

  log.info("Evaluate model....")
  val evaluator: Evaluation = new Evaluation(outputNum)
  while(mnistTest.hasNext){
    val next: DataSet = mnistTest.next()
    val output: INDArray = model.predict(next)
    evaluator.eval(next.getLabels, output)
  }
  log.info(evaluator.stats())
  log.info("****************Example finished********************")
  println(evaluator.stats())
}
