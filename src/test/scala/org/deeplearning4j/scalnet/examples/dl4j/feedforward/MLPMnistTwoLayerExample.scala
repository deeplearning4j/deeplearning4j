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

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.optimizers.SGD
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Two-layer MLP for MNIST using DL4J-style NeuralNet
  * model construction pattern.
  *
  * @author David Kale
  */
object MLPMnistTwoLayerExample extends App with LazyLogging {

  val numRows: Int = 28
  val numColumns: Int = 28
  val outputNum: Int = 10
  val batchSize: Int = 64
  val rngSeed: Int = 123
  val numEpochs: Int = 15
  val learningRate: Double = 0.0015
  val decay: Double = 0.005
  val scoreFrequency = 100

  val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, rngSeed)
  val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, rngSeed)

  logger.info("Build model....")
  val model: NeuralNet = NeuralNet(rngSeed = rngSeed)

  model.add(
    Dense(nOut = 500,
          nIn = numRows * numColumns,
          weightInit = WeightInit.XAVIER,
          activation = Activation.RELU,
          regularizer = L2(learningRate * decay))
  )
  model.add(
    Dense(nOut = 100,
          weightInit = WeightInit.XAVIER,
          activation = Activation.RELU,
          regularizer = L2(learningRate * decay))
  )
  model.add(
    Dense(outputNum,
          weightInit = WeightInit.XAVIER,
          activation = Activation.SOFTMAX,
          regularizer = L2(learningRate * decay))
  )

  model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD)

  logger.info("Train model....")
  model.fit(mnistTrain, nbEpoch = numEpochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model....")
  val accuracy = model.evaluate(mnistTest)

  logger.info(s"Model accuracy: $accuracy")
}
