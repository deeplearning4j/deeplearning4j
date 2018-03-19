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

package org.deeplearning4j.scalnet.examples.dl4j.recurrent

import org.deeplearning4j.eval.{Evaluation, RegressionEvaluation}
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.recurrent.{Bidirectional, LSTM, RnnOutputLayer}
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.utils.SequenceGenerator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object SequenceClassification extends App with Logging {

  val seed = 1234
  val timesteps = 10
  val hiddenSize = 32
  val epochs = 100
  val scoreFrequency = 10


  def generateDataset = SequenceGenerator.generate(timesteps, 0.7, seed)
  val testData = generateDataset

  logger.info("Build model...")
  val model: NeuralNet = NeuralNet(rngSeed = seed)
  model.add(Bidirectional(LSTM(timesteps, hiddenSize), Bidirectional.ADD))
  model.add(RnnOutputLayer(hiddenSize, timesteps, Activation.SIGMOID))
  model.compile(LossFunction.MEAN_ABSOLUTE_ERROR, updater = Updater.ADAM)

  logger.info("Train model...")
  model.fit(generateDataset, epochs, List(new ScoreIterationListener(scoreFrequency)))

  // TODO: evaluate model
  val trueLabels = testData.getLabels
  val predicted = model.getNetwork.output(testData.getFeatures, false)
  BooleanIndexing.replaceWhere(predicted, 0, Conditions.lessThan(0.5))
  BooleanIndexing.replaceWhere(predicted, 1, Conditions.greaterThan(0.5))
  val evaluator = new RegressionEvaluation(timesteps)
  evaluator.eval(trueLabels, predicted)
  println(evaluator.averageMeanSquaredError())
  println(trueLabels)
  println(predicted)

}
