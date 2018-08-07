/*******************************************************************************
  * Copyright (c) 2015-2018 Skymind, Inc.
  *
  * This program and the accompanying materials are made available under the
  * terms of the Apache License, Version 2.0 which is available at
  * https://www.apache.org/licenses/LICENSE-2.0.
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  * License for the specific language governing permissions and limitations
  * under the License.
  *
  * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.deeplearning4j.scalnet.examples.dl4j.recurrent

import org.deeplearning4j.eval.RegressionEvaluation
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.recurrent.{ Bidirectional, LSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.utils.SequenceGenerator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object SequenceClassification extends App with Logging {

  val timesteps = 10
  val hiddenSize = 32
  val epochs = 500
  val testSize = 100
  val scoreFrequency = 10

  def generateDataset = SequenceGenerator.generate(timesteps)

  logger.info("Build model...")
  val model: NeuralNet = NeuralNet()
  model.add(Bidirectional(LSTM(timesteps, hiddenSize), Bidirectional.ADD))
  model.add(RnnOutputLayer(hiddenSize, timesteps, Activation.SIGMOID))
  model.compile(LossFunction.MEAN_ABSOLUTE_ERROR, updater = Updater.ADAM)

  logger.info("Train model...")
  model.fit(generateDataset, epochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model...")
  val evaluator = new RegressionEvaluation(timesteps)
  for (_ <- 0 until testSize) {
    val testData = generateDataset
    val trueLabels = testData.getLabels
    val predicted = model.predict(testData.getFeatures)
    evaluator.eval(trueLabels, predicted)
  }
  logger.info(s"MAE score: ${evaluator.averageMeanAbsoluteError}")
}
