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

package org.deeplearning4j.scalnet.examples.dl4j.lstm

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.recurrent.{ LSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.utils.SequenceGenerator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object SequenceClassification extends App with Logging {

  val seed = 1234
  val rows = 10
  val timesteps = 10
  val epochs = 100
  val scoreFrequency = 10

  val dataset = SequenceGenerator.generate(rows, timesteps, 0.6, seed).splitTestAndTrain(0.75)
  val trainingData = new ListDataSetIterator(dataset.getTrain.batchBy(1))
  val testData = dataset.getTest

  logger.info("Build model...")
  val model: NeuralNet = NeuralNet(rngSeed = seed)
  model.add(LSTM(timesteps, 32))
  model.add(RnnOutputLayer(32, 10, Activation.SIGMOID))
  model.compile(LossFunction.MCXENT, updater = Updater.ADAM)

  logger.info("Train model...")
  model.fit(trainingData, epochs, List(new ScoreIterationListener(scoreFrequency)))

  // TODO: evaluate model
}
