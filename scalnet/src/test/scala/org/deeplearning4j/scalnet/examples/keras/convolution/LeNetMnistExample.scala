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

package org.deeplearning4j.scalnet.examples.keras.convolution

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.core.Dense
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.layers.reshaping.{ Flatten3D, Unflatten3D }
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.Sequential
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Simple LeNet convolutional neural net for MNIST, using
  * keras-style Sequential model construction pattern.
  *
  * @author David Kale
  */
object LeNetMnistExample extends App with Logging {

  val height: Int = 28
  val width: Int = 28
  val channels: Int = 1
  val nClasses: Int = 10

  val batchSize: Int = 64
  val epochs: Int = 10
  val weightDecay: Double = 0.0005
  val seed: Int = 12345
  val scoreFrequency = 100

  val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, seed)
  val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, seed)

  logger.info("Build model...")
  val model: Sequential = Sequential(rngSeed = seed)

  model.add(Unflatten3D(List(height, width, channels), nIn = height * width))
  model.add(Convolution2D(20, List(5, 5), channels, regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(MaxPooling2D(List(2, 2), List(2, 2)))

  model.add(Convolution2D(50, List(5, 5), regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(MaxPooling2D(List(2, 2), List(2, 2)))
  model.add(Flatten3D())

  model.add(Dense(512, regularizer = L2(weightDecay), activation = Activation.RELU))
  model.add(Dense(nClasses, activation = Activation.SOFTMAX))
  model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)

  logger.info("Train model...")
  model.fit(mnistTrain, epochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model...")
  logger.info(s"Train accuracy = ${model.evaluate(mnistTrain).accuracy}")
  logger.info(s"Test accuracy = ${model.evaluate(mnistTest).accuracy}")
}
