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

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.scalnet.layers.recurrent.{ GravesLSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.logging.Logging
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object BasicRNNExample extends App with Logging {

  // define a sentence to learn.
  // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
  val learningString = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toVector
  val learningChars = learningString.distinct
  val hiddenSize = 64
  val epochs = 200
  val seed = 1234
  val rand = new Random(seed)

  val input = Nd4j.zeros(1, learningChars.length, learningString.length)
  val labels = Nd4j.zeros(1, learningChars.length, learningString.length)

  val trainingData: DataSet = {
    learningString.zipWithIndex.foreach {
      case (currentChar, index) =>
        val nextChar = if (index + 1 > learningString.indices.max) learningString(0) else learningString(index + 1)
        input.putScalar(Array[Int](0, learningChars.indexOf(currentChar), index), 1)
        labels.putScalar(Array[Int](0, learningChars.indexOf(nextChar), index), 1)
    }
    new DataSet(input, labels)
  }

  logger.info("Build model...")
  val model: NeuralNet = {
    val model: NeuralNet = NeuralNet(rngSeed = seed, miniBatch = false)
    model.add(GravesLSTM(learningChars.length, hiddenSize, Activation.TANH))
    model.add(GravesLSTM(hiddenSize, hiddenSize, Activation.TANH))
    model.add(RnnOutputLayer(hiddenSize, learningChars.length, Activation.SOFTMAX))
    model.compile(LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, Updater.RMSPROP)
    model
  }

  val rnn = model.getNetwork

  (0 until epochs).foreach { e =>
    rnn.fit(trainingData)
    rnn.rnnClearPreviousState()
    val init = Nd4j.zeros(learningChars.length)
    init.putScalar(learningChars.indexOf(learningString(0)), 1)
    var output = rnn.rnnTimeStep(init)

    val predicted: Vector[Char] = learningString.map { _ =>
      val sampledCharacterIdx = Nd4j.getExecutioner.exec(new IMax(output), 1).getInt(0)
      val nextInput = Nd4j.zeros(learningChars.length)
      nextInput.putScalar(sampledCharacterIdx, 1)
      output = rnn.rnnTimeStep(nextInput)
      learningChars(sampledCharacterIdx)
    }
    logger.info(s"Epoch $e - ${predicted.mkString}")
  }
}
