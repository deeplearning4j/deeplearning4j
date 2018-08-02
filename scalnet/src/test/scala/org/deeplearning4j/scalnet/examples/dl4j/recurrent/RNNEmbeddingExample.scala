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

import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.embeddings.EmbeddingLayer
import org.deeplearning4j.scalnet.layers.recurrent.{ GravesLSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object RNNEmbeddingExample extends App {

  val nClassesIn = 10
  val batchSize = 3
  val timeSeriesLength = 8
  val inEmbedding = Nd4j.create(batchSize, 1, timeSeriesLength)
  val outLabels = Nd4j.create(batchSize, 4, timeSeriesLength)
  val seed = 12345
  val rand = new Random(seed)

  val timeSeries: DataSet = {
    for (i <- 0 until batchSize; j <- 0 until timeSeriesLength) {
      val classIdx = rand.nextInt(nClassesIn)
      inEmbedding.putScalar(Array[Int](i, 0, j), classIdx)
      val labelIdx = rand.nextInt(batchSize + 1)
      outLabels.putScalar(Array[Int](i, labelIdx, j), 1.0)
    }
    new DataSet(inEmbedding, outLabels)
  }

  val model: NeuralNet = {
    val model: NeuralNet = NeuralNet(inputType = InputType.recurrent(3, 8), rngSeed = seed)
    model.add(EmbeddingLayer(nClassesIn, 5))
    model.add(GravesLSTM(5, 7, Activation.SOFTSIGN))
    model.add(RnnOutputLayer(7, 4, Activation.SOFTMAX))
    model.compile(LossFunction.MCXENT)
    model
  }

  model.fit(timeSeries, 1, List(new ScoreIterationListener(1)))
}
