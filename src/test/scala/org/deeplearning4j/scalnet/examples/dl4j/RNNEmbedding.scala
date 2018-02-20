package org.deeplearning4j.scalnet.examples.dl4j

import org.deeplearning4j.scalnet.layers.{ EmbeddingLayer, GravesLSTM }
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.util.Random

object RNNEmbedding {

  val nClassesIn            = 10
  val batchSize             = 3
  val timeSeriesLength      = 8
  val inEmbedding: INDArray = Nd4j.create(batchSize, 1, timeSeriesLength)
  val outLabels: INDArray   = Nd4j.create(batchSize, 4, timeSeriesLength)
  val rngSeed               = 12345

  val r = Random
  r.setSeed(rngSeed)
  var i = 0
  while (i < batchSize) {
    var j = 0
    while (j < timeSeriesLength) {
      val classIdx = r.nextInt(nClassesIn)
      inEmbedding.putScalar(Array[Int](i, 0, j), classIdx)
      val labelIdx = r.nextInt(4)
      outLabels.putScalar(Array[Int](i, labelIdx, j), 1.0)
      j += 1
    }
    i += 1
  }

  private val model: NeuralNet = NeuralNet(rngSeed = rngSeed)
  model.add(EmbeddingLayer(nIn = nClassesIn, nOut = 5, activationFn = Activation.RELU))
  model.add(GravesLSTM(nIn = 5, nOut = 7, activationFn = Activation.SOFTSIGN))
}
