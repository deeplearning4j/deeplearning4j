package org.deeplearning4j.scalnet.examples.dl4j.recurrent

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.{EmbeddingLayer, GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object RNNEmbeddingExample extends App with LazyLogging {

  val nClassesIn = 10
  val batchSize = 3
  val nEpochs = 100
  val timeSeriesLength = 8
  val inEmbedding = Nd4j.create(batchSize, 1, timeSeriesLength)
  val outLabels = Nd4j.create(batchSize, 4, timeSeriesLength)
  val rngSeed = 12345
  val rand = new Random(rngSeed)

  val timeSeries: DataSetIterator = {
    for (i <- 0 until batchSize; j <- 0 until timeSeriesLength) {
      val classIdx = rand.nextInt(nClassesIn)
      inEmbedding.putScalar(Array[Int](i, 0, j), classIdx)
      val labelIdx = rand.nextInt(batchSize + 1)
      outLabels.putScalar(Array[Int](i, labelIdx, j), 1.0)
    }

    val dataset = new DataSet(inEmbedding, outLabels)
    val ids = dataset.asList()
    val datasetIterator = new ListDataSetIterator(ids, batchSize)
    datasetIterator
  }


  // TODO : implement preprocessors
  //    listBuilder.inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
  //    listBuilder.inputPreProcessor(1, new FeedForwardToRnnPreProcessor())

  val model: NeuralNet = {
    val model: NeuralNet = NeuralNet(rngSeed = rngSeed)
    model.add(EmbeddingLayer(nClassesIn, 5))
    model.add(GravesLSTM(5, 7, Activation.SOFTSIGN))
    model.add(RnnOutputLayer(7, 4, Activation.SOFTMAX))
    model.compile(LossFunction.MCXENT)
    model
  }

  model.fit(timeSeries, nEpochs, List(new ScoreIterationListener(1)))
//  logger.info(s"Train accuracy = ${model.accuracy(timeSeries)}")
}
