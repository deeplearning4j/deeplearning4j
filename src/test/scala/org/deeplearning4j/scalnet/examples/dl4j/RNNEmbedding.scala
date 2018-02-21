package org.deeplearning4j.scalnet.examples.dl4j

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.{ EmbeddingLayer, GravesLSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object RNNEmbedding extends App {

  val nClassesIn = 10
  val batchSize = 3
  val nEpochs = 10
  val timeSeriesLength = 8
  val inEmbedding: INDArray = Nd4j.create(batchSize, 1, timeSeriesLength)
  val outLabels: INDArray = Nd4j.create(batchSize, 4, timeSeriesLength)
  val rngSeed = 12345

  val r = Random
  r.setSeed(rngSeed)

  def makeTimeSeries(): DataSetIterator = {
    for (i <- 0 until batchSize; j <- 0 until timeSeriesLength) {
      val classIdx = r.nextInt(nClassesIn)
      inEmbedding.putScalar(Array[Int](i, 0, j), classIdx)
      val labelIdx = r.nextInt(batchSize + 1)
      outLabels.putScalar(Array[Int](i, labelIdx, j), 1.0)
    }

    val dataSet = new DataSet(inEmbedding, outLabels)
    val ids = dataSet.asList()
    val datasetIterator = new ListDataSetIterator(ids, batchSize)
    datasetIterator
  }

  def buildModel: NeuralNet = {
    val model: NeuralNet = NeuralNet(rngSeed = rngSeed)
    model.add(EmbeddingLayer(nClassesIn, 5, Activation.RELU))
    model.add(GravesLSTM(5, 7, Activation.SOFTSIGN))
    model.add(RnnOutputLayer(7, 4, Activation.SOFTMAX, LossFunction.MCXENT))
    model.compile(LossFunction.MCXENT)
    model
  }

  val timeSeries = makeTimeSeries()
  val model = buildModel
  model.fit(timeSeries, nEpochs, List(new ScoreIterationListener(1)))
  val evaluator: Evaluation = new Evaluation(4)
  while (timeSeries.hasNext) {
    val next: DataSet = timeSeries.next()
    val output: INDArray = model.predict(next)
    evaluator.eval(next.getLabels, output)
  }
  evaluator.stats()
}
