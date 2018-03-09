package org.deeplearning4j.scalnet.examples.dl4j.lstm

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.{Bidirectional, Dense, LSTM, RnnOutputLayer}
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.utils.SequenceGenerator
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object SequenceClassification extends App with LazyLogging {

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
