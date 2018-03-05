package org.deeplearning4j.scalnet.examples.dl4j.feedforward

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Two-layer MLP for MNIST using DL4J-style NeuralNet
  * model construction pattern.
  *
  * @author David Kale
  */
object MLPMnistTwoLayerExample extends App with LazyLogging {

  val height: Int = 28
  val width: Int = 28
  val nClasses: Int = 10
  val batchSize: Int = 64
  val hiddenSize = 512
  val seed: Int = 123
  val epochs: Int = 15
  val learningRate: Double = 0.0015
  val decay: Double = 0.005
  val scoreFrequency = 1000

  val mnistTrain: DataSetIterator = new MnistDataSetIterator(batchSize, true, seed)
  val mnistTest: DataSetIterator = new MnistDataSetIterator(batchSize, false, seed)

  logger.info("Build model...")
  val model: NeuralNet = NeuralNet(rngSeed = seed)

  model.add(
    Dense(nIn = height * width,
          nOut = hiddenSize,
          weightInit = WeightInit.XAVIER,
          activation = Activation.RELU,
          regularizer = L2(learningRate * decay))
  )
  model.add(
    Dense(nOut = hiddenSize,
          weightInit = WeightInit.XAVIER,
          activation = Activation.RELU,
          regularizer = L2(learningRate * decay))
  )
  model.add(
    Dense(nClasses,
          weightInit = WeightInit.XAVIER,
          activation = Activation.SOFTMAX,
          regularizer = L2(learningRate * decay))
  )

  model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD)

  logger.info("Train model...")
  model.fit(mnistTrain, epochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model...")
  logger.info(s"Train accuracy = ${model.evaluate(mnistTrain).accuracy}")
  logger.info(s"Test accuracy = ${model.evaluate(mnistTest).accuracy}")
}
