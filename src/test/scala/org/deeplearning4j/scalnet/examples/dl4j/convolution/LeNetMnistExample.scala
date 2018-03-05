package org.deeplearning4j.scalnet.examples.dl4j.convolution

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.layers.convolutional.Convolution2D
import org.deeplearning4j.scalnet.layers.pooling.MaxPooling2D
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.regularizers.L2
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Simple LeNet convolutional neural net for MNIST, using
  * DL4J-style NeuralNet model construction pattern.
  *
  * @author David Kale
  */
object LeNetMnistExample extends App with LazyLogging {

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
  val model: NeuralNet = NeuralNet(
    inputType = InputType.convolutionalFlat(height, width, channels),
    rngSeed = seed
  )
  model.add(
    Convolution2D(
      nFilter = 20,
      nChannels = channels,
      kernelSize = List(5, 5),
      stride = List(1, 1),
      weightInit = WeightInit.XAVIER,
      regularizer = L2(weightDecay),
      activation = Activation.RELU
    )
  )
  model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
  model.add(
    Convolution2D(
      nFilter = 50,
      kernelSize = List(5, 5),
      stride = List(1, 1),
      weightInit = WeightInit.XAVIER,
      regularizer = L2(weightDecay),
      activation = Activation.RELU
    )
  )
  model.add(MaxPooling2D(kernelSize = List(2, 2), stride = List(2, 2)))
  model.add(
    Dense(nOut = 512, weightInit = WeightInit.XAVIER, activation = Activation.RELU, regularizer = L2(weightDecay))
  )
  model.add(Dense(nOut = nClasses, weightInit = WeightInit.XAVIER, activation = Activation.SOFTMAX))
  model.compile(lossFunction = LossFunction.NEGATIVELOGLIKELIHOOD)

  logger.info("Train model...")
  model.fit(mnistTrain, epochs, List(new ScoreIterationListener(scoreFrequency)))

  logger.info("Evaluate model...")
  logger.info(s"Train accuracy = ${model.evaluate(mnistTrain).accuracy}")
  logger.info(s"Test accuracy = ${model.evaluate(mnistTest).accuracy}")
}
