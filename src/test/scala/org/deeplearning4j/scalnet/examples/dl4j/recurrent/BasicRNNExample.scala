package org.deeplearning4j.scalnet.examples.dl4j.recurrent

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object BasicRNNExample extends App with LazyLogging {

  // define a sentence to learn.
  // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
  val LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toVector
  val LEARNSTRING_CHARS = LEARNSTRING.distinct

  // Training data
  val input = Nd4j.zeros(1, LEARNSTRING_CHARS.length, LEARNSTRING.length)
  val labels = Nd4j.zeros(1, LEARNSTRING_CHARS.length, LEARNSTRING.length)

  // RNN dimensions
  val HIDDEN_LAYER_WIDTH = 64
  val nEpochs = 100
  val rngSeed = 1234
  val rand = new Random(rngSeed)
  val listeners = List(new ScoreIterationListener(1))

  val trainingData: DataSet = {
    LEARNSTRING.zipWithIndex.foreach {
      case (currentChar, index) =>
        val nextChar = if (index + 1 > LEARNSTRING.indices.max) LEARNSTRING(0) else LEARNSTRING(index + 1)
        input.putScalar(Array[Int](0, LEARNSTRING_CHARS.indexOf(currentChar), index), 1)
        labels.putScalar(Array[Int](0, LEARNSTRING_CHARS.indexOf(nextChar), index), 1)
    }
    new DataSet(input, labels)
  }

  val model: NeuralNet = {
    val model: NeuralNet = NeuralNet(rngSeed = rngSeed, miniBatch = false)
    model.add(GravesLSTM(LEARNSTRING_CHARS.length, HIDDEN_LAYER_WIDTH, Activation.TANH))
    model.add(GravesLSTM(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, Activation.TANH))
    model.add(RnnOutputLayer(HIDDEN_LAYER_WIDTH, LEARNSTRING_CHARS.length, Activation.SOFTMAX))
    model.compile(LossFunction.MCXENT, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT, Updater.RMSPROP)
    model
  }

  val rnn = model.getNetwork
  println(rnn.getLayers.toVector)

  (0 until 1000).foreach { _ =>
    rnn.fit(trainingData)
    rnn.rnnClearPreviousState()
    val init = Nd4j.zeros(LEARNSTRING_CHARS.length)
    init.putScalar(LEARNSTRING_CHARS.indexOf(LEARNSTRING(0)), 1)
    var output = rnn.rnnTimeStep(init)

    LEARNSTRING.foreach { _ =>
      val sampledCharacterIdx = Nd4j.getExecutioner.exec(new IMax(output), 1).getInt(0)
      print(LEARNSTRING_CHARS(sampledCharacterIdx))
      val nextInput = Nd4j.zeros(LEARNSTRING_CHARS.length)
      nextInput.putScalar(sampledCharacterIdx, 1)
      output = rnn.rnnTimeStep(nextInput)
    }
    println("")
  }
}
