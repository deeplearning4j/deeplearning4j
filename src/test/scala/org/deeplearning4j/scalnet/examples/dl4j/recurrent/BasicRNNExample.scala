package org.deeplearning4j.scalnet.examples.dl4j.recurrent

import com.typesafe.scalalogging.LazyLogging
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.examples.dl4j.recurrent.RNNEmbeddingExample.{ logger, model, timeSeries }
import org.deeplearning4j.scalnet.layers.{ GravesLSTM, RnnOutputLayer }
import org.deeplearning4j.scalnet.models.NeuralNet
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.util.Random

object BasicRNNExample extends App with LazyLogging {

  // define a sentence to learn.
  // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
  val LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray
  val LEARNSTRING_CHARS = LEARNSTRING.distinct.sorted

  // Training data
  val inputs = Nd4j.zeros(1, LEARNSTRING_CHARS.length, LEARNSTRING.length)
  val labels = Nd4j.zeros(1, LEARNSTRING_CHARS.length, LEARNSTRING.length)

  // RNN dimensions
  val HIDDEN_LAYER_WIDTH = 64
  val nEpochs = 100
  val rngSeed = 1234
  val rand = new Random(rngSeed)

  val trainingData: DataSet = {
    LEARNSTRING.zipWithIndex.foreach {
      case (currentChar, index) =>
        val nextChar = if (index + 1 > LEARNSTRING.indices.max) LEARNSTRING(0) else LEARNSTRING(index + 1)
        inputs.putScalar(Array(0, LEARNSTRING_CHARS.indexOf(currentChar), index), 1)
        labels.putScalar(Array(0, LEARNSTRING_CHARS.indexOf(nextChar), index), 1)
    }
    new DataSet(inputs, labels)
  }

  val model: NeuralNet = {
    val model: NeuralNet = NeuralNet(rngSeed = rngSeed)
    model.add(GravesLSTM(LEARNSTRING_CHARS.length, HIDDEN_LAYER_WIDTH, Activation.TANH))
    model.add(GravesLSTM(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH, Activation.TANH))
    model.add(RnnOutputLayer(HIDDEN_LAYER_WIDTH, LEARNSTRING_CHARS.length, Activation.SOFTMAX))
    model.compile(LossFunction.MCXENT)
    model
  }

  model.fit(trainingData, nEpochs, List(new ScoreIterationListener(1)))
  val evaluator: Evaluation = new Evaluation(4)
  val output = model.predict(inputs).reshape(1, 21, 65)
  evaluator.eval(labels, output)
  logger.info(evaluator.stats())

}
