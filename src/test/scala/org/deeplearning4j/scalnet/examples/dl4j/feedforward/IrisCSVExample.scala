/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *  * Dan Dixey - Implemented a working CSV Example using the New Scala Interface for DL4J
 */

package org.deeplearning4j.scalnet.examples.dl4j.feedforward

import java.util

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.scalnet.layers.Dense
import org.deeplearning4j.scalnet.regularizers.L2
import org.deeplearning4j.scalnet.models.NeuralNet
import org.deeplearning4j.scalnet.optimizers.SGD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

object IrisCSVExample extends App {

  private val log: Logger = LoggerFactory.getLogger(IrisCSVExample.getClass)

  private val numLinesToSkip = 0
  private val delimiter = ","
  private val labelIndex = 4
  private val numClasses = 3
  private val batchSize = 150
  private val learningRate = 0.0015
  private val decay = 0.005
  private val numDenseOut = 128
  private val numIn = 4
  private val numOut = 3
  private val numEpochs = 300
  private val scoreFrequency = 5

  log.info("Reading data set....")
  val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
  recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile))
  val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)

  log.info("Prepare data set for training....")
  val next: DataSet = iterator.next()
  next.shuffle()
  val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.75)
  val test_data: DataSet = testAndTrain.getTest
  val training_ : util.List[DataSet] = testAndTrain.getTrain.asList()
  val training_data = new ListDataSetIterator(training_, training_.size)

  log.info("Build model....")
  val model: NeuralNet = NeuralNet()
  model.add(Dense(numDenseOut, nIn = numIn, activation = "relu", regularizer = L2(learningRate * decay)))
  model.add(Dense(numDenseOut, activation = "relu", regularizer = L2(learningRate * decay)))
  model.add(Dense(numDenseOut, activation = "relu", regularizer = L2(learningRate * decay)))
  model.add(Dense(numOut, activation = "softmax", regularizer = L2(learningRate * decay)))
  model.compile(lossFunction = LossFunction.MCXENT, optimizer = SGD(learningRate))

  log.info("Train model....")
  model.fit(iter = training_data, nbEpoch = numEpochs, listeners = List(new ScoreIterationListener(scoreFrequency)))

  log.info("Evaluate model....")
  val evaluator = new Evaluation(3)
  val output: INDArray = model.predict(test_data.getFeatureMatrix)
  evaluator.eval(test_data.getLabels, output)
  log.info("Number of Test Examples: " + test_data.getLabels.rows().toString)
  log.info(evaluator.stats())

  log.info("**************** CSV example finished ********************")
}