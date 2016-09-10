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

package org.deeplearning4s.iris.example

import java.util

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4s.layers.{Dense, DenseOutput}
import org.deeplearning4s.models.Sequential
import org.deeplearning4s.optimizers.SGD
import org.deeplearning4s.regularizers.l2
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.{DataSet, SplitTestAndTrain}
import org.nd4j.linalg.io.ClassPathResource
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}

object IrisCSVExample extends App {

  val log: Logger = LoggerFactory.getLogger("Iris")

  // input parameters:
  val numLinesToSkip = 0
  val delimiter = ","
  val labelIndex = 4
  val numClasses = 3
  val batchSize = 150
  val learningRate = 0.0015
  val outputNum = 3
  val numEpochs = 300

  // Reading in the Dataset
  val recordReader = new CSVRecordReader(numLinesToSkip, delimiter)
  recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile))
  val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses)

  // Preparing the Dataset for Training
  val next: DataSet = iterator.next()
  next.shuffle()
  val testAndTrain: SplitTestAndTrain = next.splitTestAndTrain(0.75)
  val test_data: DataSet = testAndTrain.getTest
  val training_ : util.List[DataSet] = testAndTrain.getTrain.asList()
  val training_data = new ListDataSetIterator(training_, training_.size)

  log.info("Build model....")
  val model: Sequential = new Sequential(addReshapersAutomatically = false)
  model.add(new Dense(128, nIn = 4, activation = "relu", regularizer = l2(learningRate * 0.005)))
  model.add(new Dense(128, nIn = 4, activation = "relu", regularizer = l2(learningRate * 0.005)))
  model.add(new Dense(128, nIn = 4, activation = "relu", regularizer = l2(learningRate * 0.005)))
  model.add(new DenseOutput(outputNum, activation = "softmax", lossFunction = LossFunction.MCXENT,
    regularizer = l2(learningRate * 0.005)))
  model.compile(optimizer = SGD(learningRate))

  log.info("Train model....")
  model.fit(iter = training_data, nbEpoch = numEpochs, listeners = List(new ScoreIterationListener(5)))

  log.info("Evaluate model....")
  val evaluator = new Evaluation(3)
  val output: INDArray = model.predict(test_data.getFeatureMatrix)
  evaluator.eval(test_data.getLabels, output)
  log.info("Number of Test Examples: " + test_data.getLabels.rows().toString)
  log.info(evaluator.stats())

  log.info("**************** CSV example finished ********************")
}