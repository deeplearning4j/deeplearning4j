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
 */

package org.deeplearning4s.models

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

/**
  * Abstract base class for neural net architectures.
  *
  * @author David Kale
  */
abstract class Model {
  /**
   * Compile neural net architecture. Call immediately
   * before training.
   *
   * @param lossFunction        loss function to use
   * @param optimizationAlgo    optimization algorithm to use
   */
  def compile(lossFunction: LossFunction,
              optimizationAlgo: OptimizationAlgorithm,
              suppressConvolutionLayerSetup: Boolean = true): Unit

  /**
    * Fit neural net to data.
    *
    * @param iter         iterator over data set
    * @param nbEpoch      number of epochs to train
    * @param listeners    callbacks for monitoring training
    */
  def fit(iter: DataSetIterator, nbEpoch: Int = 10, listeners: List[IterationListener]): Unit

  /**
    * Use neural net to make prediction on input x
    *
    * @param x    input represented as INDArray
    */
  def predict(x: INDArray): INDArray

  /**
    * Use neural net to make prediction on input x.
    *
    * @param x    input represented as DataSet
    */
  def predict(x: DataSet): INDArray = predict(x.getFeatureMatrix)
}
