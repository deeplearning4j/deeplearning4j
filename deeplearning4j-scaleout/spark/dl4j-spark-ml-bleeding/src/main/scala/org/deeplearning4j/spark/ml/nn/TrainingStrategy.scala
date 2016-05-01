/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.ml.nn

import org.apache.spark.AccumulatorParam
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConversions._

/**
 * An abstract training strategy for a distributed neural network, given partitions of data.
 * @tparam RowType
 *
 * @author Eron Wright
 */
private[spark] abstract class TrainingStrategy[RowType]{
  /**
   * Train a network in some way.
   * @param rdd input RDD
   * @param partitionTrainer a function to train a local MultiLayerNetwork with a partition's data
   * @return final parameters
   */
  def train(
       rdd: RDD[RowType], 
       partitionTrainer: (MultiLayerNetwork, Iterator[RowType]) => Unit): INDArray
}

/**
 * Train a distributed neural network using parameter averaging.   
 * 
 * Training occurs over some number of epochs.  In each epoch:
 * 1. broadcast the parameters to all workers.
 * 2. train a local network on each partition until the network converges (i.e. after some iterations).
 * 3. accumulate the final parameters from each partition and compute an average.
 *
 * @author Eron Wright
 */
private[spark] class ParameterAveragingTrainingStrategy[RowType](
    @transient val conf: MultiLayerConfiguration,
    epochs: Int)
  extends TrainingStrategy[RowType] {
 
   override def train(
       rdd: RDD[RowType], 
       partitionTrainer: (MultiLayerNetwork, Iterator[RowType]) => Unit): INDArray = {
     val sc = rdd.sparkContext
     val confJson = conf.toJson()
     
     def initialParams() = {
       val network = new MultiLayerNetwork(conf)
       network.init()
       network.setListeners(List(new ScoreIterationListener(1)))
       network.params()
     }
   
     var networkParams = initialParams()
     
     for(epoch <- (1 to epochs)) {
       val broadcastedParams = sc.broadcast(networkParams)
       val accumulatedParams = sc.accumulator(Nd4j.zeros(networkParams.shape(): _*))(INDArrayAccumulatorParam)
     
       rdd.foreachPartition { iterator =>
         @transient val conf: MultiLayerConfiguration = MultiLayerConfiguration.fromJson(confJson)

         val network = new MultiLayerNetwork(conf)
         network.init()
         network.setListeners(List(new ScoreIterationListener(1)))
         network.setParams(broadcastedParams.value.dup())
         
         partitionTrainer(network, iterator)
         
         accumulatedParams += network.params()
       }
       
       networkParams = accumulatedParams.value.divi(rdd.partitions.length)
     }
     
     networkParams
   }
}

private[spark] object INDArrayAccumulatorParam extends AccumulatorParam[INDArray] {
  def addInPlace(t1: INDArray, t2: INDArray): INDArray = { t1.addi(t2); t1 }
  def zero(initialValue: INDArray): INDArray = Nd4j.zeros(initialValue.shape(): _*)
}