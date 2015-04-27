package org.apache.spark.ml.nn

import scala.collection.JavaConversions._

import org.apache.spark.AccumulatorParam
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j;
import org.apache.spark.rdd.RDD

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
 * Training occurs over some number of iterations (totalIterations).  For performance,
 * network parameters are averaged not after each iteration but after some number of iterations (windowSize).
 */
private[spark] class ParameterAveragingTrainingStrategy[RowType](
    val conf: MultiLayerConfiguration,
    windowSize: Int,
    totalIterations: Int)
  extends TrainingStrategy[RowType] {
 
   override def train(
       rdd: RDD[RowType], 
       partitionTrainer: (MultiLayerNetwork, Iterator[RowType]) => Unit): INDArray = {
     val sc = rdd.sparkContext
     
     def initialParams() = {
       val network = new MultiLayerNetwork(conf);
       network.init();
       network.params(); 
     }
   
     var networkParams = initialParams()
     
     for(iterations <- (1 to totalIterations by windowSize).map(_=>windowSize)) {
       // slide a window over the range corresponding to the total iterations
       // within each window, train the network for some number of iterations
       
       val broadcastedParams = sc.broadcast(networkParams)
       val accumulatedParams = sc.accumulator(Nd4j.zeros(networkParams.shape()))(INDArrayAccumulatorParam)
     
       rdd.foreachPartition { iterator =>
         
         for(innerConf <- conf.getConfs()) 
           innerConf.setNumIterations(iterations)
         val network = new MultiLayerNetwork(conf);
         network.init();
         network.setParams(broadcastedParams.value)
         
         partitionTrainer(network, iterator)
         
         accumulatedParams += network.params()
       }
       
       networkParams = accumulatedParams.value.divi(rdd.partitions.length);
     }
     
     networkParams
   }
}

private[spark] object INDArrayAccumulatorParam extends AccumulatorParam[INDArray] {
  def addInPlace(t1: INDArray, t2: INDArray): INDArray = { t1.addi(t2); t1 }
  def zero(initialValue: INDArray): INDArray = Nd4j.zeros(initialValue.shape())
}