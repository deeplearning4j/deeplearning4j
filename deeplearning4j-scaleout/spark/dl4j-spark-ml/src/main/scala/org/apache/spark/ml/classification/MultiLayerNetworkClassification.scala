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

package org.apache.spark.ml.classification

import scala.collection.JavaConversions._

import org.apache.spark.SparkContext
import org.apache.spark.annotation.AlphaComponent
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.nn._
import org.apache.spark.mllib.linalg.{VectorUDT, BLAS, Vector, Vectors}
import org.apache.spark.sql.types.{DataType, DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.util.MLLibUtil

import org.nd4j.linalg.util.FeatureUtil
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

/*
 * Parameters for neural network classification.
 */
trait NeuralNetworkClassificationParams extends ProbabilisticClassifierParams 
  with HasMultiLayerConfiguration 
  with HasWindowSize {
}

/**
 * Neural network-based learning algorithm for supervised classification.
 * 
 * This class is an estimator that produces a model.  Accepts a feature vector and 
 * a multiclass numeric label as input, and produces a probability vector and a predicted label as output.
 * 
 * Noteworthy parameters:
 *  - conf        - the multilayer configuration
 *  - windowSize  - the number of training iterations to perform independently on each partition,
 *                  before resynchronizing the parameters across the cluster.
 */
@AlphaComponent
class NeuralNetworkClassification
  extends ProbabilisticClassifier[Vector, NeuralNetworkClassification, NeuralNetworkClassificationModel]
  with NeuralNetworkClassificationParams {
  
  /** @group setParam */
  def setConf(value: String): NeuralNetworkClassification = set(conf, value).asInstanceOf[NeuralNetworkClassification]
  def setConf(value: MultiLayerConfiguration): NeuralNetworkClassification = set(conf, value.toJson()).asInstanceOf[NeuralNetworkClassification]

  /** @group setParam */
  def setWindowSize(value: Int): NeuralNetworkClassification = set(windowSize, value).asInstanceOf[NeuralNetworkClassification]

  override protected def train(dataset: DataFrame, paramMap: ParamMap): NeuralNetworkClassificationModel = {
    val sqlContext = dataset.sqlContext
    val sc = sqlContext.sparkContext
    
    // parameters
    @transient val c = MultiLayerConfiguration.fromJson(paramMap(conf));

    // prepare the dataset for classification
    val prepared = dataset.select(paramMap(labelCol), paramMap(featuresCol))
    val numClasses = c.getConf(c.getConfs().size() - 1).getnOut() // TODO - use ML column metadata 'numValues'
    
    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](
        c,
        paramMap(windowSize),
        c.getConf(0).getNumIterations())

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {

          // features & labels
          val (features, labels) = rows.map { row =>
            (MLLibUtil.toVector(row.getAs[Vector](1)), FeatureUtil.toOutcomeVector(row.getDouble(0).toInt, numClasses))
          }.toIterable.unzip
          
          val featureMatrix = Nd4j.vstack(features.toArray: _*)
          val labelMatrix = Nd4j.vstack(labels.toArray: _*)
          
          network.fit(featureMatrix, labelMatrix)
    })
    
    val model = new NeuralNetworkClassificationModel(this, paramMap, numClasses, sc.broadcast(networkParams))
    Params.inheritValues(paramMap, this, model)
    model
  }
}

/**
 * Neural network-based classification model.
 */
@AlphaComponent
class NeuralNetworkClassificationModel private[ml] (
    override val parent: NeuralNetworkClassification,
    override val fittingParamMap: ParamMap,
    override val numClasses: Int,
    val networkParams: Broadcast[INDArray])
  extends ProbabilisticClassificationModel[Vector, NeuralNetworkClassificationModel]
  with NeuralNetworkClassificationParams {
  
  override protected def predict(features: Vector): Double = {
    val examples: INDArray = MLLibUtil.toVector(features)
    
    val predictions = getNetwork().predict(examples)
    predictions(0)
  }
  
  override protected def predictRaw(features: Vector): Vector = {
    val examples: INDArray = MLLibUtil.toVector(features)
    
    val raw = getNetwork().feedForward(examples)
    MLLibUtil.toVector(raw.get(0))
  }
  
  override protected def predictProbabilities(features: Vector): Vector = {
    
    val examples: INDArray = MLLibUtil.toVector(features)
    
    val probabilities: INDArray = getNetwork().labelProbabilities(examples)
    MLLibUtil.toVector(probabilities)
  }
  
  override protected def copy(): NeuralNetworkClassificationModel = {
    val m = new NeuralNetworkClassificationModel(parent, fittingParamMap, numClasses, networkParams)
    Params.inheritValues(this.paramMap, this, m)
    m
  }

  @transient
  private var networkHolder: ThreadLocal[MultiLayerNetwork] = null

  private def getNetwork(): MultiLayerNetwork = {

    if(networkHolder == null) {
      networkHolder = new ThreadLocal[MultiLayerNetwork] {
        override def initialValue(): MultiLayerNetwork = {
          val network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson(paramMap(conf)))
          network.init()
          network.setParameters(networkParams.value)
          network
        }
      }
    }
    networkHolder.get()
  }
}