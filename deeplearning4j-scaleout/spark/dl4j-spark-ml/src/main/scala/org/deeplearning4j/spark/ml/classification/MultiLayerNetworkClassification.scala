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

package org.deeplearning4j.spark.ml.classification

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.ml.nn.ParameterAveragingTrainingStrategy
import org.deeplearning4j.spark.ml.param.shared.{HasEpochs, HasMultiLayerConfiguration}
import org.deeplearning4j.spark.ml.util.Identifiable
import org.deeplearning4j.spark.util.MLLibUtil
import org.deeplearning4j.spark.util.conversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil

/*
 * Parameters for neural network classification.
 */
trait NeuralNetworkClassificationParams extends Params
  with HasMultiLayerConfiguration 
  with HasEpochs {
}

/**
 * Neural network-based learning algorithm for supervised classification.
 * 
 * This class is an estimator that produces a model.  Accepts a feature vector and 
 * a multiclass numeric label as input, and produces a probability vector and a predicted label as output.
 * 
 * Noteworthy parameters:
 *  - conf        - the multilayer configuration
 *  - epochs      - the number of full passes over the dataset, with convergence on each pass.
 *
 *  @author Eron Wright
 */
@DeveloperApi
class NeuralNetworkClassification(override val uid: String)
  extends Classifier[Vector, NeuralNetworkClassification, NeuralNetworkClassificationModel]
  with NeuralNetworkClassificationParams {

  def this() = this(Identifiable.randomUID("nnClassification"))

  /** @group setParam */
  def setConf(value: String): this.type = set(conf, value)
  def setConf(value: MultiLayerConfiguration): this.type = set(conf, value.toJson)

  /** @group setParam */
  def setEpochs(value: Int): this.type = set(epochs, value)
  setDefault(epochs -> 1)

  override protected def train(dataset: DataFrame): NeuralNetworkClassificationModel = {
    val sqlContext = dataset.sqlContext
    val sc = sqlContext.sparkContext
    
    // parameters
    @transient val c = MultiLayerConfiguration.fromJson($(conf))

    // prepare the dataset for classification
    val prepared = dataset.select($(labelCol), $(featuresCol))
    val numClasses = c.getConf(c.getConfs.size() - 1).getnOut // TODO - use ML column metadata 'numValues'
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) prepared.persist(StorageLevel.MEMORY_AND_DISK)

    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](c, $(epochs))

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {

          // features & labels
          val (features, labels) = rows.map { row =>
            (
              MLLibUtil.fromVector(row.getAs[Vector](1)),
              FeatureUtil.toOutcomeVector(row.getDouble(0).toInt, numClasses)
            )
          }.toIterable.unzip
          
          val featureMatrix = Nd4j.vstack(features.toArray: _*)
          val labelMatrix = Nd4j.vstack(labels.toArray: _*)
          
          network.fit(featureMatrix, labelMatrix)
    })

    if (handlePersistence) prepared.unpersist()

    new NeuralNetworkClassificationModel(uid, numClasses, sc.broadcast(networkParams)).setParent(this)
  }
}

/**
 * Neural network-based classification model.
 *
 * @author Eron Wright
 */
@DeveloperApi
class NeuralNetworkClassificationModel private[ml] (
    override val uid: String,
    override val numClasses: Int,
    val networkParams: Broadcast[INDArray])
  extends ClassificationModel[Vector, NeuralNetworkClassificationModel]
  with NeuralNetworkClassificationParams {

  override protected def predictRaw(features: Vector): Vector = {
    val examples: INDArray = MLLibUtil.fromVector(features)

    // produces a probability distribution
    MLLibUtil.toVector(network().output(examples))
  }

  override def copy(extra: ParamMap): NeuralNetworkClassificationModel = {
    copyValues(new NeuralNetworkClassificationModel(uid, numClasses, networkParams), extra)
  }

  @transient
  private var networkHolder: ThreadLocal[MultiLayerNetwork] = null

  private def network(): MultiLayerNetwork = {

    if(networkHolder == null) {
      networkHolder = new ThreadLocal[MultiLayerNetwork] {
        override def initialValue(): MultiLayerNetwork = {
          val network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson($(conf)))
          network.init()
          network.setParameters(networkParams.value)
          network
        }
      }
    }
    networkHolder.get()
  }
}