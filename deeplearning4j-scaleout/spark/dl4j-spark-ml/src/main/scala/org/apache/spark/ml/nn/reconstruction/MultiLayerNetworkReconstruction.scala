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

package org.apache.spark.ml.nn.reconstruction

import org.apache.spark.SparkContext
import org.apache.spark.annotation.{AlphaComponent, DeveloperApi}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.nn._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.types.{DataType, DoubleType, FloatType, IntegerType, StructField, StructType, UserDefinedType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.util.MLLibUtil

import org.nd4j.linalg.util.FeatureUtil
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * Parameters for neural network reconstruction.
 */
trait NeuralNetworkReconstructionParams extends UnsupervisedLearnerParams 
  with HasMultiLayerConfiguration
  with HasEpochs
  with HasLayerIndex
  with HasReconstructionCol {
  
  override protected def validateAndTransformSchema(
      schema: StructType,
      paramMap: ParamMap,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    
    val parentSchema = super.validateAndTransformSchema(schema, paramMap, fitting, featuresDataType)
    val map = this.paramMap ++ paramMap
    addOutputColumn(parentSchema, map(reconstructionCol), new VectorUDT)
  }
}

/**
 * Neural network-based learning algorithm for unsupervised reconstruction.
 *
 * This class is an estimator that produces a model.  Accepts a feature vector as input,
 * and produces a feature vector as output.
 * 
 * Noteworthy parameters:
 *  - conf        - the multilayer configuration
 *  - layer index - the neural network layer to use for reconstruction (by default, the input layer)
 *  - epochs      - the number of full passes over the dataset, with convergence on each pass.
 */
@AlphaComponent
class NeuralNetworkReconstruction
extends UnsupervisedLearner[Vector, NeuralNetworkReconstruction, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {
  
  /** @group setParam */
  def setConf(value: String): NeuralNetworkReconstruction = set(conf, value).asInstanceOf[NeuralNetworkReconstruction]
  def setConf(value: MultiLayerConfiguration): NeuralNetworkReconstruction = set(conf, value.toJson()).asInstanceOf[NeuralNetworkReconstruction]

  /** @group setParam */
  def setEpochs(value: Int): NeuralNetworkReconstruction = set(epochs, value).asInstanceOf[NeuralNetworkReconstruction]

  /** @group setParam */
  def setLayerIndex(value: Int): NeuralNetworkReconstruction = set(layerIndex, value).asInstanceOf[NeuralNetworkReconstruction]

  /** @group setParam */
  def setReconstructionCol(value: String): NeuralNetworkReconstruction = set(reconstructionCol, value).asInstanceOf[NeuralNetworkReconstruction]

  override protected def learn(dataset: DataFrame, paramMap: ParamMap): NeuralNetworkReconstructionModel = {
    val sqlContext = dataset.sqlContext
    val sc = sqlContext.sparkContext

    // params
    @transient val c = MultiLayerConfiguration.fromJson(paramMap(conf))
    
    // prepare the dataset for reconstruction
    val prepared = dataset.select(paramMap(featuresCol))
   
    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](
        c,
        paramMap(epochs))

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {
          
          // features
          val featureArrays = rows.map(row => MLLibUtil.toVector(row.getAs[Vector](0)))
          val featureMatrix = Nd4j.vstack(featureArrays.toArray: _*)
         
          network.fit(featureMatrix)
    })
    
    val model = new NeuralNetworkReconstructionModel(this, paramMap, sc.broadcast(networkParams))
    Params.inheritValues(paramMap, this, model)
    model
  }
}

/**
 * Neural network-based reconstruction model.
 */
@AlphaComponent
class NeuralNetworkReconstructionModel private[ml] (
    override val parent: NeuralNetworkReconstruction,
    override val fittingParamMap: ParamMap,
    val networkParams: Broadcast[INDArray])
  extends UnsupervisedModel[Vector, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {

  override def predict(dataset: DataFrame, paramMap: ParamMap): DataFrame = {

    if (paramMap(reconstructionCol) != "") {
      val pred: Vector => Vector = (features) => {
        reconstruct(features, paramMap(layerIndex))
      }
      dataset.withColumn(paramMap(reconstructionCol), 
        callUDF(pred, new VectorUDT, col(paramMap(featuresCol))))
    } else {
      this.logWarning(s"$uid: NeuralNetworkReconstructionModel.transform() was called as NOOP" +
        " since no output columns were set.")
      dataset
    }
  }
  
  protected def reconstruct(features: Vector, layerIndex: Int): Vector = {
    val examples: INDArray = MLLibUtil.toVector(features)
    val reconstruction: INDArray = getNetwork().reconstruct(examples, layerIndex)
    MLLibUtil.toVector(reconstruction)
  }

  override protected def copy(): NeuralNetworkReconstructionModel = {
    val m = new NeuralNetworkReconstructionModel(parent, fittingParamMap, networkParams)
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