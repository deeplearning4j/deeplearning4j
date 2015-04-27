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

import org.apache.spark.annotation.{AlphaComponent, DeveloperApi}
import org.apache.spark.ml.param._
import org.apache.spark.ml.nn._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.nn.pretraining._
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

trait NeuralNetworkReconstructionParams extends PretrainerParams 
  with HasMultiLayerConfiguration
  with HasWindowSize
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

@AlphaComponent
class NeuralNetworkReconstruction
extends Pretrainer[Vector, NeuralNetworkReconstruction, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {
  
  override protected def pretrain(dataset: DataFrame, paramMap: ParamMap): NeuralNetworkReconstructionModel = {
    // parameters
    val conf = paramMap(confParam)
    val windowSize = paramMap(windowSizeParam)
    
    // prepare the dataset for reconstruction
    val prepared = dataset.select(paramMap(featuresCol))
   
    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](
        conf, 
        windowSize, 
        conf.getConf(0).getNumIterations())

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {
          
          // features
          val featureArrays = rows.map(row => MLLibUtil.toVector(row.getAs[Vector](0)))
          val featureMatrix = Nd4j.vstack(featureArrays.toArray: _*)
         
          network.pretrain(featureMatrix)
    })
    
    new NeuralNetworkReconstructionModel(this, paramMap, networkParams)
  }
}

@AlphaComponent
class NeuralNetworkReconstructionModel private[ml] (
    override val parent: NeuralNetworkReconstruction,
    override val fittingParamMap: ParamMap,
    val networkParams: INDArray)
  extends PretrainedModel[Vector, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {
  
  val network = new MultiLayerNetwork(paramMap(confParam))
  network.init()
  network.setParameters(networkParams)
  
  override def predict(dataset: DataFrame, paramMap: ParamMap): DataFrame = {

    val layerIndex = paramMap(layerIndexParam)
    
    if (paramMap(reconstructionCol) != "") {
      val pred: Vector => Vector = (features) => {
        reconstruct(features, layerIndex)
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
    val reconstruction: INDArray = network.reconstruct(examples, layerIndex)
    MLLibUtil.toVector(reconstruction)
  }

  override protected def copy(): NeuralNetworkReconstructionModel = {
    val m = new NeuralNetworkReconstructionModel(parent, fittingParamMap, networkParams)
    Params.inheritValues(this.paramMap, this, m)
    m
  }
}