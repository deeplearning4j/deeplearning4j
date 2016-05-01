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

package org.deeplearning4j.spark.ml.reconstruction

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.ml.nn._
import org.deeplearning4j.spark.ml.param.shared.{HasEpochs, HasLayerIndex, HasMultiLayerConfiguration, HasReconstructionCol}
import org.deeplearning4j.spark.ml.util.{Identifiable, SchemaUtils}
import org.deeplearning4j.spark.ml.{UnsupervisedLearner, UnsupervisedLearnerParams, UnsupervisedModel}
import org.deeplearning4j.spark.sql.types.VectorUDT
import org.deeplearning4j.spark.util.conversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

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
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    
    val parentSchema = super.validateAndTransformSchema(schema, fitting, featuresDataType)
    SchemaUtils.appendColumn(parentSchema, $(reconstructionCol), VectorUDT())
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
 *
 *  @author Eron Wright
 */
@DeveloperApi
class NeuralNetworkReconstruction(override val uid: String)
  extends UnsupervisedLearner[Vector, NeuralNetworkReconstruction, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {

  def this() = this(Identifiable.randomUID("nnReconstruction"))

  /** @group setParam */
  def setConf(value: String): this.type = set(conf, value)
  def setConf(value: MultiLayerConfiguration): this.type = set(conf, value.toJson)

  /** @group setParam */
  def setEpochs(value: Int): this.type = set(epochs, value)
  setDefault(epochs -> 1)

  /** @group setParam */
  def setLayerIndex(value: Int): this.type = set(layerIndex, value)
  setDefault(layerIndex -> 1)

  /** @group setParam */
  def setReconstructionCol(value: String): this.type = set(reconstructionCol, value)
  setDefault(reconstructionCol -> "reconstruction")

  override protected def learn(dataset: DataFrame): NeuralNetworkReconstructionModel = {
    val sqlContext = dataset.sqlContext
    val sc = sqlContext.sparkContext

    // params
    @transient val c = MultiLayerConfiguration.fromJson($(conf))
    
    // prepare the dataset for reconstruction
    val prepared = dataset.select($(featuresCol))
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) prepared.persist(StorageLevel.MEMORY_AND_DISK)

    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](c, $(epochs))

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {

        // features
        val featureArrays = rows.map(row => row.getAs[Vector](0): INDArray).toArray
        if(featureArrays.length >= 1) {
          val featureMatrix = Nd4j.vstack(featureArrays: _*)

          network.fit(featureMatrix)
        }
    })

    if (handlePersistence) prepared.unpersist()

    new NeuralNetworkReconstructionModel(uid, sc.broadcast(networkParams))
  }
}

/**
 * Neural network-based reconstruction model.
 *
 * @author Eron Wright
 */
@DeveloperApi
class NeuralNetworkReconstructionModel private[ml] (
    override val uid: String,
    val networkParams: Broadcast[INDArray])
  extends UnsupervisedModel[Vector, NeuralNetworkReconstructionModel]
  with NeuralNetworkReconstructionParams {

  override def predict(dataset: DataFrame): DataFrame = {

    val schema = transformSchema(dataset.schema, logging = true)

    val newRdd = dataset.mapPartitions { iterator:Iterator[Row] =>
      val network = new MultiLayerNetwork(MultiLayerConfiguration.fromJson($(conf)))
      network.init()
      network.setParameters(networkParams.value)

      // prepare the input feature matrix, while retaining the rows for later use
      val featuresIndex = schema.fieldIndex(getFeaturesCol)
      val (features, rows) = iterator.map { row =>
        (
          row.getAs[Vector](featuresIndex): INDArray,
          row
          )
      }.toIterable.unzip

      // compute output
      val newRows = rows.size match {
        case 0 => Seq()
        case _ =>
          val featureMatrix = Nd4j.vstack(features.toArray: _*)
          val outputMatrix = network.reconstruct(featureMatrix, ${layerIndex})

          // prepare column generators for required columns
          val cols = {
            schema.fieldNames flatMap {
              case f if f == $(reconstructionCol) => Seq(
                (row: Row, i: Int, output: Vector) => output)
              case _ => Seq.empty
            }
          }

          // transform the input rows, appending required columns
          rows.zipWithIndex.map {
            case (row, i) => {
              val output = outputMatrix.getRow(i): Vector
              Row.fromSeq(row.toSeq ++ cols.map(_(row, i, output)))
            }
          }
      }

      newRows.iterator
    }

    dataset.sqlContext.createDataFrame(newRdd, schema)
  }

  override def copy(extra: ParamMap): NeuralNetworkReconstructionModel = {
    copyValues(new NeuralNetworkReconstructionModel(uid, networkParams), extra)
  }
}