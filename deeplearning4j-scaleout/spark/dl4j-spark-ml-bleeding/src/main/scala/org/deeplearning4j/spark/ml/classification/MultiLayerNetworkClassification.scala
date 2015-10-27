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
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.classification.{ClassificationModel, Classifier}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.catalyst.expressions.{GenericMutableRow, GenericRowWithSchema}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.layers.{OutputLayer, FeedForwardLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.ml.nn.ParameterAveragingTrainingStrategy
import org.deeplearning4j.spark.ml.param.shared.{HasEpochs, HasMultiLayerConfiguration}
import org.deeplearning4j.spark.ml.util.Identifiable
import org.deeplearning4j.spark.util.conversions._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil

import scala.collection.JavaConversions

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

  override def copy(extra: ParamMap): NeuralNetworkClassification = {
    this.asInstanceOf[PipelineStage].copy(extra).asInstanceOf[NeuralNetworkClassification]
  }

  override protected def train(dataset: DataFrame): NeuralNetworkClassificationModel = {
    val sqlContext = dataset.sqlContext
    val sc = sqlContext.sparkContext
    
    // parameters
    @transient val c = MultiLayerConfiguration.fromJson($(conf))

    // prepare the dataset for classification
    val prepared = dataset.select($(labelCol), $(featuresCol))
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) prepared.persist(StorageLevel.MEMORY_AND_DISK)

    // resolve the number of classes/outcomes
    val numClasses = c.getConf(c.getConfs.size() - 1).getLayer match {
      case layer: OutputLayer => layer.getNOut match {
        case 0 => {
          val numClasses = NominalAttribute.fromStructField(dataset.schema($(labelCol))) match {
            case (attr: NominalAttribute) => attr.getNumValues match {
              case Some(value: Int) => value
              case _ => throw new UnsupportedOperationException("expected numValues on nominal attribute")
            }
            case _ => throw new UnsupportedOperationException(s"column ${$(labelCol)} must be indexed")
          }
          layer.setNOut(numClasses)
          numClasses
        }
        case nOut => nOut
      }
      case _ => throw new UnsupportedOperationException(s"classification requires an output layer")
    }

    // devise a training strategy for the distributed neural network
    val trainingStrategy = new ParameterAveragingTrainingStrategy[Row](c, $(epochs))

    // train
    val networkParams = trainingStrategy.train(
        prepared.rdd, (network:MultiLayerNetwork, rows:Iterator[Row]) => {

          // features & labels
          val (features, labels) = rows.map { row =>
            (
              row.getAs[Vector](1): INDArray,
              FeatureUtil.toOutcomeVector(row.getDouble(0).toInt, numClasses)
              )
          }.toIterable.unzip

          if(features.size >= 1) {
            val featureMatrix = Nd4j.vstack(features.toArray: _*)
            val labelMatrix = Nd4j.vstack(labels.toArray: _*)

            network.fit(featureMatrix, labelMatrix)
          }
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
    throw new NotImplementedError()
  }

  override def transform(dataset: DataFrame): DataFrame = {

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
          val outputMatrix = network.output(featureMatrix, true)

          // prepare column generators for required columns
          val cols = {
            schema.fieldNames flatMap {
              case f if f == $(rawPredictionCol) => Seq(
                (row: Row, i: Int, output: Vector) => output)
              case f if f == $(predictionCol) => Seq(
                (row: Row, i: Int, output: Vector) => raw2prediction(output))
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

  override def copy(extra: ParamMap): NeuralNetworkClassificationModel = {
    copyValues(new NeuralNetworkClassificationModel(uid, numClasses, networkParams), extra)
  }
}