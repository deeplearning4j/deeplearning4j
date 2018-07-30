/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.spark.ml.impl

import java.util

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, RowFactory}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.spark.api.stats.SparkTrainingStats
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer, SparkDl4jUtil}
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil

import scala.collection.JavaConverters._

abstract class SparkDl4jNetworkWrapper[T, E <: SparkDl4jNetworkWrapper[T, E, M], M <: SparkDl4jModelWrapper[T, M]]
(override val uid: String,
 protected val multiLayerConfiguration : MultiLayerConfiguration,
 protected val numLabels: Int,
 protected val trainingMaster: ParamSerializer,
 protected val epochs : Int,
 protected val listeners : util.Collection[TrainingListener],
 protected val collectStats: Boolean = false
) extends Predictor[T, E, M]  {

    protected def mapVectorFunc : Row => LabeledPoint

    override def copy(extra: ParamMap) : E = defaultCopy(extra)

    override def setLabelCol(value: String): E = super.setLabelCol(value)

    override def setFeaturesCol(value: String): E = super.setFeaturesCol(value)

    protected def trainer(dataRowsFacade: DatasetFacade) : SparkDl4jMultiLayer = {
        val dataset = dataRowsFacade.get
        val sparkNet = new SparkDl4jMultiLayer(dataset.sqlContext.sparkContext, multiLayerConfiguration, trainingMaster())
        sparkNet.setCollectTrainingStats(collectStats)
        if (listeners != null) {
            sparkNet.setListeners(listeners)
        }
        val lps = toLabelPoint(dataRowsFacade)
        val epochsToUse = if (epochs < 1) 1 else epochs
        for (i <- List.range(0, epochsToUse)) {
            sparkNet.fit(lps)
        }
        sparkNet
    }

    protected def toLabelPoint(datasetFacade: DatasetFacade): RDD[DataSet] = {
        datasetFacade.get.select(getFeaturesCol, getLabelCol).rdd
            .map(mapVectorFunc)
            .map(item => {
                val features = item.features
                val label = item.label
                if (numLabels > 1) {
                    new DataSet(Nd4j.create(features.toArray), FeatureUtil.toOutcomeVector(label.toInt, numLabels))
                } else {
                    new DataSet(Nd4j.create(features.toArray), Nd4j.create(Array(label)))
                }
            })
    }
}


abstract class SparkDl4jModelWrapper[T, E <: SparkDl4jModelWrapper[T, E]](override val uid: String,
                                                                          network: MultiLayerNetwork,
                                                                          multiLayerConfiguration: MultiLayerConfiguration)
    extends PredictionModel[T, E] with Serializable with MLWritable {

    private var trainingStats : SparkTrainingStats = null.asInstanceOf[SparkTrainingStats]

    def getMultiLayerNetwork : MultiLayerNetwork = network

    protected def predictor(features: Vector) : Double = {
        val v = output(features)
        if (v.size > 1) {
            v.argmax
        } else if (v.size == 1) {
            v.toArray(0)
        } else throw new RuntimeException("Vector size must be greater than 0")
    }

    protected def output(vector: Vector) : Vector = {
        val predicted = outputTensor(vector)
        Vectors.dense(flattenTensor(predicted))
    }

    def setTrainingStats(sparkTrainingStats: SparkTrainingStats) : this.type = {
        this.trainingStats = sparkTrainingStats
        this
    }

    def getTrainingStats : SparkTrainingStats = trainingStats

    protected def outputTensor(vector: Vector) : INDArray = getMultiLayerNetwork.output(MLLibUtil.toVector(vector))

    protected def outputFlattenedTensor(vector: Vector) : Vector = Vectors.dense(flattenTensor(outputTensor(vector)))

    protected[SparkDl4jModelWrapper] class SparkDl4jModelWriter(instance: SparkDl4jModelWrapper[T,E]) extends MLWriter {
        override protected def saveImpl(path: String): Unit = {
            val mlnJson = multiLayerConfiguration.toJson
            val params = network.params().data().asDouble()
            val parallelized = List(RowFactory.create(mlnJson, params)).asJava
            val dataset = sqlContext.createDataFrame(parallelized, SparkDl4jUtil.createScheme())
            dataset.write.parquet(path)
        }
    }

    override def write : MLWriter = new SparkDl4jModelWriter(this)

    private def flattenTensor(indArray: INDArray) : Array[Double] = indArray.data().asDouble()
}

trait SparkDl4jModelWrap extends MLReadable[SparkDl4jModel] {

    override def read: MLReader[SparkDl4jModel] = new SparkDl4jReader

    override def load(path: String): SparkDl4jModel = super.load(path)

    private class SparkDl4jReader extends MLReader[SparkDl4jModel] {

        override def load(path: String) : SparkDl4jModel = {
            val results = sqlContext.read.schema(SparkDl4jUtil.createScheme()).parquet(path)
            val row = results.first()
            val mlcJson = row.getAs[String]("mlc")
            val params = row.getAs[Seq[Double]]("params")
            val mlc = MultiLayerConfiguration.fromJson(mlcJson)
            val mln = new MultiLayerNetwork(mlc, Nd4j.create(params.toArray))
            new SparkDl4jModel(Identifiable.randomUID("dl4j"), mln, mlc)
        }

    }
}