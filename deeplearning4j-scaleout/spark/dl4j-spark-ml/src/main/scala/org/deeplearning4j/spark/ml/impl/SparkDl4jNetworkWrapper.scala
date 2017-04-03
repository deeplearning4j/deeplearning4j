package org.deeplearning4j.spark.ml.impl

import java.util

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer}
import org.deeplearning4j.spark.util.MLLibUtil
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import org.nd4j.linalg.api.ndarray.INDArray


abstract class SparkDl4jNetworkWrapper[T, E <: SparkDl4jNetworkWrapper[T, E, M], M <: SparkDl4jModelWrapper[T, M]]
(override val uid: String,
 protected val multiLayerConfiguration : MultiLayerConfiguration,
 protected val numLabels: Int,
 protected val trainingMaster: ParamSerializer,
 protected val epochs : Int,
 protected val listeners : util.Collection[IterationListener],
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
        val lps = dataset.select(getFeaturesCol, getLabelCol).rdd
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
        val epochsToUse = if (epochs < 1) 1 else epochs
        for (i <- List.range(0, epochsToUse)) {
            sparkNet.fit(lps)
        }
        sparkNet
    }
}

abstract class SparkDl4jModelWrapper[T, E <: SparkDl4jModelWrapper[T, E]](override val uid: String, network: SparkDl4jMultiLayer)
    extends PredictionModel[T, E] with Serializable with MLWritable {

    def getMultiLayerNetwork : MultiLayerNetwork = network.getNetwork

    protected def predictor(features: Vector) : Double = {
        val v = output(features)
        if (v.size > 1) {
            v.argmax
        } else if (v.size == 1) {
            v.toArray(0)
        } else throw new RuntimeException("Vector size must be greater than 0")
    }

    protected def output(vector: Vector) : Vector = network.predict(vector)

    protected def outputTensor(vector: Vector) : INDArray = getMultiLayerNetwork.output(MLLibUtil.toVector(vector))

    protected def outputFlattenedTensor(vector: Vector) : Vector = Vectors.dense(flattenTensor(outputTensor(vector)))

    protected[SparkDl4jModelWrapper] class SparkDl4jModelWriter(instance: SparkDl4jModelWrapper[T,E]) extends MLWriter {
        override protected def saveImpl(path: String): Unit = {
            ModelSerializer.writeModel(network.getNetwork, path, true)
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
            val mln = ModelSerializer.restoreMultiLayerNetwork(path)
            new SparkDl4jModel(Identifiable.randomUID("dl4j"), new SparkDl4jMultiLayer(sc, mln, null))
        }

    }
}