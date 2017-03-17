package org.deeplearning4j.spark.ml.impl

import java.util

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.Row
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer}
import org.deeplearning4j.util.ModelSerializer


abstract class SparkDl4jNetworkWrapper[T, E <: SparkDl4jNetworkWrapper[T, E, M], M <: SparkDl4jModelWrapper[T, M]]
(override val uid: String) extends Predictor[T, E, M]  {

    protected var _multiLayerConfiguration : MultiLayerConfiguration = _
    protected var _numLabels : Int = 2
    protected var _freq : Int = 10
    protected var _trainingMaster : ParamSerializer = _
    protected var _listeners : util.Collection[IterationListener] = _
    protected var _collectStats: Boolean = false

    override def copy(extra: ParamMap) : E = defaultCopy(extra)

    val mapVectorFunc : Row => LabeledPoint

    def setNumLabels(value: Int) : E = {
        this._numLabels = value
        this.asInstanceOf[E]
    }

    def setMultiLayerConfiguration(multiLayerConfiguration: MultiLayerConfiguration) : E = {
        this._multiLayerConfiguration = multiLayerConfiguration
        this.asInstanceOf[E]
    }

    def setTrainingMaster(tm: ParamSerializer) : E = {
        this._trainingMaster = tm
        this.asInstanceOf[E]
    }

    def setListeners(iterationListener: util.Collection[IterationListener]) : E = {
        this._listeners = iterationListener
        this.asInstanceOf[E]
    }

    def setCollectionStats(collectStats: Boolean) : E = {
        this._collectStats = collectStats
        this.asInstanceOf[E]
    }

    override def setLabelCol(value: String): E = super.setLabelCol(value)

    protected def trainer(dataRowsFacade: DatasetFacade) : SparkDl4jMultiLayer = {
        val dataset = dataRowsFacade.get
        val sparkNet = new SparkDl4jMultiLayer(dataset.sqlContext.sparkContext, _multiLayerConfiguration, _trainingMaster())
        sparkNet.setCollectTrainingStats(_collectStats)
        if (_listeners != null) {
            sparkNet.setListeners(_listeners)
        }
        val lps = dataset.select(getFeaturesCol, getLabelCol).rdd
            .map(mapVectorFunc)
            .map(item => {
                val features = item.features
                val label = item.label
                if (_numLabels > 1) {
                    new DataSet(Nd4j.create(features.toArray), FeatureUtil.toOutcomeVector(label.asInstanceOf[Int], _numLabels))
                } else {
                    new DataSet(Nd4j.create(features.toArray), Nd4j.create(Array(label)))
                }
            })
        sparkNet.fit(lps)
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

    protected[SparkDl4jModelWrapper] class SparkDl4jModelWriter(instance: SparkDl4jModelWrapper[T,E]) extends MLWriter {
        override protected def saveImpl(path: String): Unit = {
            ModelSerializer.writeModel(network.getNetwork, path, true)
        }
    }

    override def write : MLWriter = new SparkDl4jModelWriter(this)
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