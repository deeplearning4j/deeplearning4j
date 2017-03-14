package org.deeplearning4j.spark.impl.transformers

import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil


final class SparkDl4jNetwork(override val uid: String) extends Predictor[Vector, SparkDl4jNetwork, SparkDl4jModel]  {

    private var _multiLayerConfiguration : MultiLayerConfiguration = _
    private var _numLabels : Int = 2
    private var _freq : Int = 10
    private var _trainingMaster : Serializer = _

    def this() = this(Identifiable.randomUID("dl4j"))

    override def train(dataset: DataFrame) : SparkDl4jModel = {
        val sparkNet = new SparkDl4jMultiLayer(dataset.sqlContext.sparkContext, _multiLayerConfiguration, _trainingMaster())
        val lps = dataset.select(getFeaturesCol, getLabelCol).rdd
            .map(row => new LabeledPoint(row.getAs[Double](getLabelCol), row.getAs[Vector](getFeaturesCol)))
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
        new SparkDl4jModel(uid, sparkNet)
    }

    override def copy(extra: ParamMap) : SparkDl4jNetwork = defaultCopy(extra)

    def setNumLabels(value: Int) : SparkDl4jNetwork = {
        this._numLabels = value
        this
    }

    def setMultiLayerConfiguration(multiLayerConfiguration: MultiLayerConfiguration) : SparkDl4jNetwork = {
        this._multiLayerConfiguration = multiLayerConfiguration
        this
    }

    def setTrainingMaster(tm: Serializer) : SparkDl4jNetwork = {
        this._trainingMaster = tm
        this
    }
}

class SparkDl4jModel(override val uid: String, network: SparkDl4jMultiLayer)
    extends PredictionModel[Vector, SparkDl4jModel] with Serializable with MLWritable {

    override def copy(extra: ParamMap) : SparkDl4jModel = {
        copyValues(new SparkDl4jModel(uid, network)).setParent(parent)
    }

    override def predict(features: Vector) : Double = {
        val v = output(features)
        if (v.size > 1) {
            v.argmax
        } else if (v.size == 1) {
            v.toArray(0)
        } else throw new RuntimeException("Vector size must be greater than 1")
    }

    def getMultiLayerNetwork : MultiLayerNetwork = network.getNetwork

    def output(vector: Vector) : Vector = network.predict(vector)

    protected[SparkDl4jModel] class SparkDl4jModelWriter(instance: SparkDl4jModel) extends MLWriter {
        override protected def saveImpl(path: String): Unit = {
            ModelSerializer.writeModel(network.getNetwork, path, true)
        }
    }

    override def write : MLWriter = new SparkDl4jModelWriter(this)

    object SparkDl4jNetworkModel extends MLReadable[SparkDl4jMultiLayer] {
        override def read: MLReader[SparkDl4jMultiLayer] = new SparkDl4jReader
        override def load(path: String): SparkDl4jMultiLayer = super.load(path)
        private class SparkDl4jReader extends MLReader[SparkDl4jMultiLayer] {
            override def load(path: String) : SparkDl4jMultiLayer = {
                val mln = ModelSerializer.restoreMultiLayerNetwork(path)
                new SparkDl4jMultiLayer(sc, mln, null)
            }
        }
    }

}

trait Serializer extends Serializable {
    def apply() : ParameterAveragingTrainingMaster
}
