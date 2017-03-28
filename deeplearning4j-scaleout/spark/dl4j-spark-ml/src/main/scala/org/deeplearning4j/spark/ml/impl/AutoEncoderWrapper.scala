package org.deeplearning4j.spark.ml.impl

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{Row, UserDefinedFunction}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer, SparkDl4jUtil}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j


trait AutoEncoderParams extends Params {

    final val compressedLayer  = new IntParam(this, "outputLayer", "Layer where to place autoencoder")
    def getCompressedLayer : Int = $(compressedLayer)
    final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
    final def getInputCol: String = $(inputCol)
    final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
    setDefault(outputCol, uid + "__output")
    /** @group getParam */
    final def getOutputCol: String = $(outputCol)
}

abstract class AutoEncoderWrapper[E <: AutoEncoderWrapper[E, M], M <: AutoEncoderModelWrapper[M]](override val uid: String) extends Estimator[M] with AutoEncoderParams {
    private var _multiLayerConfiguration : MultiLayerConfiguration = _
    private var _trainingMaster : ParamSerializer = _

    def this() = this(Identifiable.randomUID("dl4j_autoencoder"))

    protected def mapVectorFunc : Row => Vector

    override def copy(extra: ParamMap) : E = defaultCopy(extra)

    protected def fitter(datasetFacade: DatasetFacade) : SparkDl4jMultiLayer = {
        validator()
        val dataset = datasetFacade.get
        val sparkNet = new SparkDl4jMultiLayer(dataset.sqlContext.sparkContext, _multiLayerConfiguration, _trainingMaster())
        val layerCount = sparkNet.getNetwork.getLayers.length
        if ($(compressedLayer) < 1 || $(compressedLayer) >= layerCount) {
            set(compressedLayer, layerCount / 2)
        }
        val set2 = dataset.select($(inputCol)).rdd.map(mapVectorFunc)
        val ds = set2.map(v => new DataSet(Nd4j.create(v.toArray), Nd4j.create(v.toArray)))
        val fitted = sparkNet.fit(ds)
        sparkNet
    }

    def setCompressedLayer(value: Int) : E = {
        set(compressedLayer, value)
        this.asInstanceOf[E]
    }

    def setInputCol(value: String) : E = {
        set(inputCol, value)
        this.asInstanceOf[E]
    }

    def setOutputCol(value: String) : E = {
        set(outputCol, value)
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

    private def validator(): Unit ={
        if (_multiLayerConfiguration == null) {
            throw new RuntimeException("multiLayerConfiguration must be set")
        }
        if (_trainingMaster == null) {
            throw new RuntimeException("Training Master must be set")
        }
    }
}

abstract class AutoEncoderModelWrapper[E <: AutoEncoderModelWrapper[E]](override val uid : String,
                                                                        sparkDl4jMultiLayer: SparkDl4jMultiLayer) extends Model[E] with MLWritable with AutoEncoderParams {

    protected def udfTransformer : UserDefinedFunction

    def getNetwork : MultiLayerNetwork = sparkDl4jMultiLayer.getNetwork

    def setCompressedLayer(value: Int) : E = {
        set(compressedLayer, value)
        this.asInstanceOf[E]
    }

    def setInputCol(value: String) : E = {
        set(inputCol, value)
        this.asInstanceOf[E]
    }

    def setOutputCol(value: String) : E = {
        set(outputCol, value)
        this.asInstanceOf[E]
    }

    override def write : MLWriter = new AutoEncoderWriter(this)

    protected[AutoEncoderModelWrapper] class AutoEncoderWriter(instance: AutoEncoderModelWrapper[E]) extends MLWriter {
        override protected def saveImpl(path: String) : Unit = {
            SparkDl4jUtil.saveMetadata(instance, path, sc)
            ModelSerializer.writeModel(instance.getNetwork, path, true)
        }
    }

}

trait AutoEncoderModelLoader extends MLReadable[AutoEncoderModel] {

    override def read: MLReader[AutoEncoderModel] = new AutoEncoderReader

    override def load(path: String) : AutoEncoderModel = super.load(path)

    private class AutoEncoderReader extends MLReader[AutoEncoderModel] {

        private val className = classOf[AutoEncoderModel].getName

        override def load(path: String) : AutoEncoderModel = {
            val metaData = SparkDl4jUtil.loadMetadata(path, sc, className)
            val mln = ModelSerializer.restoreMultiLayerNetwork(path)
            val model = new AutoEncoderModel(
                Identifiable.randomUID("dl4j_autoencoder"),
                new SparkDl4jMultiLayer(sc, mln, null)
            )
            SparkDl4jUtil.getAndSetParams(model, metaData)
            model
        }

    }
}
