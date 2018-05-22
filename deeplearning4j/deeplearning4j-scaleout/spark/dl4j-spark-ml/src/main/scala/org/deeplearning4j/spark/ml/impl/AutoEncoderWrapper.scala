package org.deeplearning4j.spark.ml.impl

import org.apache.spark.ml.param.{IntParam, Param, ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql._
import org.apache.spark.sql.expressions._
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer, SparkDl4jUtil}
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

import scala.collection.JavaConverters._


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
    protected var _multiLayerConfiguration : MultiLayerConfiguration = _
    protected var _trainingMaster : ParamSerializer = _

    def this() = this(Identifiable.randomUID("dl4j_autoencoder"))

    protected def mapVectorFunc : Row => Vector

    override def copy(extra: ParamMap) : E = defaultCopy(extra)

    protected def fitter(datasetFacade: DatasetFacade) : MultiLayerNetwork = {
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
        fitted
    }

    /**
      * The compressed layer of the autoencoder
      * @param value 0 based index of the compressed layer
      * @return Returns the class that extends the wrapper
      */
    def setCompressedLayer(value: Int) : E = {
        set(compressedLayer, value)
        this.asInstanceOf[E]
    }

    /**
      * The input column name
      * @param value name of the input column
      * @return Returns the class that extends the wrapper
      */
    def setInputCol(value: String) : E = {
        set(inputCol, value)
        this.asInstanceOf[E]
    }

    /**
      * The output column name
      * @param value name of the output column
      * @return Returns the class that extends the wrapper
      */
    def setOutputCol(value: String) : E = {
        set(outputCol, value)
        this.asInstanceOf[E]
    }

    /**
      * The multilayer configuration of the autoencoder
      * @param multiLayerConfiguration MultiLayerConfiguration
      * @return Returns the class that extends the wrapper
      */
    def setMultiLayerConfiguration(multiLayerConfiguration: MultiLayerConfiguration) : E = {
        this._multiLayerConfiguration = multiLayerConfiguration
        this.asInstanceOf[E]
    }

    /**
      * The training master configuration for the spark network
      * @param tm ParamSerializer -> a wrapper around the Training Master
      * @return Returns the class that extends the wrapper
      */
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
                                                                        multiLayerNetwork: MultiLayerNetwork,
                                                                        multiLayerConfiguration: MultiLayerConfiguration)
    extends Model[E] with MLWritable with AutoEncoderParams {

    protected def udfTransformer : UserDefinedFunction

    /**
      *
      * @return Returns the multiLayerNetwork
      */
    def getNetwork : MultiLayerNetwork = multiLayerNetwork

    /**
      * The compressed layer of the autoencoder
      * @param value Int
      * @return Returns the class that extends the wrapper
      */
    def setCompressedLayer(value: Int) : E = {
        set(compressedLayer, value)
        this.asInstanceOf[E]
    }

    /**
      * The input column name
      * @param value name of the input column
      * @return Returns the class that extends the wrapper
      */
    def setInputCol(value: String) : E = {
        set(inputCol, value)
        this.asInstanceOf[E]
    }

    /**
      * The output column name
      * @param value name of the output column
      * @return Returns the class that extends the wrapper
      */
    def setOutputCol(value: String) : E = {
        set(outputCol, value)
        this.asInstanceOf[E]
    }

    override def write : MLWriter = new AutoEncoderWriter(this)

    protected[AutoEncoderModelWrapper] class AutoEncoderWriter(instance: AutoEncoderModelWrapper[E]) extends MLWriter {
        override protected def saveImpl(path: String) : Unit = {
            SparkDl4jUtil.saveMetadata(instance, path, sc)
            val mlnJson = multiLayerConfiguration.toJson
            val params = multiLayerNetwork.params().data().asDouble()
            val parallelized = List(RowFactory.create(mlnJson, params)).asJava
            val dataset = sqlContext.createDataFrame(parallelized, SparkDl4jUtil.createScheme())
            dataset.write.parquet(path)
        }
    }

}

trait AutoEncoderModelLoader extends MLReadable[AutoEncoderModel] {

    override def read: MLReader[AutoEncoderModel] = new AutoEncoderReader

    /**
      * Loads the autoencoder model
      * @param path where to load the model
      * @return AutoEncoderModel
      */
    override def load(path: String) : AutoEncoderModel = super.load(path)

    private class AutoEncoderReader extends MLReader[AutoEncoderModel] {

        private val className = classOf[AutoEncoderModel].getName

        override def load(path: String) : AutoEncoderModel = {
            val metaData = SparkDl4jUtil.loadMetadata(path, sc, className)
            val results = sqlContext.read.schema(SparkDl4jUtil.createScheme()).parquet(path)
            val row = results.first()
            val mlcJson = row.getAs[String]("mlc")
            val params = row.getAs[Seq[Double]]("params")
            val mlc = MultiLayerConfiguration.fromJson(mlcJson)
            val mln = new MultiLayerNetwork(mlc, Nd4j.create(params.toArray))
            val model = new AutoEncoderModel(Identifiable.randomUID("dl4j"), mln, mlc)
            SparkDl4jUtil.getAndSetParams(model, metaData)
            model
        }

    }
}
