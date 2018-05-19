package org.deeplearning4j.spark.ml.impl

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.ml.utils._
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType


class AutoEncoder(uid: String) extends AutoEncoderWrapper[AutoEncoder, AutoEncoderModel](uid){

    def this() {
        this(Identifiable.randomUID("dl4j"))
    }

    override def mapVectorFunc = row => org.apache.spark.mllib.linalg.Vectors.fromML(row.get(0).asInstanceOf[Vector])

    /**
      * Fits a dataset to the specified network configuration
      * @param dataset DataFrame
      * @return Returns an autoencoder model, which can run transformations on the vector
      */
    override def fit(dataset: Dataset[_]) : AutoEncoderModel = {
        val sparkdl4j = fitter(DatasetFacade.dataRows(dataset))
        new AutoEncoderModel(uid, sparkdl4j, _multiLayerConfiguration)
            .setInputCol($(inputCol))
            .setOutputCol($(outputCol))
            .setCompressedLayer($(compressedLayer))
    }

    override def transformSchema(schema: StructType) : StructType = {
        SchemaUtils.appendColumn(schema, $(outputCol), VectorType, false)
    }
}

class AutoEncoderModel(uid: String, multiLayerNetwork: MultiLayerNetwork, multiLayerConfiguration: MultiLayerConfiguration)
    extends AutoEncoderModelWrapper[AutoEncoderModel](uid, multiLayerNetwork, multiLayerConfiguration) {

    override def udfTransformer = udf[Vector, Vector](vec => {
        val out = multiLayerNetwork.feedForwardToLayer($(compressedLayer), Nd4j.create(vec.toArray))
        val mainLayer = out.get(out.size() - 1)

        // FIXME: int cast
        val size = mainLayer.size(1).toInt
        val values = Array.fill(size)(0.0)
        for (i <- 0 until size) {
            values(i) = mainLayer.getDouble(i.toLong)
        }
        Vectors.dense(values)
    })

    /**
      * copys an autoencoder model, including the param map
      * @param extra ParamMap
      * @return returns a copy of the autoencoder model
      */
    override def copy(extra: ParamMap) : AutoEncoderModel = {
        copyValues(new AutoEncoderModel(uid, multiLayerNetwork, multiLayerConfiguration)).setParent(parent)
    }

    /**
      * Transforms an incoming dataset
      * @param dataFrame DataFrame
      * @return Returns a transformed dataframe.
      */
    override def transform(dataFrame: Dataset[_]) : Dataset[Row] = {
        dataFrame.withColumn($(outputCol), udfTransformer(col($(inputCol))))
    }

    /**
      * Updates the schema from the new dataset
      * @param schema StructType
      * @return Returns a struct type
      */
    override def transformSchema(schema: StructType) : StructType = {
        SchemaUtils.appendColumn(schema, $(outputCol), VectorType, false)
    }
}

object AutoEncoderModel extends AutoEncoderModelLoader