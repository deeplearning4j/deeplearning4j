package org.deeplearning4j.spark.ml.impl

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Dataset, Row}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.DatasetFacade


final class SparkDl4jNetwork(override val uid: String) extends SparkDl4jNetworkWrapper[Vector, SparkDl4jNetwork, SparkDl4jModel](uid) {

    override val mapVectorFunc: Row => LabeledPoint = row => new LabeledPoint(row.getAs[Double]($(labelCol)), Vectors.fromML(row.getAs[Vector]($(featuresCol))))

    def this() = this(Identifiable.randomUID("dl4j"))

    override def train(dataset: Dataset[_]): SparkDl4jModel = {
        val spn = trainer(DatasetFacade.dataRows(dataset))
        new SparkDl4jModel(uid, spn)
    }
}

class SparkDl4jModel(override val uid: String, network: SparkDl4jMultiLayer)
    extends SparkDl4jModelWrapper[Vector, SparkDl4jModel](uid, network) {

    override def copy(extra: ParamMap) : SparkDl4jModel = {
        copyValues(new SparkDl4jModel(uid, network)).setParent(parent)
    }

    override def predict(features: Vector) : Double = {
        predictor(Vectors.fromML(features))
    }

    def output(vector: Vector): Vector = org.apache.spark.ml.linalg.Vectors.dense(super.output(Vectors.fromML(vector)).toArray)

}

object SparkDl4jModel extends SparkDl4jModelWrap
