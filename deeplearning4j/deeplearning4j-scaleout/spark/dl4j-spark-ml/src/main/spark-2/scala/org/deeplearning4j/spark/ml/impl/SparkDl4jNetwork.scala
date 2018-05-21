package org.deeplearning4j.spark.ml.impl

import java.util

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{Dataset, Row}
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.ml.utils.{DatasetFacade, ParamSerializer}
import org.nd4j.linalg.api.ndarray.INDArray

final class SparkDl4jNetwork(
                                override val multiLayerConfiguration: MultiLayerConfiguration,
                                override val numLabels: Int,
                                override val trainingMaster: ParamSerializer,
                                override val epochs : Int,
                                override val listeners: util.Collection[TrainingListener],
                                override val collectStats: Boolean = false,
                                override val uid: String = Identifiable.randomUID("dl4j"))
    extends SparkDl4jNetworkWrapper[Vector, SparkDl4jNetwork, SparkDl4jModel](
        uid, multiLayerConfiguration, numLabels, trainingMaster, epochs, listeners, collectStats) {

    def this(multiLayerConfiguration: MultiLayerConfiguration, numLabels: Int, trainingMaster: ParamSerializer,
             epochs: Int,  listeners: util.Collection[TrainingListener]) {
        this(multiLayerConfiguration, numLabels, trainingMaster, epochs, listeners, false, Identifiable.randomUID("dl4j"))
    }

    def this(multiLayerConfiguration: MultiLayerConfiguration, numLabels: Int, trainingMaster: ParamSerializer, epochs: Int,
             listeners: util.Collection[TrainingListener], collectStats: Boolean) {
        this(multiLayerConfiguration, numLabels, trainingMaster, epochs, listeners, collectStats, Identifiable.randomUID("dl4j"))
    }

    override val mapVectorFunc: Row => LabeledPoint = row => new LabeledPoint(row.getAs[Double]($(labelCol)), Vectors.fromML(row.getAs[Vector]($(featuresCol))))

    /**
      * Trains the dataset with the spark multi-layer network
      * @param dataset Dataframe
      * @return returns a SparkDl4jModel
      */
    override def train(dataset: Dataset[_]): SparkDl4jModel = {
        val spn = trainer(DatasetFacade.dataRows(dataset))
        handleTrainedData(spn)
    }

    private def handleTrainedData(spn: SparkDl4jMultiLayer) : SparkDl4jModel = {
        val model = new SparkDl4jModel(uid, spn.getNetwork, multiLayerConfiguration)
        if (collectStats) model.setTrainingStats(spn.getSparkTrainingStats)
        else model
    }
}

class SparkDl4jModel(override val uid: String, network: MultiLayerNetwork, multiLayerConfiguration: MultiLayerConfiguration)
    extends SparkDl4jModelWrapper[Vector, SparkDl4jModel](uid, network, multiLayerConfiguration) {

    override def copy(extra: ParamMap) : SparkDl4jModel = {
        copyValues(new SparkDl4jModel(uid, network, multiLayerConfiguration)).setParent(parent)
    }

    /**
      * Argmax prediction for classification, and continuous for regression.
      * @param features Vector to predict
      * @return a double of the outcome
      */
    override def predict(features: Vector) : Double = {
        predictor(Vectors.fromML(features))
    }

    /**
      * Vector of the network output
      * @param vector features to predict
      * @return Vector of the network output
      */
    def output(vector: Vector): Vector = org.apache.spark.ml.linalg.Vectors.dense(super.output(Vectors.fromML(vector)).toArray)

    def outputFlattenedTensor(vector: Vector) : Vector = org.apache.spark.ml.linalg.Vectors.dense(super.outputFlattenedTensor(Vectors.fromML(vector)).toArray)

    def outputTensor(vector: Vector) : INDArray = super.outputTensor(Vectors.fromML(vector))

}

object SparkDl4jModel extends SparkDl4jModelWrap
