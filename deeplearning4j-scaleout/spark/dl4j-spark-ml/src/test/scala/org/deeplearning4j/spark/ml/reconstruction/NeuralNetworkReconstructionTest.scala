package org.deeplearning4j.spark.ml.reconstruction

import org.apache.spark.Logging
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{Updater, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.sql.sources.iris.DefaultSource
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.scalatest._
import org.scalatest.junit.JUnitRunner
import org.springframework.core.io.ClassPathResource

/**
 * Test reconstruction.
 */
@RunWith(classOf[JUnitRunner])
class NeuralNetworkReconstructionTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  private def getConfiguration(): MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .seed(11L)
      .iterations(100)
      .learningRate(1e-3f)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .momentum(0.9)
      .constrainGradientToUnitNorm(true)
      .useDropConnect(true)
      .list(2)
      .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
        .nIn(4).nOut(3).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD).activation("relu").dropOut(0.5)
        .lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nIn(3).nOut(3).weightInit(WeightInit.XAVIER).updater(Updater.ADAGRAD).activation("softmax").dropOut(0.5).build())
      .build();
  }

  test("iris") {
    val conf = getConfiguration()

    val path = new ClassPathResource("data/irisSvmLight.txt").getFile.toURI.toString
    val dataFrame = sqlContext.read.format(classOf[DefaultSource].getName).load(path)
    val Array(trainDF, testDF) = dataFrame.randomSplit(Array(.6,.4), 11L)

    val classification = new NeuralNetworkReconstruction()
      .setFeaturesCol("features")
      .setReconstructionCol("reconstruction")
      .setConf(conf)

    val model = classification.fit(trainDF)
    val predictions = model.transform(testDF)

    predictions.col("reconstruction") should not be (null)
    predictions.show()
  }
}
