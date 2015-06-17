package org.deeplearning4j.spark.ml.reconstruction

import org.apache.spark.Logging
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.`override`.ConfOverride
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.conf.rng.DefaultRandom
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

  test("iris") {
    val conf = new NeuralNetConfiguration.Builder()
      .rng(new DefaultRandom(11L))
      .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
      .nIn(4).nOut(3)
      .layer(new RBM)
      .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
      .activationFunction("tanh")
      .list(2)
      .hiddenLayerSizes(3)
      .`override`(1, new ConfOverride() {
      def overrideLayer(i: Int, builder: NeuralNetConfiguration.Builder) {
        if (i == 1) {
          builder.activationFunction("softmax")
          builder.layer(new OutputLayer)
          builder.lossFunction(LossFunctions.LossFunction.MCXENT)
        }
      }
    }).build

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
