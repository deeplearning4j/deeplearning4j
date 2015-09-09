package org.deeplearning4j.spark.ml.classification

import org.apache.spark.Logging
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.DataFrame
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, Updater, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.sql.sources.iris._
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.scalatest._
import org.scalatest.junit.JUnitRunner
import org.springframework.core.io.ClassPathResource

/**
 * Test classification.
 */
@RunWith(classOf[JUnitRunner])
class NeuralNetworkClassificationTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  private def getConfiguration(): MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .seed(11L)
      .iterations(1)
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
      .build()
  }

  var trainDF: DataFrame = null
  var testDF: DataFrame = null
  var emptyDF: DataFrame = null

  override def beforeAll(): Unit = {
    super.beforeAll()

    val sql = sqlContext
    import sql.implicits._

    val path = new ClassPathResource("data/irisSvmLight.txt").getFile.toURI.toString
    val dataFrame = sqlContext.read.iris(path).select("features", "label")
    val Array(trainDF, testDF) = dataFrame.randomSplit(Array(.6, .4), 11L)
    this.trainDF = trainDF
    this.testDF = testDF

    this.emptyDF = sqlContext.sparkContext.parallelize(Seq[LabeledPoint]()).toDF
  }

  test("input") {

    val conf = getConfiguration()
    val classification = new NeuralNetworkClassification()
      .setFeaturesCol("features").setLabelCol("label")
      .setConf(conf)

    Some(classification.fit(emptyDF)) map { m =>
      Some(m.transform(emptyDF)) map { df =>
        df.show()
      }
    }

    Some(classification.fit(trainDF)) map { m =>
      Some(m.transform(testDF)) map { df =>
        df.show()
      }
    }
  }

  test("model pruning") {

    val conf = getConfiguration()
    val classification = new NeuralNetworkClassification()
      .setFeaturesCol("features").setLabelCol("label")
      .setConf(conf)
    val model = classification.fit(trainDF)

    Some(model.copy(ParamMap()).transform(testDF)) map { df =>
      df.columns should contain theSameElementsInOrderAs Seq("features", "label", "prediction", "rawPrediction")
      Some(df.take(1)(0)).map { row =>
        row.length should be (4)
        row.get(0) shouldBe a [Vector]
        row.get(1) shouldBe a [java.lang.Double]
        row.get(2) shouldBe a [java.lang.Double]
        row.get(3) shouldBe a [Vector]
      }
      df.show()
    }

    Some(model.copy(ParamMap(model.rawPredictionCol -> "")).transform(testDF)) map { df =>
      df.columns should contain theSameElementsInOrderAs Seq("features", "label", "prediction")
      Some(df.take(1)(0)).map { row =>
        row.length should be (3)
        row.get(0) shouldBe a [Vector]
        row.get(1) shouldBe a [java.lang.Double]
        row.get(2) shouldBe a [java.lang.Double]
      }
      df.show()
    }

    Some(model.copy(ParamMap(model.predictionCol -> "")).transform(testDF)) map { df =>
      df.columns should contain theSameElementsInOrderAs Seq("features", "label", "rawPrediction")
      Some(df.take(1)(0)).map { row =>
        row.length should be (3)
        row.get(0) shouldBe a [Vector]
        row.get(1) shouldBe a [java.lang.Double]
        row.get(2) shouldBe a [Vector]
      }
      df.show()
    }

    Some(model.copy(ParamMap(model.predictionCol -> "", model.rawPredictionCol -> "")).transform(testDF)) map { df =>
      df.columns should contain theSameElementsInOrderAs Seq("features", "label")
      Some(df.take(1)(0)).map { row =>
        row.length should be (2)
        row.get(0) shouldBe a [Vector]
        row.get(1) shouldBe a [java.lang.Double]
      }
      df.show()
    }
  }

}
