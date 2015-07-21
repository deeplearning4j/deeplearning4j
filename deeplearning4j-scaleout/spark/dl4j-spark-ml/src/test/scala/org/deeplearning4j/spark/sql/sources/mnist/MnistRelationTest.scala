package org.deeplearning4j.spark.sql.sources.mnist

import org.apache.spark.Logging
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.{Matchers, FunSuite}
import org.scalatest.junit.JUnitRunner
import org.springframework.core.io.ClassPathResource

/**
 * Test Mnist relation
 */
@RunWith(classOf[JUnitRunner])
class MnistRelationTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  val labels = new ClassPathResource("/data/t10k-labels-idx1-ubyte", classOf[MnistRelationTest]).getURI.toString
  val images = new ClassPathResource("/data/t10k-images-idx3-ubyte", classOf[MnistRelationTest]).getURI.toString

  test("select") {
    val df = sqlContext.read.format(classOf[DefaultSource].getName)
      .option("labelsPath", labels)
      .option("imagesPath", images)
      .load()

    df.count() shouldEqual 10000
    df.select("label").count() shouldEqual 10000
    df.select("features").count() shouldEqual 10000
    df.show
  }

}

