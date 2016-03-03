package org.deeplearning4j.spark.ml.feature

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.nd4j.linalg.io.ClassPathResource
import org.scalatest._
import org.scalatest.junit.JUnitRunner

/**
 * Test ImageVectorizer.
 */
@RunWith(classOf[JUnitRunner])
@org.junit.Ignore
class ImageVectorizerTest
  extends FunSuite with TestSparkContext with Logging with Matchers {
  var imageDF: DataFrame = null

  override def beforeAll(): Unit = {
    super.beforeAll()

    val sql = sqlContext
    import sql.implicits._

    val path = new ClassPathResource("data/image.hsf").getFile.toURI.toString
    this.imageDF = sql.sparkContext.sequenceFile(path, classOf[Text], classOf[BytesWritable])
      .map { case (path, bytes) => (path.toString, bytes.getBytes()) }
      .toDF("path", "bytes")
  }

  test("transform") {
    val vectorizer = new ImageVectorizer()
      .setInputCol("bytes")
      .setOutputCol("image_vector")
      .setHeight(32)
      .setWidth(32)
      .setChannels(3)

    Some(vectorizer.transform(imageDF)) map { df =>
      df.columns should contain theSameElementsInOrderAs Seq("path", "bytes", "image_vector")
      Some(df.take(1)(0)).map { row =>
        row.length should be (3)
        row.get(0) shouldBe a [String]
        row.get(2) shouldBe a [Vector]
        row.getAs[Vector](2).size shouldBe 32 * 32 * 3
      }
      df.show()
    }
  }
}
