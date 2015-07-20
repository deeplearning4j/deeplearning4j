package org.deeplearning4j.spark.sql.sources.mnist

import org.apache.spark.Logging
import org.deeplearning4j.spark.util.TestSparkContext
import org.junit.runner.RunWith
import org.scalatest.{Matchers, FunSuite}
import org.scalatest.junit.JUnitRunner
import org.springframework.core.io.ClassPathResource

/**
 * @author Eron Wright
 */
@RunWith(classOf[JUnitRunner])
class MnistRelationTest
  extends FunSuite with TestSparkContext with Logging with Matchers {

  val labels = new ClassPathResource("/mnist2500_labels.txt").getFile.getAbsolutePath
  val images = new ClassPathResource("/mnist2500_X.txt").getFile.getAbsolutePath

  test("pruning") {
    val dataFrame = sqlContext.read.format(classOf[DefaultSource].getName)
      .option("labels_file", labels)
      .option("images_file", images)
      .load()

    dataFrame.show
  }
}

