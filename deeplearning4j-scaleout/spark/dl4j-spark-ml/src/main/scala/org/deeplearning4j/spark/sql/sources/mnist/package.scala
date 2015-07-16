package org.deeplearning4j.spark.sql.sources

import org.apache.spark.sql.SQLContext
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource

package object mnist {
  /**
   * Adds a method, `mnist`, to SQLContext that allows reading the Mnist dataset.
   */
  implicit class IrisContext(sqlContext: SQLContext) {
    def mnist(imagesFile: String, labelsFile: String, numExamples: Int) = {
      val parameters = Map("images_file" -> imagesFile,
        "labels_file" -> labelsFile, "num_examples" -> numExamples.toString)
      sqlContext.read.format(classOf[DefaultSource].getName).options(parameters).load()
    }
  }
}
