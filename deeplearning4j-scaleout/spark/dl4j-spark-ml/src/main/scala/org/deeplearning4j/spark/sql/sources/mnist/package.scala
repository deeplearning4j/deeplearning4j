package org.deeplearning4j.spark.sql.sources

import org.apache.spark.sql.SQLContext
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource._

package object mnist {
  /**
   * Adds a method, `mnist`, to SQLContext that allows reading the Mnist dataset.
   */
  implicit class IrisContext(sqlContext: SQLContext) {
    def mnist(imagesPath: String, labelsPath: String) = {
      val parameters = Map(ImagesPath -> imagesPath, LabelsPath -> labelsPath)
      sqlContext.read.format(classOf[DefaultSource].getName).options(parameters).load()
    }
  }
}
