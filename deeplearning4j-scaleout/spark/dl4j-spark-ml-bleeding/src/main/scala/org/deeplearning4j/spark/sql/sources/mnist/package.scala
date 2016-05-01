package org.deeplearning4j.spark.sql.sources

import org.apache.spark.sql.{DataFrameReader, SQLContext}
import org.deeplearning4j.spark.sql.sources.mnist.DefaultSource._

package object mnist {
  /**
   * Adds a method, `mnist`, to SQLContext that allows reading the Mnist dataset.
   */
  implicit class MnistContext(sqlContext: SQLContext) {
    @Deprecated
    def mnist(imagesPath: String, labelsPath: String) = {
      sqlContext.read.mnist(imagesPath, labelsPath)
    }
  }

  /**
   * Adds a method, `mnist`, to DataFrameReader that allows reading the Mnist dataset.
   */
  implicit class MnistDataFrameReader(read: DataFrameReader) {
    def mnist(imagesPath: String, labelsPath: String) = {
      val parameters = Map(ImagesPath -> imagesPath, LabelsPath -> labelsPath)
      read.format(classOf[DefaultSource].getName).options(parameters).load()
    }
  }
}
