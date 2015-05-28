
package org.apache.spark.sql.sources

import org.apache.spark.sql.{SQLContext, DataFrame}

package object lfw {

  /**
   * Adds a method, `lfw`, to SQLContext that allows reading the LFW dataset.
   */
  implicit class LfwContext(sqlContext: SQLContext) {
    def lfw(filePath: String) =
      sqlContext.baseRelationToDataFrame(LfwRelation(filePath)(sqlContext))
  }
}