package org.apache.spark.sql.sources.lfw

import org.apache.spark.sql.{SQLContext, DataFrame}

class JavaLfwContext(val sqlContext: SQLContext) {
  def lfw(filePath: String) =
    sqlContext.baseRelationToDataFrame(LfwRelation(filePath)(sqlContext))
}
